from __future__ import annotations

"""Small shared policy/value blueprint for three- to six-player Presine.

The network has one shared representation and one policy/value head per
round.  It stores no information-state table and is therefore bounded by the
chosen hidden sizes rather than by the number of states visited in training.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from presine.game.observation import InformationState
from presine.learning.deep_cfr.encoding import ACTION_DIM, action_id
from presine.learning.multiplayer.multiplayer_encoding import (
    encode_information_state,
    encoding_version,
    state_dim,
)

NEURAL_BUNDLE_FORMAT = "presine-neural-multiplayer-blueprint-v1"


@dataclass(frozen=True, slots=True)
class NeuralBlueprintArchitecture:
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> NeuralBlueprintArchitecture:
        raw = dict(values)
        if "hidden_sizes" in raw:
            raw["hidden_sizes"] = tuple(raw["hidden_sizes"])  # type: ignore[arg-type]
        return cls(**raw)


class SharedRoundPolicyValueNetwork(nn.Module):
    """Shared trunk with independent R1..R5 policy and relative-value heads."""

    def __init__(
        self,
        players: int,
        architecture: NeuralBlueprintArchitecture | None = None,
    ) -> None:
        super().__init__()
        if not 3 <= players <= 6:
            raise ValueError("neural multiplayer blueprints require 3..6 players")
        self.players = players
        self.architecture = architecture or NeuralBlueprintArchitecture()
        self.input_dim = state_dim(players)
        layers: list[nn.Module] = []
        incoming = self.input_dim
        for size in self.architecture.hidden_sizes:
            layers.extend((nn.Linear(incoming, size), nn.LayerNorm(size), nn.GELU()))
            if self.architecture.dropout:
                layers.append(nn.Dropout(self.architecture.dropout))
            incoming = size
        self.trunk = nn.Sequential(*layers)
        self.policy_heads = nn.ModuleDict(
            {str(hand_size): nn.Linear(incoming, ACTION_DIM) for hand_size in range(1, 6)}
        )
        self.value_heads = nn.ModuleDict(
            {str(hand_size): nn.Linear(incoming, players) for hand_size in range(1, 6)}
        )

    def forward(self, states: torch.Tensor, hand_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= hand_size <= 5:
            raise ValueError("hand_size must be in 1..5")
        hidden = self.trunk(states)
        policy = self.policy_heads[str(hand_size)](hidden)
        # Round utilities are normalized to roughly [-1, 1].
        values = torch.tanh(self.value_heads[str(hand_size)](hidden))
        return policy, values

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class NeuralMultiplayerBlueprint:
    """Deployable policy around a shared five-round neural network."""

    def __init__(
        self,
        players: int,
        network: SharedRoundPolicyValueNetwork,
        *,
        device: str = "cpu",
        metadata: dict[str, object] | None = None,
    ) -> None:
        if network.players != players:
            raise ValueError("network and policy player counts differ")
        self.players = players
        self.network = network
        self.device = torch.device(device)
        self.metadata = dict(metadata or {})
        self.network.to(self.device).eval()

    @torch.inference_mode()
    def _outputs(self, info: InformationState) -> tuple[np.ndarray, np.ndarray]:
        if len(info.bids) != self.players:
            raise ValueError("information state has the wrong player count")
        encoded = torch.from_numpy(encode_information_state(info)).unsqueeze(0)
        policy, values = self.network(encoded.to(self.device), info.hand_size)
        return policy[0].cpu().numpy(), values[0].cpu().numpy()

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        logits, _ = self._outputs(info)
        ids = [action_id(info, action) for action in info.legal_actions]
        legal = np.asarray([logits[index] for index in ids], dtype=np.float64)
        legal -= float(legal.max())
        probabilities = np.exp(np.clip(legal, -60.0, 60.0))
        probabilities /= float(probabilities.sum())
        return tuple(float(value) for value in probabilities)

    def lookup(self, info: InformationState) -> tuple[float, ...]:
        return self.probabilities(info)

    def relative_values(self, info: InformationState) -> tuple[float, ...]:
        """Return utilities ordered actor, next clockwise seat, and so on."""
        _, values = self._outputs(info)
        return tuple(float(value) for value in values)

    def for_hand_size(self, hand_size: int) -> NeuralMultiplayerBlueprint:
        if not 1 <= hand_size <= 5:
            raise ValueError("hand_size must be in 1..5")
        return self

    def stats(self) -> dict[str, object]:
        parameter_count = self.network.parameter_count
        return {
            "kind": NEURAL_BUNDLE_FORMAT,
            "players": self.players,
            "rounds": (1, 2, 3, 4, 5),
            "encoding": encoding_version(self.players),
            "parameters": parameter_count,
            "fp32_parameter_bytes": parameter_count * 4,
            "stored_state_rows": 0,
            "architecture": asdict(self.network.architecture),
        }


def make_blueprint(
    players: int,
    architecture: NeuralBlueprintArchitecture | None = None,
    *,
    device: str = "cpu",
    seed: int = 20260826,
) -> NeuralMultiplayerBlueprint:
    torch.manual_seed(seed)
    network = SharedRoundPolicyValueNetwork(players, architecture)
    return NeuralMultiplayerBlueprint(players, network, device=device)


def save_neural_multiplayer_blueprint(
    policy: NeuralMultiplayerBlueprint,
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, Any] = {
        "format": NEURAL_BUNDLE_FORMAT,
        "players": policy.players,
        "encoding": encoding_version(policy.players),
        "architecture": asdict(policy.network.architecture),
        "state_dict": {
            key: value.detach().cpu() for key, value in policy.network.state_dict().items()
        },
        "metadata": dict(metadata or policy.metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)
    result = policy.stats()
    result.update({"path": str(path.resolve()), "file_bytes": path.stat().st_size})
    return result


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    # Explicit weights_only=False keeps compatibility with torch 2.3 while the
    # artifact itself remains a plain dict of tensors and primitive metadata.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - torch < 2.0 compatibility
        return torch.load(path, map_location=device)


def load_neural_multiplayer_blueprint(
    path: Path, *, device: str = "cpu", players: int | None = None
) -> NeuralMultiplayerBlueprint:
    payload = _torch_load(path, device)
    if payload.get("format") != NEURAL_BUNDLE_FORMAT:
        raise ValueError("unsupported neural multiplayer blueprint format")
    stored_players = int(payload["players"])
    if players is not None and players != stored_players:
        raise ValueError(f"blueprint is for {stored_players} players, not the requested {players}")
    expected_encoding = encoding_version(stored_players)
    if payload.get("encoding") != expected_encoding:
        raise ValueError(
            f"blueprint encoding {payload.get('encoding')!r} is incompatible with {expected_encoding!r}"
        )
    architecture = NeuralBlueprintArchitecture.from_dict(dict(payload["architecture"]))
    network = SharedRoundPolicyValueNetwork(stored_players, architecture)
    network.load_state_dict(payload["state_dict"])
    return NeuralMultiplayerBlueprint(
        stored_players,
        network,
        device=device,
        metadata=dict(payload.get("metadata", {})),
    )
