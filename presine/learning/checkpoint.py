from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from presine.game.config import GameConfig

from .config import NetworkConfig
from .deep_cfr.encoding import ACTION_DIM, ENCODING_VERSION, STATE_DIM
from .multiplayer.multiplayer_encoding import ENCODING_VERSION as MULTIPLAYER_ENCODING_VERSION
from .multiplayer.multiplayer_encoding import STATE_DIM as MULTIPLAYER_STATE_DIM

if TYPE_CHECKING:
    from .deep_cfr.network import StrategyNetwork
    from .deep_cfr.trainer import DeepCFRTrainer
    from .tabular_mccfr.mccfr_trainer import MCCFRTrainer
    from .tabular_mccfr.tabular_trainer import TabularCFRTrainer


CHECKPOINT_FORMAT = 1
HYBRID_FALLBACK_FORMAT = 1
HYBRID_FALLBACK_KIND = "sampled-tabular-neural-fallback-policy"


class _LazyTorch:
    """Keep pure tabular policy loading independent from PyTorch imports."""

    def __getattr__(self, name: str):
        import importlib

        return getattr(importlib.import_module("torch"), name)


torch = _LazyTorch()


def _network(
    config: NetworkConfig,
    *,
    input_dim: int = STATE_DIM,
    output_dim: int = ACTION_DIM,
) -> StrategyNetwork:
    from .deep_cfr.network import StrategyNetwork

    return StrategyNetwork(
        config.hidden_sizes,
        dropout=config.dropout,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(trainer: DeepCFRTrainer, path: Path) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "deep-cfr-training",
        "encoding": trainer.encoding_version,
        "game": trainer.game_config.to_dict(),
        "training": trainer.training_config.to_dict(),
        "iteration": trainer.iteration,
        "nodes_visited": trainer.nodes_visited,
        "diagnostics": {
            "regret_queries": tuple(trainer.regret_queries),
            "uniform_fallbacks": tuple(trainer.uniform_fallbacks),
        },
        "rng_state": trainer.rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "advantage_networks": [network.state_dict() for network in trainer.advantage_networks],
        "policy_networks": [network.state_dict() for network in trainer.policy_networks],
        "advantage_memories": [memory.state_dict() for memory in trainer.advantage_memories],
        "policy_memories": [memory.state_dict() for memory in trainer.policy_memories],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    atomic_torch_save(payload, path)


def save_policy_checkpoint(trainer: DeepCFRTrainer, path: Path) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "deep-cfr-average-policy",
        "encoding": trainer.encoding_version,
        "game": trainer.game_config.to_dict(),
        "network": trainer.training_config.network.__dict__
        if hasattr(trainer.training_config.network, "__dict__")
        else {
            "hidden_sizes": list(trainer.training_config.network.hidden_sizes),
            "dropout": trainer.training_config.network.dropout,
        },
        "state_dim": trainer.state_dim,
        "action_dim": ACTION_DIM,
        "iteration": trainer.iteration,
        "policy_networks": [network.state_dict() for network in trainer.policy_networks],
    }
    atomic_torch_save(payload, path)


def load_policy_checkpoint(
    path: Path, *, device: str = "cpu"
) -> tuple[GameConfig, tuple[StrategyNetwork, ...]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("kind") != "deep-cfr-average-policy"
    ):
        raise ValueError(f"unsupported policy checkpoint: {path}")
    game = GameConfig.from_dict(payload["game"])
    expected_encoding = ENCODING_VERSION if game.players == 2 else MULTIPLAYER_ENCODING_VERSION
    if payload.get("encoding") != expected_encoding:
        raise ValueError(f"checkpoint uses a different encoder: {path}")
    network_config = NetworkConfig.from_dict(payload["network"])
    default_state_dim = STATE_DIM if game.players == 2 else MULTIPLAYER_STATE_DIM
    networks = tuple(
        _network(
            network_config,
            input_dim=int(payload.get("state_dim", default_state_dim)),
            output_dim=int(payload.get("action_dim", ACTION_DIM)),
        )
        for _ in range(game.players)
    )
    for network, state in zip(networks, payload["policy_networks"]):
        network.load_state_dict(state)
        network.to(device).eval()
    return game, networks


def load_training_payload(path: Path, *, device: str = "cpu") -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("kind") != "deep-cfr-training":
        raise ValueError(f"unsupported training checkpoint: {path}")
    game = GameConfig.from_dict(payload["game"])
    expected_encoding = ENCODING_VERSION if game.players == 2 else MULTIPLAYER_ENCODING_VERSION
    if payload.get("encoding") != expected_encoding:
        raise ValueError(f"checkpoint uses a different encoder: {path}")
    return payload


def save_tabular_training_checkpoint(trainer: TabularCFRTrainer, path: Path) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "tabular-cfr-training",
        "game": trainer.game_config.to_dict(),
        "tabular": trainer.training_config.to_dict(),
        "iteration": trainer.iteration,
        "nodes_visited": trainer.nodes_visited,
        "convergence": {
            "stable_checks": trainer._convergence_stable_checks,
            "converged": trainer.converged,
        },
        "nodes": {
            key: (node.actions, tuple(node.regrets), tuple(node.strategy_sum))
            for key, node in trainer.nodes.items()
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    atomic_torch_save(payload, path)


def save_tabular_policy_checkpoint(trainer: TabularCFRTrainer, path: Path) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "tabular-cfr-average-policy",
        "game": trainer.game_config.to_dict(),
        "iteration": trainer.iteration,
        "entries": trainer.policy_entries(),
        "nashconv_upper_bound": sum(trainer.regret_bounds()),
    }
    atomic_torch_save(payload, path)


def load_tabular_training_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("kind") != "tabular-cfr-training":
        raise ValueError(f"unsupported tabular training checkpoint: {path}")
    return payload


def save_mccfr_training_checkpoint(trainer: MCCFRTrainer, path: Path) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "mccfr-plus-training",
        "game": trainer.game_config.to_dict(),
        "mccfr": trainer.training_config.to_dict(),
        "iteration": trainer.iteration,
        "nodes_visited": trainer.nodes_visited,
        "rng_state": trainer.rng.getstate(),
        "chance_pool": trainer.chance_pool,
        "network_fitted": trainer.network_fitted,
        "network": {
            "hidden_sizes": list(trainer.training_config.network.hidden_sizes),
            "dropout": trainer.training_config.network.dropout,
        },
        "networks": [network.state_dict() for network in trainer.networks]
        if trainer.networks
        else None,
        "nodes": {
            key: (node.actions, tuple(node.regrets), tuple(node.strategy_sum))
            for key, node in trainer.nodes.items()
        },
    }
    atomic_torch_save(payload, path)


def save_mccfr_policy_checkpoint(trainer: MCCFRTrainer, path: Path) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "mccfr-plus-average-policy",
        "game": trainer.game_config.to_dict(),
        "mccfr": trainer.training_config.to_dict(),
        "iteration": trainer.iteration,
        "entries": trainer.policy_entries(),
        "regret_bound_sum": sum(trainer.regret_bounds()),
        "warning": "sampled MCCFR+ regret bounds are not an exact NashConv certificate",
        "network": {
            "hidden_sizes": list(trainer.training_config.network.hidden_sizes),
            "dropout": trainer.training_config.network.dropout,
        },
        "networks": [network.state_dict() for network in trainer.networks]
        if trainer.networks
        else None,
        "network_fitted": trainer.network_fitted,
    }
    atomic_torch_save(payload, path)


def save_hybrid_fallback_checkpoint(
    path: Path,
    *,
    source_policy: Path,
    game: GameConfig,
    network_config: NetworkConfig,
    networks: tuple[StrategyNetwork, StrategyNetwork],
    neural_fallback_phases: tuple[int, ...],
    training_report: dict[str, object],
) -> None:
    """Save a small derived policy; the source tabular policy stays untouched."""
    payload = {
        "format": HYBRID_FALLBACK_FORMAT,
        "kind": HYBRID_FALLBACK_KIND,
        "encoding": ENCODING_VERSION,
        "game": game.to_dict(),
        "source_policy": str(source_policy.resolve()),
        "network": {
            "hidden_sizes": list(network_config.hidden_sizes),
            "dropout": network_config.dropout,
        },
        "networks": [
            {name: value.detach().cpu() for name, value in network.state_dict().items()}
            for network in networks
        ],
        "neural_fallback_phases": list(neural_fallback_phases),
        "training_report": training_report,
    }
    atomic_torch_save(payload, path)


def load_mccfr_training_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("kind") != "mccfr-plus-training":
        raise ValueError(f"unsupported MCCFR training checkpoint: {path}")
    return payload


def load_policy(path: Path, *, device: str = "cpu"):
    """Load either a neural or tabular policy checkpoint."""
    if path.is_dir() or path.name == "manifest.json":
        from .checkpoint_deployment.packed_policy import load_packed_policy

        return load_packed_policy(path)
    from .checkpoint_deployment.sampled_checkpoint import is_sampled_checkpoint, load_sampled_policy

    if is_sampled_checkpoint(path):
        return load_sampled_policy(path)

    payload = torch.load(path, map_location=device, weights_only=False)
    kind = payload.get("kind")
    if kind == HYBRID_FALLBACK_KIND:
        from presine.policies.hybrid import HybridPolicy

        if payload.get("format") != HYBRID_FALLBACK_FORMAT:
            raise ValueError(f"unsupported hybrid fallback checkpoint: {path}")
        if payload.get("encoding") != ENCODING_VERSION:
            raise ValueError(f"checkpoint uses a different encoder: {path}")
        source = Path(payload["source_policy"])
        if not source.is_absolute():
            source = path.parent / source
        source_game, source_tabular = load_policy(source, device="cpu")
        game = GameConfig.from_dict(payload["game"])
        if source_game != game:
            raise ValueError("hybrid source policy game differs from its manifest")
        network_config = NetworkConfig.from_dict(payload["network"])
        networks = (_network(network_config), _network(network_config))
        for network, state in zip(networks, payload["networks"]):
            network.load_state_dict(state)
        return game, HybridPolicy(
            source_tabular.entries,
            networks,
            device=device,
            network_fitted=True,
            mode="auto",
            neural_fallback_phases=tuple(payload["neural_fallback_phases"]),
        )
    if kind == "deep-cfr-average-policy":
        from presine.policies.neural import NeuralPolicy

        game, networks = load_policy_checkpoint(path, device=device)
        return game, NeuralPolicy(networks, device=device)
    if kind == "tabular-cfr-average-policy":
        from presine.policies.tabular import TabularPolicy

        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported policy checkpoint: {path}")
        game = GameConfig.from_dict(payload["game"])
        entries = {
            key: (tuple(actions), tuple(probabilities))
            for key, (actions, probabilities) in payload["entries"].items()
        }
        return game, TabularPolicy(entries)
    if kind == "mccfr-plus-average-policy":
        from presine.policies.hybrid import HybridPolicy

        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported policy checkpoint: {path}")
        game = GameConfig.from_dict(payload["game"])
        entries = {
            key: (tuple(actions), tuple(probabilities))
            for key, (actions, probabilities) in payload["entries"].items()
        }
        networks = None
        if payload.get("networks") is not None:
            network_config = NetworkConfig.from_dict(payload.get("network", {}))
            networks = (_network(network_config), _network(network_config))
            for network, state in zip(networks, payload["networks"]):
                network.load_state_dict(state)
        return game, HybridPolicy(
            entries,
            networks,
            device=device,
            network_fitted=bool(payload.get("network_fitted", False)),
        )
    raise ValueError(f"unsupported policy checkpoint: {path}")
