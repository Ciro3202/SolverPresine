from __future__ import annotations

from pathlib import Path

import torch

from presine.game.observation import InformationState
from presine.learning.deep_cfr.encoding import (
    action_probabilities_from_logits,
    encode_information_state,
)
from presine.learning.deep_cfr.network import StrategyNetwork
from presine.learning.multiplayer import multiplayer_encoding
from presine.policies.base import Policy


class NeuralPolicy:
    def __init__(
        self,
        networks: tuple[StrategyNetwork, ...],
        *,
        device: str = "cpu",
        regret_matching: bool = False,
    ) -> None:
        self.networks = networks
        self.device = torch.device(device)
        self.regret_matching = regret_matching
        for network in self.networks:
            network.to(self.device).eval()

    @torch.no_grad()
    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        encoder = (
            encode_information_state
            if len(info.bids) == 2
            else multiplayer_encoding.encode_information_state
        )
        state = torch.from_numpy(encoder(info)).to(self.device).unsqueeze(0)
        logits = self.networks[info.player](state)[0].cpu().numpy()
        return action_probabilities_from_logits(info, logits, regret_matching=self.regret_matching)


class PolicyBundle:
    def __init__(self, policies: dict[int, Policy]) -> None:
        self.policies = policies

    def for_hand_size(self, hand_size: int) -> Policy:
        try:
            return self.policies[hand_size]
        except KeyError as error:
            raise KeyError(f"bundle has no policy for hand size {hand_size}") from error

    @classmethod
    def load(cls, root: Path, *, device: str = "cpu") -> PolicyBundle:
        from presine.learning.checkpoint import load_policy

        policies = {}
        for hand_size in range(1, 6):
            path = root / f"round_{hand_size}" / "policy.pt"
            if path.exists():
                config, policy = load_policy(path, device=device)
                if config.hand_size != hand_size:
                    raise ValueError(f"checkpoint {path} has inconsistent hand size")
                if config.starting_player != (5 - hand_size) % 2:
                    raise ValueError(f"checkpoint {path} has the wrong starting player")
                policies[hand_size] = policy
        if not policies:
            raise FileNotFoundError(f"no round checkpoints found below {root}")
        return cls(policies)


class MultiplayerPolicyBundle(PolicyBundle):
    @classmethod
    def load(cls, root: Path, *, device: str = "cpu", players: int = 4) -> MultiplayerPolicyBundle:
        from presine.learning.checkpoint import load_policy

        policies = {}
        for hand_size in range(1, 6):
            path = root / f"round_{hand_size}" / "policy.pt"
            if not path.exists():
                continue
            config, policy = load_policy(path, device=device)
            if config.players != players or config.hand_size != hand_size:
                raise ValueError(
                    f"checkpoint {path} is not a {players}-player round {hand_size} policy"
                )
            if config.starting_player != (5 - hand_size) % players:
                raise ValueError(f"checkpoint {path} has the wrong starting player")
            policies[hand_size] = policy
        if len(policies) != 5:
            missing = sorted(set(range(1, 6)) - set(policies))
            raise FileNotFoundError(f"multiplayer bundle is incomplete; missing rounds {missing}")
        return cls(policies)
