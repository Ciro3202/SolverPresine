from __future__ import annotations

from presine.game.observation import InformationState


class RandomPolicy:
    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        probability = 1.0 / len(info.legal_actions)
        return (probability,) * len(info.legal_actions)
