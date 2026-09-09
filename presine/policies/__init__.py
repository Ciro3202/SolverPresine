"""Deployable strategies sharing the ``Policy.probabilities(info)`` contract.

``base`` defines the contract; ``heuristic`` and ``random`` are baselines;
``tabular``, ``neural`` and ``linear`` are learned policies; ``resolving``
wraps a base policy with optional online search. Neural dependencies are lazy.
"""

from typing import Any

__all__ = [
    "ClosedBlindRoundPolicy",
    "HeuristicPolicy",
    "NeuralPolicy",
    "Policy",
    "PolicyBundle",
    "RandomPolicy",
]


def __getattr__(name: str) -> Any:
    if name == "Policy":
        from .base import Policy

        return Policy
    if name == "HeuristicPolicy":
        from .heuristic import HeuristicPolicy

        return HeuristicPolicy
    if name == "ClosedBlindRoundPolicy":
        from .blind_round import ClosedBlindRoundPolicy

        return ClosedBlindRoundPolicy
    if name in {"NeuralPolicy", "PolicyBundle"}:
        from .neural import NeuralPolicy, PolicyBundle

        return {"NeuralPolicy": NeuralPolicy, "PolicyBundle": PolicyBundle}[name]
    if name == "RandomPolicy":
        from .random import RandomPolicy

        return RandomPolicy
    raise AttributeError(name)
