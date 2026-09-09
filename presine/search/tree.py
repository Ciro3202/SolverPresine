from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .protocols import Action


@dataclass(slots=True)
class EdgeStats:
    visits: int
    availability: int
    value_sum: list[float]

    @classmethod
    def empty(cls, players: int) -> EdgeStats:
        return cls(0, 0, [0.0] * players)


@dataclass(slots=True)
class TreeNode:
    players: int
    visits: int = 0
    edges: dict[Action, EdgeStats] = field(default_factory=dict)

    def mark_available(self, actions: tuple[Action, ...]) -> None:
        for action in actions:
            edge = self.edges.get(action)
            if edge is not None:
                edge.availability += 1

    def expand(self, action: Action) -> EdgeStats:
        edge = EdgeStats.empty(self.players)
        edge.availability = 1
        self.edges[action] = edge
        return edge

    def select(
        self,
        actions: tuple[Action, ...],
        player: int,
        exploration: float,
        rng: random.Random,
    ) -> Action:
        best_score = -math.inf
        best: list[Action] = []
        for action in actions:
            edge = self.edges[action]
            mean = edge.value_sum[player] / edge.visits
            bonus = exploration * math.sqrt(math.log(max(2, edge.availability)) / edge.visits)
            score = mean + bonus
            if score > best_score + 1e-15:
                best_score = score
                best = [action]
            elif abs(score - best_score) <= 1e-15:
                best.append(action)
        return rng.choice(best)
