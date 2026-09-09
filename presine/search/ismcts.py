from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generic, TypeVar

from .config import ISMCTSConfig
from .protocols import Action, BeliefSampler, GameAdapter
from .tree import EdgeStats, TreeNode

StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class SearchResult:
    actions: tuple[Action, ...]
    probabilities: tuple[float, ...]
    values: tuple[float, ...]
    visits: tuple[int, ...]
    simulations: int
    tree_nodes: int

    def action(self) -> Action:
        if not self.actions:
            raise ValueError("search result contains no actions")
        return max(zip(self.actions, self.probabilities), key=lambda item: item[1])[0]


def visit_probabilities(visits: list[int], temperature: float) -> tuple[float, ...]:
    if not visits:
        return ()
    if temperature == 0:
        best = max(range(len(visits)), key=visits.__getitem__)
        return tuple(1.0 if index == best else 0.0 for index in range(len(visits)))
    power = 1.0 / temperature
    scaled = [float(max(0, visit)) ** power for visit in visits]
    total = sum(scaled)
    if total <= 0:
        return tuple(1.0 / len(visits) for _ in visits)
    return tuple(value / total for value in scaled)


class ISMCTS(Generic[StateT]):
    """Root-sampling Information Set MCTS.

    The belief and returned value use the root observer.  The adapter chooses
    whose information identifies deeper nodes; Presine uses the acting
    player's information set.  Action availability is tracked across
    determinizations, and multiplayer selection uses the acting player's own
    utility component (Max-N style), avoiding a hidden two-player/zero-sum
    assumption.
    """

    def __init__(
        self,
        adapter: GameAdapter[StateT],
        belief: BeliefSampler[StateT],
        observer: int,
        config: ISMCTSConfig,
        *,
        seed: int,
    ) -> None:
        self.adapter = adapter
        self.belief = belief
        self.observer = observer
        self.config = config
        self.rng = random.Random(seed)
        self.nodes: dict[object, TreeNode] = {}

    def search(self) -> SearchResult:
        root_actions: tuple[Action, ...] | None = None
        root_key = None
        players = 0
        for _ in range(self.config.simulations):
            state = self.belief.sample_state(self.rng)
            players = self.adapter.player_count(state)
            if not 0 <= self.observer < players:
                raise ValueError("observer is out of range for determinized state")
            if root_actions is None:
                root_actions = self.adapter.legal_actions(state)
                root_key = self.adapter.information_key(state, self.observer)
                if not root_actions:
                    raise ValueError("search root has no legal actions")
            path: list[tuple[TreeNode, EdgeStats]] = []
            depth = 0
            leaf = getattr(self.adapter, "leaf_utilities", lambda _state: None)(state)
            while (
                leaf is None
                and not self.adapter.is_terminal(state)
                and depth < self.config.rollout_depth
            ):
                legal = self.adapter.legal_actions(state)
                if not legal:
                    break
                key = self.adapter.information_key(state, self.observer)
                node = self.nodes.get(key)
                if node is None:
                    if len(self.nodes) >= self.config.max_tree_nodes:
                        break
                    node = TreeNode(players)
                    self.nodes[key] = node
                node.mark_available(legal)
                unexpanded = [action for action in legal if action not in node.edges]
                if unexpanded:
                    action = self.rng.choice(unexpanded)
                    edge = node.expand(action)
                    path.append((node, edge))
                    self.adapter.apply_action(state, action)
                    depth += 1
                    break
                actor = self.adapter.current_player(state)
                action = node.select(legal, actor, self.config.exploration, self.rng)
                edge = node.edges[action]
                path.append((node, edge))
                self.adapter.apply_action(state, action)
                depth += 1

            while not self.adapter.is_terminal(state) and depth < self.config.rollout_depth:
                legal = self.adapter.legal_actions(state)
                if not legal:
                    break
                action = self.adapter.rollout_action(state, legal, self.rng)
                self.adapter.apply_action(state, action)
                depth += 1
                leaf = getattr(self.adapter, "leaf_utilities", lambda _state: None)(state)
            cutoff = None
            if leaf is None and not self.adapter.is_terminal(state):
                cutoff_method = getattr(self.adapter, "cutoff_utilities", None)
                if callable(cutoff_method):
                    cutoff = cutoff_method(state)
            utilities = (
                leaf
                if leaf is not None
                else self.adapter.utilities(state)
                if self.adapter.is_terminal(state)
                else cutoff
                if cutoff is not None
                else (0.0,) * players
            )
            for node, edge in path:
                node.visits += 1
                edge.visits += 1
                for player, utility in enumerate(utilities):
                    edge.value_sum[player] += utility

        assert root_actions is not None and root_key is not None
        root = self.nodes.get(root_key)
        if root is None:
            uniform = tuple(1.0 / len(root_actions) for _ in root_actions)
            return SearchResult(
                root_actions,
                uniform,
                (0.0,) * len(root_actions),
                (0,) * len(root_actions),
                self.config.simulations,
                len(self.nodes),
            )
        visits = [
            root.edges[action].visits if action in root.edges else 0 for action in root_actions
        ]
        values = tuple(
            root.edges[action].value_sum[self.observer] / root.edges[action].visits
            if action in root.edges and root.edges[action].visits
            else 0.0
            for action in root_actions
        )
        return SearchResult(
            root_actions,
            visit_probabilities(visits, self.config.temperature),
            values,
            tuple(visits),
            self.config.simulations,
            len(self.nodes),
        )
