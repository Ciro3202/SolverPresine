"""Exact normal-form evaluation of a small heads-up tabular profile.

This has a hard node limit. It is a validator and stopping metric for R1, not
an attempt to hold R2's full game tree in RAM. A best response is selected by
backward induction over information sets, so it never observes hidden cards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

from presine.game.cards import NUM_CARDS
from presine.game.config import GameConfig
from presine.game.state import Phase, RoundState
from presine.policies.base import Policy
from presine.policies.tabular import InformationKey, PolicyEntry


@dataclass(slots=True)
class _Node:
    depth: int
    actor: int | None
    key: InformationKey | None
    actions: tuple[int, ...]
    children: tuple[_Node, ...]
    utility0: float | None
    reach: float = 0.0
    value: float = 0.0


@dataclass(frozen=True, slots=True)
class ExactNashConvReport:
    """Exact two-player response values for one fixed policy profile."""

    deals: int
    tree_nodes: int
    information_states: int
    on_policy_values: tuple[float, float]
    best_response_values: tuple[float, float]
    unilateral_gains: tuple[float, float]
    nashconv: float
    exploitability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "deals": self.deals,
            "tree_nodes": self.tree_nodes,
            "information_states": self.information_states,
            "on_policy_values": self.on_policy_values,
            "best_response_values": self.best_response_values,
            "unilateral_gains": self.unilateral_gains,
            "nashconv": self.nashconv,
            "exploitability": self.exploitability,
        }


def _deals(game: GameConfig):
    cards = tuple(range(NUM_CARDS))
    for hand0 in combinations(cards, game.hand_size):
        remaining = tuple(card for card in cards if card not in hand0)
        for hand1 in combinations(remaining, game.hand_size):
            yield (hand0, hand1)


def _distribution(
    entries: dict[InformationKey, PolicyEntry], key: InformationKey, actions: tuple[int, ...]
) -> tuple[float, ...]:
    entry = entries.get(key)
    if entry is None:
        return (1.0 / len(actions),) * len(actions)
    stored_actions, probabilities = entry
    if stored_actions != actions:
        raise ValueError("policy action order differs from the game")
    return probabilities


def materialize_policy_entries(
    game: GameConfig,
    policy: Policy,
    *,
    max_nodes: int,
    verify_repeated_probabilities: bool = True,
    progress_every: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[InformationKey, PolicyEntry], int]:
    """Evaluate a policy on every HU information state without hidden-card access.

    Multiple underlying deals can map to the same information state. Requiring
    the returned distribution to agree across all of them is an explicit guard
    against a state-aware policy accidentally depending on hidden cards.
    """

    if game.players != 2:
        raise ValueError("exact policy materialization currently supports heads-up only")
    entries: dict[InformationKey, PolicyEntry] = {}
    count = 0

    def visit(state: RoundState) -> None:
        nonlocal count
        count += 1
        if count > max_nodes:
            raise ValueError(
                f"policy materialization needs more than {max_nodes:,} nodes; disabled for this run"
            )
        if progress is not None and progress_every > 0 and count % progress_every == 0:
            progress(count, len(entries))
        if state.phase == Phase.TERMINAL:
            return
        info = state.information_state()
        key = info.key()
        previous = entries.get(key)
        if previous is None or verify_repeated_probabilities:
            probabilities = tuple(float(value) for value in policy.probabilities(info))
            if len(probabilities) != len(info.legal_actions):
                raise ValueError("policy returned the wrong probability vector")
            if any(value < 0.0 for value in probabilities):
                raise ValueError("policy returned a negative probability")
            if abs(sum(probabilities) - 1.0) > 1e-9:
                raise ValueError("policy probabilities do not sum to one")
            entry = (info.legal_actions, probabilities)
            if previous is None:
                entries[key] = entry
            elif previous != entry:
                raise ValueError("policy changes within one information set")
        for action in info.legal_actions:
            token = (
                state.apply_action_fast(action)
                if game.hand_size > 1
                else state.apply_action(action)
            )
            visit(state)
            if game.hand_size > 1:
                state.undo_fast(token)
            else:
                state.undo(token)

    for deal in _deals(game):
        state = RoundState(game)
        state.apply_chance(deal)
        visit(state)
    return entries, count


def exact_nashconv_report(
    game: GameConfig,
    entries: dict[InformationKey, PolicyEntry],
    *,
    max_nodes: int,
    progress_every: int = 0,
    progress: Callable[[int], None] | None = None,
) -> ExactNashConvReport:
    """Return exact HU profile and unilateral best-response values."""

    if game.players != 2:
        raise ValueError("exact NashConv evaluator currently supports heads-up only")
    count = 0

    def build(state: RoundState, depth: int) -> _Node:
        nonlocal count
        count += 1
        if count > max_nodes:
            raise ValueError(
                f"exact evaluation needs more than {max_nodes:,} nodes; disabled for this run"
            )
        if progress is not None and progress_every > 0 and count % progress_every == 0:
            progress(count)
        if state.phase == Phase.TERMINAL:
            return _Node(depth, None, None, (), (), state.utilities()[0])
        info = state.information_state()
        children: list[_Node] = []
        for action in info.legal_actions:
            token = (
                state.apply_action_fast(action)
                if game.hand_size > 1
                else state.apply_action(action)
            )
            children.append(build(state, depth + 1))
            if game.hand_size > 1:
                state.undo_fast(token)
            else:
                state.undo(token)
        return _Node(depth, info.player, info.key(), info.legal_actions, tuple(children), None)

    roots: list[_Node] = []
    for deal in _deals(game):
        state = RoundState(game)
        state.apply_chance(deal)
        roots.append(build(state, 0))

    chance = 1.0 / len(roots)

    def profile_value0(node: _Node) -> float:
        if node.utility0 is not None:
            return node.utility0
        assert node.key is not None
        probabilities = _distribution(entries, node.key, node.actions)
        return sum(
            probability * profile_value0(child)
            for probability, child in zip(probabilities, node.children)
        )

    on_policy0 = chance * sum(profile_value0(root) for root in roots)

    def best_response(target: int) -> float:
        by_depth: dict[int, list[_Node]] = {}

        def forward(node: _Node, reach: float) -> None:
            node.reach = reach
            if node.utility0 is not None:
                node.value = node.utility0 if target == 0 else -node.utility0
                return
            assert node.actor is not None and node.key is not None
            probabilities = _distribution(entries, node.key, node.actions)
            for index, child in enumerate(node.children):
                child_reach = reach if node.actor == target else reach * probabilities[index]
                forward(child, child_reach)
            by_depth.setdefault(node.depth, []).append(node)

        for root in roots:
            forward(root, chance)
        for depth in sorted(by_depth, reverse=True):
            groups: dict[InformationKey, list[_Node]] = {}
            for node in by_depth[depth]:
                assert node.actor is not None and node.key is not None
                if node.actor != target:
                    probabilities = _distribution(entries, node.key, node.actions)
                    node.value = sum(
                        probability * child.value
                        for probability, child in zip(probabilities, node.children)
                    )
                else:
                    groups.setdefault(node.key, []).append(node)
            for nodes in groups.values():
                scores = [
                    sum(node.reach * node.children[index].value for node in nodes)
                    for index in range(len(nodes[0].actions))
                ]
                selected = max(range(len(scores)), key=scores.__getitem__)
                for node in nodes:
                    node.value = node.children[selected].value
        return chance * sum(root.value for root in roots)

    on_policy = (on_policy0, -on_policy0)
    response_values = (best_response(0), best_response(1))
    gains = tuple(response_values[seat] - on_policy[seat] for seat in (0, 1))
    nashconv = sum(gains)
    return ExactNashConvReport(
        deals=len(roots),
        tree_nodes=count,
        information_states=len(entries),
        on_policy_values=on_policy,
        best_response_values=response_values,
        unilateral_gains=(gains[0], gains[1]),
        nashconv=nashconv,
        exploitability=nashconv / 2.0,
    )


def exact_nashconv(
    game: GameConfig,
    entries: dict[InformationKey, PolicyEntry],
    *,
    max_nodes: int,
) -> float:
    """Return exact NashConv for a HU profile, or reject an oversized tree."""
    return exact_nashconv_report(game, entries, max_nodes=max_nodes).nashconv
