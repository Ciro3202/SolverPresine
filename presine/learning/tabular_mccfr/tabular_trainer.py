from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import TypeAlias

from presine.game.cards import NUM_CARDS
from presine.game.config import GameConfig
from presine.game.state import Phase, RoundState
from presine.policies.tabular import InformationKey, PolicyEntry, TabularPolicy

from ..checkpoint import (
    load_tabular_training_payload,
    save_tabular_policy_checkpoint,
    save_tabular_training_checkpoint,
)
from ..exact_evaluation import exact_nashconv
from ..tabular_config import TabularCFRConfig

SnapshotEntry: TypeAlias = tuple[tuple[int, ...], tuple[float, ...]]
Snapshot: TypeAlias = dict[InformationKey, SnapshotEntry]
Deal: TypeAlias = tuple[tuple[int, ...], ...]


@dataclass(slots=True)
class TabularNode:
    actions: tuple[int, ...]
    regrets: list[float]
    strategy_sum: list[float]
    # The own-player reach contribution is identical for all hidden opponent
    # deals in an information set. This marker lets the parent accept it once
    # per synchronous iteration, without building a second 15M-key reducer.
    average_iteration: int = 0


@dataclass(frozen=True, slots=True)
class TabularIterationMetrics:
    iteration: int
    elapsed_seconds: float
    deals: int
    nodes_visited: int
    information_states: int
    regret_bound: tuple[float, float]
    nashconv_upper_bound: float
    nashconv_exact: float | None = None
    policy_change: float | None = None
    convergence_stable_checks: int = 0
    converged: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "deals": self.deals,
            "nodes_visited": self.nodes_visited,
            "nodes_per_second": self.nodes_visited / self.elapsed_seconds
            if self.elapsed_seconds
            else 0.0,
            "information_states": self.information_states,
            "regret_bound": self.regret_bound,
            "nashconv_upper_bound": self.nashconv_upper_bound,
            "nashconv_exact": self.nashconv_exact,
            "policy_change": self.policy_change,
            "convergence_stable_checks": self.convergence_stable_checks,
            "converged": self.converged,
        }


@dataclass(slots=True)
class _WorkerResult:
    deals: int
    nodes_visited: int
    regrets: dict[InformationKey, tuple[tuple[int, ...], list[float]]]
    average: dict[InformationKey, tuple[tuple[int, ...], list[float]]]


@dataclass(slots=True)
class _StaticNode:
    """Immutable decision tree node used by the tiny one-card game."""

    key: InformationKey | None
    actions: tuple[int, ...]
    children: tuple[_StaticNode, ...]
    utility: float | None


_WORKER_GAME: GameConfig | None = None
_WORKER_SNAPSHOT: Snapshot | dict[InformationKey, TabularNode] | None = None
_WORKER_DEAL_BLOCKS: tuple[tuple[Deal, ...], ...] | None = None


def _initialize_worker(
    game: GameConfig,
    snapshot: Snapshot | dict[InformationKey, TabularNode],
    deal_blocks: tuple[tuple[Deal, ...], ...] | None = None,
) -> None:
    global _WORKER_GAME, _WORKER_SNAPSHOT, _WORKER_DEAL_BLOCKS
    _WORKER_GAME = game
    _WORKER_SNAPSHOT = snapshot
    _WORKER_DEAL_BLOCKS = deal_blocks


def _regret_matching(actions: tuple[int, ...], regrets: tuple[float, ...]) -> tuple[float, ...]:
    positive = [max(0.0, value) for value in regrets]
    total = sum(positive)
    if total <= 1e-15:
        probability = 1.0 / len(actions)
        return (probability,) * len(actions)
    return tuple(value / total for value in positive)


def _stored_actions_regrets(
    entry: SnapshotEntry | TabularNode,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if isinstance(entry, TabularNode):
        return entry.actions, tuple(entry.regrets)
    return entry


class _Traversal:
    def __init__(self, snapshot: Snapshot | dict[InformationKey, TabularNode]) -> None:
        self.snapshot = snapshot
        self.regrets: dict[InformationKey, tuple[tuple[int, ...], list[float]]] = {}
        self.average: dict[InformationKey, tuple[tuple[int, ...], list[float]]] = {}
        self.strategy_cache: dict[InformationKey, tuple[float, ...]] = {}
        self.nodes_visited = 0

    def strategy(self, key: InformationKey, actions: tuple[int, ...]) -> tuple[float, ...]:
        cached = self.strategy_cache.get(key)
        if cached is not None:
            return cached
        entry = self.snapshot.get(key)
        if entry is None:
            regrets = (0.0,) * len(actions)
        else:
            stored_actions, regrets = _stored_actions_regrets(entry)
            if stored_actions != actions:
                raise ValueError("legal actions changed for a tabular information state")
        strategy = _regret_matching(actions, regrets)
        self.strategy_cache[key] = strategy
        return strategy

    def cfr(self, state: RoundState, reach0: float, reach1: float) -> float:
        self.nodes_visited += 1
        if state.phase == Phase.TERMINAL:
            return state.utilities()[0]
        if state.phase == Phase.CHANCE:
            raise AssertionError("chance must be applied before the CFR traversal")

        actor = state.current_player
        info = state.information_state()
        key = info.key()
        actions = info.legal_actions
        strategy = self.strategy(key, actions)
        own_reach = reach0 if actor == 0 else reach1
        if key not in self.average:
            self.average[key] = (
                actions,
                [own_reach * probability for probability in strategy],
            )

        action_values: list[float] = []
        use_fast_undo = state.config.hand_size == 2
        for action, probability in zip(actions, strategy):
            token = state.apply_action_fast(action) if use_fast_undo else state.apply_action(action)
            if actor == 0:
                value = self.cfr(state, reach0 * probability, reach1)
            else:
                value = self.cfr(state, reach0, reach1 * probability)
            if use_fast_undo:
                state.undo_fast(token)
            else:
                state.undo(token)
            action_values.append(value)
        node_value = sum(probability * value for probability, value in zip(strategy, action_values))

        entry = self.regrets.get(key)
        if entry is None:
            deltas = [0.0] * len(actions)
            self.regrets[key] = (actions, deltas)
        else:
            stored_actions, deltas = entry
            if stored_actions != actions:
                raise ValueError("legal actions changed during regret accumulation")
        opponent_reach = reach1 if actor == 0 else reach0
        actor_node_value = node_value if actor == 0 else -node_value
        for index, value in enumerate(action_values):
            actor_action_value = value if actor == 0 else -value
            deltas[index] += opponent_reach * (actor_action_value - actor_node_value)
        return node_value


class _StaticTraversal:
    """CFR traversal over pre-built one-card trees."""

    def __init__(self, snapshot: Snapshot | dict[InformationKey, TabularNode]) -> None:
        self.snapshot = snapshot
        self.regrets: dict[InformationKey, tuple[tuple[int, ...], list[float]]] = {}
        self.average: dict[InformationKey, tuple[tuple[int, ...], list[float]]] = {}
        self.strategy_cache: dict[InformationKey, tuple[float, ...]] = {}
        self.nodes_visited = 0

    def strategy(self, key: InformationKey, actions: tuple[int, ...]) -> tuple[float, ...]:
        cached = self.strategy_cache.get(key)
        if cached is not None:
            return cached
        entry = self.snapshot.get(key)
        if entry is None:
            regrets = (0.0,) * len(actions)
        else:
            stored_actions, regrets = _stored_actions_regrets(entry)
            if stored_actions != actions:
                raise ValueError("legal actions changed for a static information state")
        strategy = _regret_matching(actions, regrets)
        self.strategy_cache[key] = strategy
        return strategy

    def cfr(self, node: _StaticNode, reach0: float, reach1: float) -> float:
        self.nodes_visited += 1
        if node.utility is not None:
            return node.utility
        if node.key is None:
            raise AssertionError("decision node has no information-state key")
        strategy = self.strategy(node.key, node.actions)
        actor = node.key[0]
        own_reach = reach0 if actor == 0 else reach1
        if node.key not in self.average:
            self.average[node.key] = (
                node.actions,
                [own_reach * probability for probability in strategy],
            )
        action_values = []
        for action, probability, child in zip(node.actions, strategy, node.children):
            if actor == 0:
                value = self.cfr(child, reach0 * probability, reach1)
            else:
                value = self.cfr(child, reach0, reach1 * probability)
            action_values.append(value)
        node_value = sum(probability * value for probability, value in zip(strategy, action_values))
        entry = self.regrets.get(node.key)
        if entry is None:
            deltas = [0.0] * len(node.actions)
            self.regrets[node.key] = (node.actions, deltas)
        else:
            stored_actions, deltas = entry
            if stored_actions != node.actions:
                raise ValueError("legal actions changed during static regret accumulation")
        opponent_reach = reach1 if actor == 0 else reach0
        actor_node_value = node_value if actor == 0 else -node_value
        for index, value in enumerate(action_values):
            actor_action_value = value if actor == 0 else -value
            deltas[index] += opponent_reach * (actor_action_value - actor_node_value)
        return node_value


def _process_hand0_block(hand0_block: tuple[tuple[int, ...], ...]) -> _WorkerResult:
    if _WORKER_GAME is None or _WORKER_SNAPSHOT is None:
        raise RuntimeError("tabular CFR worker was not initialized")
    traversal = _Traversal(_WORKER_SNAPSHOT)
    hand_size = _WORKER_GAME.hand_size
    deals = 0
    all_cards = tuple(range(NUM_CARDS))
    for hand0 in hand0_block:
        hand0_set = set(hand0)
        remaining = tuple(card for card in all_cards if card not in hand0_set)
        for hand1 in combinations(remaining, hand_size):
            state = RoundState(_WORKER_GAME)
            # Queste carte sono create direttamente qui e la mano viene poi
            # buttata: non serve conservare una copia per tornare indietro.
            state.apply_chance_fast_unchecked((hand0, hand1))
            traversal.cfr(state, 1.0, 1.0)
            deals += 1
    return _WorkerResult(
        deals=deals,
        nodes_visited=traversal.nodes_visited,
        regrets=traversal.regrets,
        average=traversal.average,
    )


def _process_deal_block(block_index: int) -> _WorkerResult:
    if _WORKER_GAME is None or _WORKER_SNAPSHOT is None or _WORKER_DEAL_BLOCKS is None:
        raise RuntimeError("tabular CFR worker was not initialized")
    traversal = _Traversal(_WORKER_SNAPSHOT)
    deals = 0
    for deal in _WORKER_DEAL_BLOCKS[block_index]:
        state = RoundState(_WORKER_GAME)
        # Le carte sono già state elencate dal programma: evitiamo un controllo
        # e una copia che qui non servirebbero.
        state.apply_chance_fast_unchecked(deal)
        traversal.cfr(state, 1.0, 1.0)
        deals += 1
    return _WorkerResult(
        deals=deals,
        nodes_visited=traversal.nodes_visited,
        regrets=traversal.regrets,
        average=traversal.average,
    )


class TabularCFRTrainer:
    """Full-tree CFR with exact chance enumeration for one- and two-card rounds.

    Chance outcomes are partitioned by player-zero hand. Each worker traverses
    every action below its deals against the same strategy snapshot. Regret
    deltas are then reduced synchronously, so parallel execution does not alter
    the CFR update semantics.
    """

    def __init__(self, game_config: GameConfig, training_config: TabularCFRConfig) -> None:
        if game_config.players != 2:
            raise ValueError(
                "the current full-tree tabular CFR traversal supports exactly two players"
            )
        if game_config.hand_size not in (1, 2):
            raise ValueError("tabular CFR is intentionally limited to hand sizes 1 and 2")
        self.game_config = game_config
        self.training_config = training_config
        self.nodes: dict[InformationKey, TabularNode] = {}
        self.iteration = 0
        self.nodes_visited = 0
        self._one_card_trees: dict[int, tuple[_StaticNode, ...]] = {}
        self._deal_blocks: tuple[tuple[Deal, ...], ...] | None = None
        self._convergence_stable_checks = 0
        self._last_convergence_policy: dict[InformationKey, PolicyEntry] | None = None
        self.converged = False

    def _build_one_card_tree(self, deal: Deal, game_config: GameConfig) -> _StaticNode:
        state = RoundState(game_config)
        # Anche qui le carte arrivano da un elenco già corretto e controllato.
        state.apply_chance_fast_unchecked(deal)

        def build() -> _StaticNode:
            if state.phase == Phase.TERMINAL:
                return _StaticNode(None, (), (), state.utilities()[0])
            info = state.information_state()
            actions = info.legal_actions
            children: list[_StaticNode] = []
            for action in actions:
                token = state.apply_action(action)
                children.append(build())
                state.undo(token)
            return _StaticNode(info.key(), actions, tuple(children), None)

        return build()

    def _one_card_iteration(
        self, snapshot: Snapshot | dict[InformationKey, TabularNode], game_config: GameConfig
    ) -> _WorkerResult:
        trees = self._one_card_trees.get(game_config.starting_player)
        if trees is None:
            deals = tuple(
                (hand0, hand1)
                for hand0 in combinations(range(NUM_CARDS), 1)
                for hand1 in combinations(
                    tuple(card for card in range(NUM_CARDS) if card not in hand0), 1
                )
            )
            trees = tuple(self._build_one_card_tree(deal, game_config) for deal in deals)
            self._one_card_trees[game_config.starting_player] = trees
        traversal = _StaticTraversal(snapshot)
        for tree in trees:
            traversal.cfr(tree, 1.0, 1.0)
        return _WorkerResult(
            deals=len(trees),
            nodes_visited=traversal.nodes_visited,
            regrets=traversal.regrets,
            average=traversal.average,
        )

    def _build_deal_blocks(self) -> tuple[tuple[Deal, ...], ...]:
        hand_size = self.game_config.hand_size
        hands = tuple(combinations(range(NUM_CARDS), hand_size))
        worker_count = min(self.training_config.workers, len(hands))
        blocks: list[list[Deal]] = [[] for _ in range(worker_count)]
        all_cards = tuple(range(NUM_CARDS))
        for hand_index, hand0 in enumerate(hands):
            hand0_set = set(hand0)
            remaining = tuple(card for card in all_cards if card not in hand0_set)
            for hand1 in combinations(remaining, hand_size):
                blocks[hand_index % worker_count].append((hand0, hand1))
        return tuple(tuple(block) for block in blocks)

    def _snapshot(self) -> Snapshot:
        return {key: (node.actions, tuple(node.regrets)) for key, node in self.nodes.items()}

    def _hand0_blocks(self) -> list[tuple[tuple[int, ...], ...]]:
        hands = tuple(combinations(range(NUM_CARDS), self.game_config.hand_size))
        worker_count = min(self.training_config.workers, len(hands))
        return [tuple(hands[index::worker_count]) for index in range(worker_count)]

    def _run_workers(
        self, snapshot: Snapshot | dict[InformationKey, TabularNode], game_config: GameConfig
    ) -> Iterator[_WorkerResult]:
        if self.game_config.hand_size == 1:
            yield self._one_card_iteration(snapshot, game_config)
            return
        if self._deal_blocks is None:
            self._deal_blocks = self._build_deal_blocks()
        if len(self._deal_blocks) == 1:
            _initialize_worker(game_config, snapshot, self._deal_blocks)
            yield _process_deal_block(0)
            return

        # Linux/HPC uses fork so the potentially large snapshot is shared
        # copy-on-write. Spawn remains supported for local Windows validation.
        methods = mp.get_all_start_methods()
        context = mp.get_context("fork" if "fork" in methods else "spawn")
        if context.get_start_method() == "fork":
            _initialize_worker(game_config, snapshot, self._deal_blocks)
            with context.Pool(processes=len(self._deal_blocks)) as pool:
                yield from pool.imap_unordered(_process_deal_block, range(len(self._deal_blocks)))
            return
        with context.Pool(
            processes=len(self._deal_blocks),
            initializer=_initialize_worker,
            initargs=(game_config, snapshot, self._deal_blocks),
        ) as pool:
            yield from pool.imap_unordered(_process_deal_block, range(len(self._deal_blocks)))

    def train_iteration(self) -> TabularIterationMetrics:
        started = time.perf_counter()
        next_iteration = self.iteration + 1
        iteration_game = self._iteration_game_config(next_iteration)
        expected_deals = RoundState(iteration_game).chance_outcome_count()
        # On Linux the forked workers see this mapping as an immutable COW
        # snapshot. Reducing a completed worker immediately therefore keeps
        # exact synchronous CFR semantics while avoiding eight giant returned
        # dictionaries plus a duplicated parent snapshot.
        methods = mp.get_all_start_methods()
        snapshot: Snapshot | dict[InformationKey, TabularNode]
        snapshot = self.nodes if "fork" in methods else self._snapshot()
        total_deals = 0
        visited = 0
        for result in self._run_workers(snapshot, iteration_game):
            total_deals += result.deals
            visited += result.nodes_visited
            for key, (actions, deltas) in result.regrets.items():
                node = self.nodes.get(key)
                if node is None:
                    node = TabularNode(actions, [0.0] * len(actions), [0.0] * len(actions))
                    self.nodes[key] = node
                elif node.actions != actions:
                    raise ValueError("legal actions changed while reducing regrets")
                for index, delta in enumerate(deltas):
                    node.regrets[index] += delta / expected_deals
            for key, (actions, contribution) in result.average.items():
                node = self.nodes.get(key)
                if node is None:
                    node = TabularNode(actions, [0.0] * len(actions), [0.0] * len(actions))
                    self.nodes[key] = node
                elif node.actions != actions:
                    raise ValueError("legal actions changed while reducing average strategy")
                if node.average_iteration != next_iteration:
                    for index, value in enumerate(contribution):
                        node.strategy_sum[index] += value
                    node.average_iteration = next_iteration
        if total_deals != expected_deals:
            raise AssertionError(f"enumerated {total_deals} deals, expected {expected_deals}")

        self.iteration = next_iteration
        self.nodes_visited += visited
        bounds = self.regret_bounds()
        return TabularIterationMetrics(
            iteration=self.iteration,
            elapsed_seconds=time.perf_counter() - started,
            deals=total_deals,
            nodes_visited=visited,
            information_states=len(self.nodes),
            regret_bound=bounds,
            nashconv_upper_bound=bounds[0] + bounds[1],
        )

    def _iteration_game_config(self, iteration: int) -> GameConfig:
        if not self.training_config.alternate_starting_player:
            return self.game_config
        offset = (iteration - 1) % self.game_config.players
        return replace(
            self.game_config,
            starting_player=(self.game_config.starting_player + offset) % self.game_config.players,
        )

    def regret_bounds(self) -> tuple[float, float]:
        if self.iteration <= 0:
            return (math.inf, math.inf)
        totals = [0.0, 0.0]
        for key, node in self.nodes.items():
            player = int(key[0])
            totals[player] += max(0.0, max(node.regrets))
        return totals[0] / self.iteration, totals[1] / self.iteration

    def policy_entries(self) -> dict[InformationKey, PolicyEntry]:
        entries: dict[InformationKey, PolicyEntry] = {}
        for key, node in self.nodes.items():
            total = sum(node.strategy_sum)
            if total > 1e-15:
                probabilities = tuple(value / total for value in node.strategy_sum)
            else:
                probabilities = _regret_matching(node.actions, tuple(node.regrets))
            entries[key] = (node.actions, probabilities)
        return entries

    def average_policy(self) -> TabularPolicy:
        return TabularPolicy(self.policy_entries())

    def _convergence_assessment(self, report: TabularIterationMetrics) -> TabularIterationMetrics:
        """Measure and gate convergence without silently weakening a criterion."""
        config = self.training_config
        if (
            config.convergence_check_every == 0
            or report.iteration % config.convergence_check_every != 0
        ):
            return report
        needs_entries = (
            config.target_nashconv_exact is not None or config.target_policy_change is not None
        )
        entries = self.policy_entries() if needs_entries else None
        exact: float | None = None
        if config.target_nashconv_exact is not None:
            games = (self.game_config,)
            if config.alternate_starting_player:
                games = tuple(
                    replace(self.game_config, starting_player=seat)
                    for seat in range(self.game_config.players)
                )
            values: list[float] = []
            for game in games:
                try:
                    values.append(
                        exact_nashconv(game, entries, max_nodes=config.exact_evaluation_max_nodes)
                    )
                except ValueError:
                    # An oversized R2 evaluation is unavailable, never treated
                    # as a passing convergence test.
                    exact = None
                    break
            else:
                exact = sum(values) / len(values)

        policy_change: float | None = None
        if (
            config.target_policy_change is not None
            and entries is not None
            and len(entries) <= config.policy_change_max_states
        ):
            if self._last_convergence_policy is not None:
                keys = set(entries) | set(self._last_convergence_policy)
                total = 0.0
                for key in keys:
                    current = entries.get(key)
                    previous = self._last_convergence_policy.get(key)
                    if current is None or previous is None:
                        total += 1.0
                    else:
                        total += 0.5 * sum(
                            abs(left - right) for left, right in zip(current[1], previous[1])
                        )
                policy_change = total / max(1, len(keys))
            self._last_convergence_policy = entries

        conditions: list[bool] = []
        if config.target_nashconv_upper_bound is not None:
            conditions.append(report.nashconv_upper_bound <= config.target_nashconv_upper_bound)
        if config.target_nashconv_exact is not None:
            conditions.append(exact is not None and exact <= config.target_nashconv_exact)
        if config.target_policy_change is not None:
            conditions.append(
                policy_change is not None and policy_change <= config.target_policy_change
            )
        satisfied = (
            bool(conditions) and report.iteration >= config.min_iterations and all(conditions)
        )
        self._convergence_stable_checks = self._convergence_stable_checks + 1 if satisfied else 0
        self.converged = self._convergence_stable_checks >= config.stable_checks
        return replace(
            report,
            nashconv_exact=exact,
            policy_change=policy_change,
            convergence_stable_checks=self._convergence_stable_checks,
            converged=self.converged,
        )

    def train(self, output_dir: Path) -> list[TabularIterationMetrics]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / ".internal" / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        reports: list[TabularIterationMetrics] = []
        while self.iteration < self.training_config.iterations:
            report = self.train_iteration()
            report = self._convergence_assessment(report)
            reports.append(report)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
            if report.iteration % self.training_config.checkpoint_every == 0:
                self.save_training(output_dir)
            if report.converged:
                break
        self.save(output_dir)
        return reports

    def save(self, output_dir: Path) -> None:
        self.save_training(output_dir)
        self.save_policy(output_dir)

    def save_training(self, output_dir: Path) -> None:
        save_tabular_training_checkpoint(self, output_dir / "training.pt")

    def save_policy(self, output_dir: Path) -> None:
        save_tabular_policy_checkpoint(self, output_dir / "policy.pt")

    @classmethod
    def restore(cls, path: Path) -> TabularCFRTrainer:
        payload = load_tabular_training_payload(path)
        trainer = cls(
            GameConfig.from_dict(payload["game"]),
            TabularCFRConfig.from_dict(payload["tabular"]),
        )
        trainer.iteration = int(payload["iteration"])
        trainer.nodes_visited = int(payload["nodes_visited"])
        convergence = payload.get("convergence", {})
        trainer._convergence_stable_checks = int(convergence.get("stable_checks", 0))
        trainer.converged = bool(convergence.get("converged", False))
        for key, values in payload["nodes"].items():
            actions, regrets, strategy_sum = values
            trainer.nodes[key] = TabularNode(
                tuple(actions), list(regrets), list(strategy_sum), trainer.iteration
            )
        return trainer
