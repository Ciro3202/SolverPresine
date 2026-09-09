from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from presine.game.config import GameConfig
from presine.game.state import Deal, Phase, RoundState
from presine.policies.tabular import InformationKey, PolicyEntry, TabularPolicy

from ..checkpoint_deployment.compact_store import CompactMCCFRStore, MMapCompactMCCFRStore
from ..checkpoint_deployment.sampled_checkpoint import (
    load_sampled_training_payload,
    save_sampled_policy_checkpoint,
    save_sampled_training_checkpoint,
)
from ..mccfr_config import MCCFRConfig


@dataclass(slots=True)
class MCCFRNode:
    actions: tuple[int, ...]
    regrets: list[float]
    strategy_sum: list[float]
    last_discount_iteration: int = 0


@dataclass(frozen=True, slots=True)
class MCCFRIterationMetrics:
    iteration: int
    elapsed_seconds: float
    traversal_seconds: float
    diagnostics_seconds: float
    traversals: int
    traversals_by_player: tuple[int, int]
    traversals_by_starting_player: tuple[int, int]
    chance_samples: int
    nodes_visited: int
    cumulative_nodes_visited: int
    information_states: int
    information_states_by_seat_phase: dict[str, int]
    regret_diagnostics: dict[str, float]
    regret_diagnostics_as_of_iteration: int
    estimated_table_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "record_role": (
                "periodic training telemetry; use it for throughput, table growth "
                "and stability checks, not as a policy-quality certificate"
            ),
            "iteration": self.iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "traversal_seconds": self.traversal_seconds,
            "diagnostics_seconds": self.diagnostics_seconds,
            "traversals": self.traversals,
            "traversals_by_player": self.traversals_by_player,
            "traversals_by_starting_player": self.traversals_by_starting_player,
            "chance_samples": self.chance_samples,
            "nodes_visited": self.nodes_visited,
            "cumulative_nodes_visited": self.cumulative_nodes_visited,
            "nodes_per_second": self.nodes_visited / self.elapsed_seconds
            if self.elapsed_seconds
            else 0.0,
            "traversal_nodes_per_second": (
                self.nodes_visited / self.traversal_seconds if self.traversal_seconds else 0.0
            ),
            "information_states": self.information_states,
            "information_states_by_seat_phase": self.information_states_by_seat_phase,
            "regret_diagnostics": self.regret_diagnostics,
            "regret_diagnostics_as_of_iteration": self.regret_diagnostics_as_of_iteration,
            "estimated_table_bytes": self.estimated_table_bytes,
            "coverage": {
                "stored_information_states": self.information_states,
                "by_seat_phase": self.information_states_by_seat_phase,
            },
            "warning": (
                "sampled cumulative regrets are empirical diagnostics, not a "
                "NashConv or exploitability certificate"
            ),
        }


def log_training_metrics(report: MCCFRIterationMetrics) -> None:
    """Mirror the machine-readable JSONL record with one operator-friendly line."""
    gib = report.estimated_table_bytes / (1024**3)
    rate = report.nodes_visited / report.traversal_seconds if report.traversal_seconds else 0.0
    print(
        "[sampled-tabular] metrics: "
        f"iteration={report.iteration:,}; "
        f"visited_nodes_total={report.cumulative_nodes_visited:,}; "
        f"information_states={report.information_states:,}; "
        f"estimated_mutable_table={gib:.2f} GiB; "
        f"traversal_rate={rate:,.0f} nodes/s",
        file=sys.stderr,
        flush=True,
    )


def log_training_checkpoint(
    iteration: int,
    checkpoint_seconds: float,
    checkpoint_bytes: int,
    *,
    mmap_storage: bool,
) -> None:
    """Explain what checkpoint size measures, especially for mmap-backed runs."""
    mib = checkpoint_bytes / (1024**2)
    scope = (
        "serialized manifests/shards only; mutable mmap slabs are separate"
        if mmap_storage
        else "complete serialized checkpoint files"
    )
    print(
        "[sampled-tabular] checkpoint: "
        f"iteration={iteration:,}; recoverable training state published; "
        f"write_time={checkpoint_seconds:.2f}s; files={mib:.2f} MiB "
        f"({scope})",
        file=sys.stderr,
        flush=True,
    )


def _regret_matching(
    actions: tuple[int, ...], regrets: list[float] | tuple[float, ...]
) -> tuple[float, ...]:
    positive = [max(0.0, value) for value in regrets]
    total = sum(positive)
    if total <= 1e-15:
        return (1.0 / len(actions),) * len(actions)
    return tuple(value / total for value in positive)


class MCCFRTrainer:
    """Sparse external-sampling MCCFR, CFR+ or DCFR for heads-up rounds 2--5.

    Chance and opponent actions are sampled from the real game/current policy;
    all traverser actions are enumerated.  In the two-player simple-average
    estimator, an update pass for player ``i`` accumulates the current strategy
    at sampled nodes of player ``1-i`` *before* sampling their action.  The
    on-policy probability of visiting that node already contains that player's
    own reach, so no additional own/opponent reach multiplier is applied.

    DCFR uses one-based iteration ``t``.  Before adding iteration ``t`` regret,
    existing positive regret is multiplied by ``t**alpha/(t**alpha+1)`` and
    negative regret by ``t**beta/(t**beta+1)``.  Per-node timestamps and global
    log-prefix products apply every skipped factor lazily and exactly up to
    floating-point roundoff.  Average-strategy samples receive weight
    ``t**gamma``, algebraically equivalent (after normalization) to recursively
    multiplying old contributions by ``(t/(t+1))**gamma``.
    """

    def __init__(
        self,
        game_config: GameConfig,
        training_config: MCCFRConfig,
        *,
        _testing_only_allow_small_game: bool = False,
        storage_directory: Path | None = None,
    ) -> None:
        valid_hand_size = game_config.hand_size in (2, 3, 4, 5) or (
            _testing_only_allow_small_game and game_config.hand_size == 1
        )
        if game_config.players != 2 or not valid_hand_size:
            raise ValueError(
                "sampled tabular MCCFR requires exactly two players and hand_size in 2..5"
            )
        self.game_config = game_config
        self.training_config = training_config
        if training_config.storage_backend == "mmap":
            if storage_directory is None:
                raise ValueError("mmap MCCFR storage requires a run-local directory")
            self.nodes = MMapCompactMCCFRStore(storage_directory, training_config.numeric_dtype)
        else:
            self.nodes = CompactMCCFRStore()
        self.iteration = 0
        self.nodes_visited = 0
        self.rng = random.Random(training_config.seed)
        self.diagnostics: dict[str, object] = {
            "traversals_by_player": [0, 0],
            "traversals_by_starting_player": [0, 0],
            "chance_samples": 0,
            "policy_lookup": {
                "exact": 0,
                "uniform_fallback": 0,
                "by_seat_phase": {},
            },
        }
        # prefix[t] is the log product of discount factors for k=1..t.
        self._positive_log_prefix = [0.0]
        self._negative_log_prefix = [0.0]
        self._nodes_this_traversal = 0
        self._last_estimated_table_bytes = 0
        self._state_counts: dict[str, int] = {}
        self._cached_regret_diagnostics = {
            "positive_sum": 0.0,
            "negative_sum": 0.0,
            "max_positive": 0.0,
            "min_negative": 0.0,
        }
        self._regret_diagnostics_as_of_iteration = 0
        self._current_average_weight = 1.0

    def _ensure_discount_prefix(self, iteration: int) -> None:
        if not self.training_config.uses_dcfr:
            return
        alpha = self.training_config.alpha
        beta = self.training_config.beta
        while len(self._positive_log_prefix) <= iteration:
            t = len(self._positive_log_prefix)
            self._positive_log_prefix.append(
                self._positive_log_prefix[-1] - math.log1p(t ** (-alpha))
            )
            self._negative_log_prefix.append(
                self._negative_log_prefix[-1] - math.log1p(t ** (-beta))
            )

    def _discount_node(self, node: MCCFRNode, iteration: int) -> None:
        if not self.training_config.uses_dcfr:
            return
        if node.last_discount_iteration > iteration:
            raise ValueError("node discount timestamp is ahead of trainer iteration")
        if node.last_discount_iteration == iteration:
            return
        self._ensure_discount_prefix(iteration)
        positive_factor = math.exp(
            self._positive_log_prefix[iteration]
            - self._positive_log_prefix[node.last_discount_iteration]
        )
        negative_factor = math.exp(
            self._negative_log_prefix[iteration]
            - self._negative_log_prefix[node.last_discount_iteration]
        )
        for index, regret in enumerate(node.regrets):
            node.regrets[index] = regret * (positive_factor if regret >= 0.0 else negative_factor)
        node.last_discount_iteration = iteration

    def _discount_index(self, index: int, iteration: int) -> None:
        if not self.training_config.uses_dcfr:
            return
        last_iteration = self.nodes.timestamp(index)
        if last_iteration > iteration:
            raise ValueError("node discount timestamp is ahead of trainer iteration")
        if last_iteration == iteration:
            return
        self._ensure_discount_prefix(iteration)
        positive_factor = math.exp(
            self._positive_log_prefix[iteration] - self._positive_log_prefix[last_iteration]
        )
        negative_factor = math.exp(
            self._negative_log_prefix[iteration] - self._negative_log_prefix[last_iteration]
        )
        self.nodes.scale_regrets(index, positive_factor, negative_factor)
        # Timestamps are stored in the compact numeric slab.  Updating the
        # array directly avoids materializing a Python node on every visit.
        self.nodes.set_timestamp(index, iteration)

    def materialize_discounts(self) -> None:
        """Bring every sparse node to the current DCFR iteration."""
        if not self.training_config.uses_dcfr:
            return
        self._ensure_discount_prefix(self.iteration)
        for index in self.nodes.iter_indices():
            self._discount_index(index, self.iteration)

    def _node(self, key: InformationKey, actions: tuple[int, ...], iteration: int) -> int:
        index, is_new = self.nodes.lookup(key, actions, iteration)
        if is_new:
            label = f"seat_{int(key[0])}/phase_{int(key[1])}"
            self._state_counts[label] = self._state_counts.get(label, 0) + 1
        else:
            self._discount_index(index, iteration)
        return index

    def _average_weight(self, iteration: int) -> float:
        if self.training_config.uses_dcfr:
            return float(iteration) ** self.training_config.gamma
        if self.training_config.uses_cfr_plus:
            # Linear averaging is the standard companion to regret clipping in
            # CFR+.  It affects only the exported average strategy, not the
            # external-sampling regret estimator.
            return float(iteration)
        return 1.0

    def _cfr(self, state: RoundState, traverser: int, iteration: int) -> float:
        self._nodes_this_traversal += 1
        if state.phase == Phase.TERMINAL:
            utility0 = state.utilities()[0]
            return utility0 if traverser == 0 else -utility0
        if state.phase == Phase.CHANCE:
            raise AssertionError("chance must be sampled before traversal")

        player, key, actions = state.solver_information()
        node_index = self._node(key, actions, iteration)
        strategy = _regret_matching(actions, self.nodes.regrets(node_index))

        if player != traverser:
            # Standard two-player simple averaging.  This is deliberately done
            # before sampling the opponent action and without another reach
            # factor: the visit itself is the on-policy own-reach sample.
            weight = self._current_average_weight
            self.nodes.add_strategy(node_index, strategy, weight)
            # This is equivalent to random.choices(..., k=1), including its
            # right-bisect boundary rule, but avoids rebuilding range/cumulative
            # helper objects in the hottest sampled branch.
            target = self.rng.random() * sum(strategy)
            cumulative = 0.0
            action_index = len(actions) - 1
            for index, probability in enumerate(strategy[:-1]):
                cumulative += probability
                if target < cumulative:
                    action_index = index
                    break
            use_fast_undo = state.config.hand_size > 1
            token = (
                state.apply_action_solver(actions[action_index])
                if use_fast_undo
                else state.apply_action(actions[action_index])
            )
            value = self._cfr(state, traverser, iteration)
            if use_fast_undo:
                state.undo_solver(token)
            else:
                state.undo(token)
            return value

        action_values: list[float] = []
        use_fast_undo = state.config.hand_size > 1
        for action in actions:
            token = (
                state.apply_action_solver(action) if use_fast_undo else state.apply_action(action)
            )
            action_values.append(self._cfr(state, traverser, iteration))
            if use_fast_undo:
                state.undo_solver(token)
            else:
                state.undo(token)
        node_value = sum(probability * value for probability, value in zip(strategy, action_values))
        # External sampling already samples chance/opponent reach.  Multiplying
        # by that reach again would square the sampling probability and bias the
        # regret estimator.  Vanilla keeps signed cumulative regret; DCFR only
        # discounts it and never applies CFR+ clipping.
        if self.training_config.uses_cfr_plus:
            self.nodes.add_regrets_plus(node_index, action_values, node_value)
        else:
            self.nodes.add_regrets(node_index, action_values, node_value)
        return node_value

    def _run_traversal(
        self, traverser: int, starting_player: int, deal: Deal, iteration: int
    ) -> int:
        game = replace(self.game_config, starting_player=starting_player)
        state = RoundState(game)
        # Queste carte sono state preparate dal programma: non serve ricontrollarle
        # né conservare una copia di una situazione che stiamo per abbandonare.
        state.apply_chance_fast_unchecked(deal)
        self._nodes_this_traversal = 0
        self._cfr(state, traverser, iteration)
        cast_player = self.diagnostics["traversals_by_player"]
        cast_start = self.diagnostics["traversals_by_starting_player"]
        assert isinstance(cast_player, list) and isinstance(cast_start, list)
        cast_player[traverser] += 1
        cast_start[starting_player] += 1
        return self._nodes_this_traversal

    def train_shard_iteration(
        self,
        iteration: int,
        deals_by_traverser: tuple[tuple[Deal, ...], tuple[Deal, ...]],
        *,
        collect_light_diagnostics: bool,
        collect_regret_diagnostics: bool,
    ) -> dict[str, object]:
        """Advance one fixed-starting-player shard from an explicit deal tape.

        The two-process coordinator sends the same paired deal tape to the two
        disjoint starting-player shards.  Each shard still performs traverser 0
        before traverser 1, preserving the reference Gauss-Seidel dependency.
        """
        if iteration != self.iteration + 1:
            raise ValueError("shard iterations must be consecutive")
        starting_player = self.game_config.starting_player
        started = time.perf_counter()
        self._ensure_discount_prefix(iteration)
        self._current_average_weight = self._average_weight(iteration)
        iteration_nodes = 0
        traversal_counts = [0, 0]
        for traverser in (0, 1):
            for deal in deals_by_traverser[traverser]:
                iteration_nodes += self._run_traversal(traverser, starting_player, deal, iteration)
                traversal_counts[traverser] += 1

        traversal_seconds = time.perf_counter() - started
        self.iteration = iteration
        self.nodes_visited += iteration_nodes
        self.diagnostics["chance_samples"] = int(self.diagnostics["chance_samples"]) + sum(
            map(len, deals_by_traverser)
        )
        diagnostics_started = time.perf_counter()
        if collect_light_diagnostics or collect_regret_diagnostics:
            self._last_estimated_table_bytes = self.estimated_table_bytes()
        if collect_regret_diagnostics:
            self.materialize_discounts()
            self._cached_regret_diagnostics = self.regret_diagnostics()
            self._regret_diagnostics_as_of_iteration = iteration
        diagnostics_seconds = time.perf_counter() - diagnostics_started
        return {
            "iteration": iteration,
            "traversal_seconds": traversal_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "traversals_by_player": tuple(traversal_counts),
            "nodes_visited": iteration_nodes,
            "cumulative_nodes_visited": self.nodes_visited,
            "information_states": len(self.nodes),
            "information_states_by_seat_phase": self._states_by_seat_phase(),
            "regret_diagnostics": dict(self._cached_regret_diagnostics),
            "regret_diagnostics_as_of_iteration": (self._regret_diagnostics_as_of_iteration),
            "estimated_table_bytes": self._last_estimated_table_bytes,
        }

    def train_iteration(self) -> MCCFRIterationMetrics:
        started = time.perf_counter()
        iteration = self.iteration + 1
        self._ensure_discount_prefix(iteration)
        self._current_average_weight = self._average_weight(iteration)
        iteration_nodes = 0
        traversal_counts = [0, 0]
        starting_counts = [0, 0]
        chance_samples = 0

        starts = (
            (0, 1)
            if self.training_config.alternate_starting_player
            else (self.game_config.starting_player,)
        )
        for traverser in (0, 1):
            for _ in range(self.training_config.traversals_per_player):
                if len(starts) == 2 and self.training_config.paired_starting_player:
                    sampler = RoundState(replace(self.game_config, starting_player=starts[0]))
                    deal = sampler.sample_chance(self.rng)
                    chance_samples += 1
                    for starting_player in starts:
                        iteration_nodes += self._run_traversal(
                            traverser, starting_player, deal, iteration
                        )
                        traversal_counts[traverser] += 1
                        starting_counts[starting_player] += 1
                else:
                    for starting_player in starts:
                        sampler = RoundState(
                            replace(self.game_config, starting_player=starting_player)
                        )
                        deal = sampler.sample_chance(self.rng)
                        chance_samples += 1
                        iteration_nodes += self._run_traversal(
                            traverser, starting_player, deal, iteration
                        )
                        traversal_counts[traverser] += 1
                        starting_counts[starting_player] += 1

        traversal_seconds = time.perf_counter() - started
        self.iteration = iteration
        self.nodes_visited += iteration_nodes
        self.diagnostics["chance_samples"] = (
            int(self.diagnostics["chance_samples"]) + chance_samples
        )
        collect_light_diagnostics = (
            iteration == 1 or iteration % self.training_config.log_every == 0
        )
        collect_regret_diagnostics = (
            iteration == 1
            or iteration % self.training_config.checkpoint_every == 0
            or iteration == self.training_config.iterations
        )
        diagnostics_started = time.perf_counter()
        if collect_light_diagnostics or collect_regret_diagnostics:
            self._last_estimated_table_bytes = self.estimated_table_bytes()
        if collect_regret_diagnostics:
            # Checkpoint/export semantics require every lazy node to represent
            # the current global iteration.  Doing this once here also makes
            # the regret diagnostic current; save_training then becomes a
            # no-op materialization followed by serialization.
            self.materialize_discounts()
            self._cached_regret_diagnostics = self.regret_diagnostics()
            self._regret_diagnostics_as_of_iteration = iteration
        diagnostics_seconds = time.perf_counter() - diagnostics_started
        elapsed_seconds = time.perf_counter() - started
        return MCCFRIterationMetrics(
            iteration=iteration,
            elapsed_seconds=elapsed_seconds,
            traversal_seconds=traversal_seconds,
            diagnostics_seconds=diagnostics_seconds,
            traversals=sum(traversal_counts),
            traversals_by_player=tuple(traversal_counts),
            traversals_by_starting_player=tuple(starting_counts),
            chance_samples=chance_samples,
            nodes_visited=iteration_nodes,
            cumulative_nodes_visited=self.nodes_visited,
            information_states=len(self.nodes),
            information_states_by_seat_phase=self._states_by_seat_phase(),
            regret_diagnostics=dict(self._cached_regret_diagnostics),
            regret_diagnostics_as_of_iteration=self._regret_diagnostics_as_of_iteration,
            estimated_table_bytes=self._last_estimated_table_bytes,
        )

    def _states_by_seat_phase(self) -> dict[str, int]:
        return dict(self._state_counts)

    def regret_diagnostics(self) -> dict[str, float]:
        positive_sum = 0.0
        negative_sum = 0.0
        max_positive = 0.0
        min_negative = 0.0
        # Streaming is important here: materializing all regrets and then
        # positive/negative sublists caused multi-GB transient allocations on
        # large sparse tables.  This diagnostic must not change training RSS.
        for index in self.nodes.iter_indices():
            for value in self.nodes.regrets(index):
                if value > 0.0:
                    positive_sum += value
                    max_positive = max(max_positive, value)
                elif value < 0.0:
                    negative_sum += value
                    min_negative = min(min_negative, value)
        return {
            "positive_sum": positive_sum,
            "negative_sum": negative_sum,
            "max_positive": max_positive,
            "min_negative": min_negative,
        }

    def estimated_table_bytes(self) -> int:
        """Estimate the live Python table graph from a bounded node sample."""
        return self.nodes.estimated_bytes()

    def policy_entries(self) -> dict[InformationKey, PolicyEntry]:
        self.materialize_discounts()
        entries: dict[InformationKey, PolicyEntry] = {}
        for index in self.nodes.iter_indices():
            key = self.nodes.key_at(index)
            actions = self.nodes.actions(index)
            strategy_sum = self.nodes.strategy_sum(index)
            total = sum(strategy_sum)
            probabilities = (
                tuple(value / total for value in strategy_sum)
                if total > 1e-15
                else (1.0 / len(actions),) * len(actions)
            )
            entries[key] = (actions, probabilities)
        return entries

    def average_policy(self) -> TabularPolicy:
        return TabularPolicy(self.policy_entries())

    def train(self, output_dir: Path) -> list[MCCFRIterationMetrics]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / ".internal" / "metrics.jsonl"
        checkpoints_path = output_dir / ".internal" / "checkpoints.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        reports: list[MCCFRIterationMetrics] = []
        wall_started = time.monotonic()
        self.stop_reason: str | None = None
        while self.iteration < self.training_config.iterations:
            if (
                self.training_config.max_wall_seconds is not None
                and time.monotonic() - wall_started >= self.training_config.max_wall_seconds
            ):
                self.stop_reason = "wall_clock_budget"
                break
            report = self.train_iteration()
            reports.append(report)
            if report.iteration % self.training_config.log_every == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
                log_training_metrics(report)
            if report.iteration % self.training_config.checkpoint_every == 0:
                checkpoint_started = time.perf_counter()
                self.save_training(output_dir)
                checkpoint_seconds = time.perf_counter() - checkpoint_started
                with checkpoints_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "iteration": report.iteration,
                                "checkpoint_seconds": checkpoint_seconds,
                                "checkpoint_bytes": (output_dir / "training.pt").stat().st_size,
                                "checkpoint_bytes_scope": (
                                    "serialized checkpoint file only; live mmap "
                                    "numeric slabs are excluded"
                                    if self.training_config.storage_backend == "mmap"
                                    else "complete serialized checkpoint file"
                                ),
                                "artifact_role": ("recoverable mutable training checkpoint"),
                                "information_states": len(self.nodes),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                log_training_checkpoint(
                    report.iteration,
                    checkpoint_seconds,
                    (output_dir / "training.pt").stat().st_size,
                    mmap_storage=self.training_config.storage_backend == "mmap",
                )
        # Avoid serializing the large training table twice when the final
        # iteration is already a checkpoint boundary.
        if self.iteration % self.training_config.checkpoint_every != 0:
            self.save_training(output_dir)
        if self.training_config.export_policy_on_finish:
            self.save_policy(output_dir)
        return reports

    def save_training(self, output_dir: Path) -> None:
        self.materialize_discounts()
        save_sampled_training_checkpoint(self, output_dir / "training.pt")

    def save_policy(self, output_dir: Path) -> None:
        self.materialize_discounts()
        save_sampled_policy_checkpoint(self, output_dir / "policy.pt")

    def save(self, output_dir: Path) -> None:
        self.save_training(output_dir)
        self.save_policy(output_dir)

    def close(self) -> None:
        close = getattr(self.nodes, "close", None)
        if close is not None:
            close()

    @classmethod
    def restore(cls, path: Path) -> MCCFRTrainer:
        payload = load_sampled_training_payload(path)
        if payload.get("kind") == "external-sampling-tabular-sharded-training":
            from .mccfr_parallel import MCCFRParallelTrainer

            return MCCFRParallelTrainer.restore(path)  # type: ignore[return-value]
        training = MCCFRConfig.from_dict(payload["mccfr"])
        persistent = payload.get("persistent_store")
        if persistent is not None:
            directory = path.parent / str(persistent["directory"])
            trainer = cls(
                GameConfig.from_dict(payload["game"]),
                training,
                storage_directory=directory,
            )
            trainer.nodes = MMapCompactMCCFRStore.restore(directory)
        else:
            trainer = cls(GameConfig.from_dict(payload["game"]), training)
        trainer.iteration = int(payload["iteration"])
        trainer.nodes_visited = int(payload["nodes_visited"])
        trainer.rng.setstate(payload["rng_state"])
        trainer.diagnostics = payload["diagnostics"]
        trainer._ensure_discount_prefix(trainer.iteration)
        trainer.information_states = int(payload.get("information_states", 0))
        trainer._state_counts = {
            str(key): int(value)
            for key, value in payload.get("information_states_by_seat_phase", {}).items()
        }
        if persistent is not None:
            pass
        elif "nodes_compact" in payload:
            trainer.nodes = CompactMCCFRStore.from_payload(payload["nodes_compact"])
        else:
            # Consume legacy nodes one by one while migrating.  Popping from
            # the old dict releases its Python tuple/list graph progressively,
            # avoiding a second full-size copy during the 650k migration.
            legacy_nodes = payload.pop("nodes")
            while legacy_nodes:
                key, values = legacy_nodes.popitem()
                actions, regrets, strategy_sum, last_discount_iteration = values
                trainer.nodes[key] = MCCFRNode(
                    tuple(actions),
                    list(regrets),
                    list(strategy_sum),
                    int(last_discount_iteration),
                )
                label = f"seat_{int(key[0])}/phase_{int(key[1])}"
                trainer._state_counts[label] = trainer._state_counts.get(label, 0) + 1
            trainer.information_states = len(trainer.nodes)
        return trainer
