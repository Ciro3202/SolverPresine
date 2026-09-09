from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from presine.game.config import GameConfig
from presine.game.state import Deal, RoundState

from ..checkpoint_deployment.sampled_checkpoint import (
    ENSEMBLE_CHECKPOINT_FORMAT,
    ENSEMBLE_POLICY_KIND,
    ENSEMBLE_TRAINING_KIND,
    SHARDED_CHECKPOINT_FORMAT,
    SHARDED_POLICY_KIND,
    SHARDED_TRAINING_KIND,
    load_sampled_training_payload,
    save_sampled_policy_checkpoint,
    save_sampled_training_checkpoint,
    save_sharded_manifest,
)
from ..mccfr_config import MCCFRConfig
from .mccfr_trainer import (
    MCCFRIterationMetrics,
    MCCFRTrainer,
    log_training_checkpoint,
    log_training_metrics,
)


def _shard_seed(seed: int, starting_player: int) -> int:
    """Stable independent RNG stream for sampled opponent actions."""
    mask = (1 << 64) - 1
    salt = 0x9E3779B97F4A7C15 * (starting_player + 1)
    return (int(seed) ^ salt) & mask


def _replica_seed(seed: int, replica: int) -> int:
    """Independent deterministic root seed for one two-shard replica."""
    mask = (1 << 64) - 1
    salt = 0xD1B54A32D192ED03 * (replica + 1)
    return (int(seed) ^ salt) & mask


def _worker_main(
    connection: Any,
    game_values: dict[str, object],
    config_values: dict[str, object],
    starting_player: int,
    resume_path: str | None,
    storage_directory: str | None,
) -> None:
    """Own one disjoint starting-player table for the worker lifetime."""
    try:
        if resume_path is not None:
            trainer = MCCFRTrainer.restore(Path(resume_path))
        else:
            game = replace(
                GameConfig.from_dict(game_values),
                starting_player=starting_player,
            )
            parallel_config = MCCFRConfig.from_dict(config_values)
            serial_config = replace(
                parallel_config,
                workers=1,
                alternate_starting_player=False,
                paired_starting_player=False,
            )
            trainer = MCCFRTrainer(
                game,
                serial_config,
                storage_directory=(Path(storage_directory) if storage_directory else None),
            )
            trainer.rng.seed(_shard_seed(parallel_config.seed, starting_player))

        if trainer.game_config.starting_player != starting_player:
            raise ValueError("checkpoint was assigned to the wrong starting shard")

        while True:
            message = connection.recv()
            command = message[0]
            if command == "train":
                _, iteration, deals, light, regret = message
                result = trainer.train_shard_iteration(
                    int(iteration),
                    deals,
                    collect_light_diagnostics=bool(light),
                    collect_regret_diagnostics=bool(regret),
                )
                connection.send(("ok", result))
            elif command == "save_training":
                path = Path(message[1])
                trainer.materialize_discounts()
                save_sampled_training_checkpoint(trainer, path)
                connection.send(("ok", {"bytes": path.stat().st_size}))
            elif command == "save_policy":
                path = Path(message[1])
                save_sampled_policy_checkpoint(trainer, path)
                connection.send(("ok", {"bytes": path.stat().st_size}))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown MCCFR shard command: {command}")
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except BaseException:
            pass
    finally:
        connection.close()


class MCCFRParallelTrainer:
    """Two persistent, disjoint starting-player MCCFR shards.

    The coordinator owns chance sampling.  Both workers receive the same paired
    deal tape, own independent opponent-action RNG streams, and synchronize at
    each full iteration.  Since ``starting_player`` is part of every
    information key, shard updates commute and no regret merge is required.
    """

    def __init__(self, game_config: GameConfig, training_config: MCCFRConfig) -> None:
        if training_config.workers != 2:
            raise ValueError("MCCFRParallelTrainer requires workers=2")
        if game_config.players != 2 or game_config.hand_size not in (2, 3, 4, 5):
            raise ValueError("two-shard MCCFR requires a heads-up round with 2..5 cards")
        self.game_config = game_config
        self.training_config = training_config
        self.iteration = 0
        self.nodes_visited = 0
        self.rng = random.Random(training_config.seed)
        self.diagnostics: dict[str, object] = {
            "traversals_by_player": [0, 0],
            "traversals_by_starting_player": [0, 0],
            "chance_samples": 0,
            "parallel_scheme": "disjoint-starting-player-v1",
            "policy_lookup": {
                "exact": 0,
                "uniform_fallback": 0,
                "by_seat_phase": {},
            },
        }
        self.information_states = 0
        self._state_counts: dict[str, int] = {}
        self._cached_regret_diagnostics = {
            "positive_sum": 0.0,
            "negative_sum": 0.0,
            "max_positive": 0.0,
            "min_negative": 0.0,
        }
        self._regret_diagnostics_as_of_iteration = 0
        self._last_estimated_table_bytes = 0
        self._active_slot = -1
        self._resume_paths: tuple[Path, Path] | None = None
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self._storage_root: Path | None = None

    def set_storage_root(self, root: Path) -> None:
        """Choose a private, run-local root before persistent workers start."""
        if self._processes:
            raise RuntimeError("cannot change MCCFR storage after workers start")
        self._storage_root = Path(root)

    def _start_workers(self) -> None:
        if self._processes:
            return
        context = mp.get_context("spawn")
        for starting_player in (0, 1):
            parent, child = context.Pipe()
            resume = (
                str(self._resume_paths[starting_player]) if self._resume_paths is not None else None
            )
            storage = (
                str(self._storage_root / f"start-{starting_player}")
                if self.training_config.storage_backend == "mmap" and resume is None
                else None
            )
            process = context.Process(
                target=_worker_main,
                args=(
                    child,
                    self.game_config.to_dict(),
                    self.training_config.to_dict(),
                    starting_player,
                    resume,
                    storage,
                ),
                name=f"mccfr-start-{starting_player}",
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    def _receive_all(self) -> list[Any]:
        results: list[Any] = []
        for connection in self._connections:
            status, payload = connection.recv()
            if status != "ok":
                raise RuntimeError(f"MCCFR shard failed:\n{payload}")
            results.append(payload)
        return results

    def _send_all(self, messages: tuple[tuple[Any, ...], tuple[Any, ...]]) -> list[Any]:
        self._start_workers()
        for connection, message in zip(self._connections, messages):
            connection.send(message)
        return self._receive_all()

    def train_iteration(self) -> MCCFRIterationMetrics:
        started = time.perf_counter()
        iteration = self.iteration + 1
        sampler = RoundState(replace(self.game_config, starting_player=0))
        deals: list[tuple[Deal, ...]] = []
        for _traverser in (0, 1):
            deals.append(
                tuple(
                    sampler.sample_chance(self.rng)
                    for _ in range(self.training_config.traversals_per_player)
                )
            )
        deal_tape = (deals[0], deals[1])
        chance_samples = sum(map(len, deal_tape))
        collect_light = iteration == 1 or iteration % self.training_config.log_every == 0
        collect_regret = (
            iteration == 1
            or iteration % self.training_config.checkpoint_every == 0
            or iteration == self.training_config.iterations
        )
        command = ("train", iteration, deal_tape, collect_light, collect_regret)
        shards = self._send_all((command, command))

        iteration_nodes = sum(int(item["nodes_visited"]) for item in shards)
        traversal_counts = tuple(
            sum(int(item["traversals_by_player"][player]) for item in shards) for player in (0, 1)
        )
        starts_per_shard = sum(map(len, deal_tape))
        starting_counts = (starts_per_shard, starts_per_shard)
        self.iteration = iteration
        self.nodes_visited += iteration_nodes
        self.information_states = sum(int(item["information_states"]) for item in shards)
        state_counts: dict[str, int] = {}
        for item in shards:
            for label, count in item["information_states_by_seat_phase"].items():
                state_counts[label] = state_counts.get(label, 0) + int(count)
        self._state_counts = state_counts
        self._last_estimated_table_bytes = sum(
            int(item["estimated_table_bytes"]) for item in shards
        )
        if collect_regret:
            diagnostics = [item["regret_diagnostics"] for item in shards]
            self._cached_regret_diagnostics = {
                "positive_sum": sum(float(item["positive_sum"]) for item in diagnostics),
                "negative_sum": sum(float(item["negative_sum"]) for item in diagnostics),
                "max_positive": max(float(item["max_positive"]) for item in diagnostics),
                "min_negative": min(float(item["min_negative"]) for item in diagnostics),
            }
            self._regret_diagnostics_as_of_iteration = iteration

        cast_player = self.diagnostics["traversals_by_player"]
        cast_start = self.diagnostics["traversals_by_starting_player"]
        assert isinstance(cast_player, list) and isinstance(cast_start, list)
        for player in (0, 1):
            cast_player[player] += traversal_counts[player]
            cast_start[player] += starting_counts[player]
        self.diagnostics["chance_samples"] = (
            int(self.diagnostics["chance_samples"]) + chance_samples
        )
        elapsed = time.perf_counter() - started
        return MCCFRIterationMetrics(
            iteration=iteration,
            elapsed_seconds=elapsed,
            traversal_seconds=max(float(item["traversal_seconds"]) for item in shards),
            diagnostics_seconds=max(float(item["diagnostics_seconds"]) for item in shards),
            traversals=sum(traversal_counts),
            traversals_by_player=traversal_counts,
            traversals_by_starting_player=starting_counts,
            chance_samples=chance_samples,
            nodes_visited=iteration_nodes,
            cumulative_nodes_visited=self.nodes_visited,
            information_states=self.information_states,
            information_states_by_seat_phase=dict(self._state_counts),
            regret_diagnostics=dict(self._cached_regret_diagnostics),
            regret_diagnostics_as_of_iteration=self._regret_diagnostics_as_of_iteration,
            estimated_table_bytes=self._last_estimated_table_bytes,
        )

    def _shard_targets(self, output_dir: Path, stem: str, slot: int) -> tuple[Path, Path]:
        root = output_dir / ".internal" / "shards"
        return tuple(root / f"{stem}-slot{slot}-start{start}.pt" for start in (0, 1))  # type: ignore[return-value]

    def estimated_table_bytes(self) -> int:
        return self._last_estimated_table_bytes

    def save_training(self, output_dir: Path) -> None:
        previous_slot = self._active_slot
        slot = 0 if previous_slot < 0 else 1 - previous_slot
        targets = self._shard_targets(output_dir, "training", slot)
        replies = self._send_all(
            tuple(("save_training", str(path)) for path in targets)  # type: ignore[arg-type]
        )
        payload = {
            "format": SHARDED_CHECKPOINT_FORMAT,
            "kind": SHARDED_TRAINING_KIND,
            "artifact_role": (
                "recoverable mutable training manifest; its shard paths and mmap "
                "stores are required together to resume"
            ),
            "game": self.game_config.to_dict(),
            "mccfr": self.training_config.to_dict(),
            "iteration": self.iteration,
            "nodes_visited": self.nodes_visited,
            "rng_state": self.rng.getstate(),
            "diagnostics": self.diagnostics,
            "information_states": self.information_states,
            "information_states_by_seat_phase": self._state_counts,
            "active_slot": slot,
            "shards": [path.relative_to(output_dir).as_posix() for path in targets],
            "shard_bytes": [int(reply["bytes"]) for reply in replies],
            "runtime": {"python": sys.version, "platform": platform.platform()},
        }
        save_sharded_manifest(payload, output_dir / "training.pt")
        # In quota-constrained runs, retain only the newly published slot.
        # The previous slot is removed *after* the new manifest is durable, so
        # an interrupted save still leaves the previous checkpoint usable.
        if os.environ.get("PRESINE_SINGLE_CHECKPOINT", "0") == "1" and previous_slot >= 0:
            for path in self._shard_targets(output_dir, "training", previous_slot):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        self._active_slot = slot

    def save_policy(self, output_dir: Path) -> None:
        slot = max(self._active_slot, 0)
        targets = self._shard_targets(output_dir, "policy", slot)
        replies = self._send_all(
            tuple(("save_policy", str(path)) for path in targets)  # type: ignore[arg-type]
        )
        payload = {
            "format": SHARDED_CHECKPOINT_FORMAT,
            "kind": SHARDED_POLICY_KIND,
            "artifact_role": ("read-only average-policy manifest over two starting-player shards"),
            "game": self.game_config.to_dict(),
            "mccfr": self.training_config.to_dict(),
            "iteration": self.iteration,
            "diagnostics": self.diagnostics,
            "shards": [path.relative_to(output_dir).as_posix() for path in targets],
            "shard_bytes": [int(reply["bytes"]) for reply in replies],
            "warning": (
                "external-sampling guarantees convergence in expectation/high "
                "probability under its assumptions; sampled regrets are not an "
                "exploitability certificate"
            ),
        }
        save_sharded_manifest(payload, output_dir / "policy.pt")

    def train(self, output_dir: Path) -> list[MCCFRIterationMetrics]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / ".internal" / "metrics.jsonl"
        checkpoints_path = output_dir / ".internal" / "checkpoints.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.training_config.storage_backend == "mmap" and self._storage_root is None:
            self.set_storage_root(output_dir / ".internal" / "mmap-table")
        reports: list[MCCFRIterationMetrics] = []
        wall_started = time.monotonic()
        self.stop_reason: str | None = None
        try:
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
                    checkpoint_bytes = sum(
                        path.stat().st_size
                        for path in self._shard_targets(output_dir, "training", self._active_slot)
                    )
                    with checkpoints_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "iteration": report.iteration,
                                    "checkpoint_seconds": checkpoint_seconds,
                                    "checkpoint_bytes": checkpoint_bytes,
                                    "checkpoint_bytes_scope": (
                                        "serialized checkpoint shard files only; "
                                        "live mmap numeric slabs are excluded"
                                    ),
                                    "artifact_role": ("recoverable mutable training checkpoint"),
                                    "information_states": self.information_states,
                                    "sharded": True,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    log_training_checkpoint(
                        report.iteration,
                        checkpoint_seconds,
                        checkpoint_bytes,
                        mmap_storage=self.training_config.storage_backend == "mmap",
                    )
            if self.iteration % self.training_config.checkpoint_every != 0:
                self.save_training(output_dir)
            if self.training_config.export_policy_on_finish:
                self.save_policy(output_dir)
            return reports
        finally:
            self.close()

    def close(self) -> None:
        if not self._processes:
            return
        try:
            for connection in self._connections:
                connection.send(("close",))
            self._receive_all()
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            for connection in self._connections:
                connection.close()
            for process in self._processes:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
            self._connections.clear()
            self._processes.clear()

    @classmethod
    def restore(cls, path: Path) -> MCCFRParallelTrainer:
        payload = load_sampled_training_payload(path)
        if payload.get("kind") != SHARDED_TRAINING_KIND:
            raise ValueError("not a sharded sampled-tabular checkpoint")
        trainer = cls(
            GameConfig.from_dict(payload["game"]),
            MCCFRConfig.from_dict(payload["mccfr"]),
        )
        trainer.iteration = int(payload["iteration"])
        trainer.nodes_visited = int(payload["nodes_visited"])
        trainer.rng.setstate(payload["rng_state"])
        trainer.diagnostics = payload["diagnostics"]
        trainer.information_states = int(payload["information_states"])
        trainer._state_counts = {
            str(key): int(value)
            for key, value in payload["information_states_by_seat_phase"].items()
        }
        trainer._active_slot = int(payload["active_slot"])
        trainer._resume_paths = tuple(path.parent / str(relative) for relative in payload["shards"])  # type: ignore[assignment]
        return trainer


class MCCFREnsembleTrainer:
    """Run independent two-shard DCFR replicas concurrently.

    Each replica keeps its own regret table and therefore needs no locks or
    cross-process regret merge.  The final policy is an equal-weight ensemble
    over replicas that covered a queried information state.  This is an
    empirical variance-reduction technique, not a replacement for a single
    shared-table CFR convergence proof.
    """

    def __init__(self, game_config: GameConfig, training_config: MCCFRConfig) -> None:
        if training_config.replicas <= 1 or training_config.workers != 2:
            raise ValueError("ensemble training requires replicas>1 and workers=2")
        self.game_config = game_config
        self.training_config = training_config
        self.replicas = [
            MCCFRParallelTrainer(
                game_config,
                replace(
                    training_config,
                    replicas=1,
                    seed=_replica_seed(training_config.seed, replica),
                ),
            )
            for replica in range(training_config.replicas)
        ]
        self.iteration = 0
        self.nodes_visited = 0
        self.information_states = 0
        self.diagnostics: dict[str, object] = {
            "parallel_scheme": "independent-replica-ensemble-v1",
            "replicas": training_config.replicas,
            "traversals_by_player": [0, 0],
            "traversals_by_starting_player": [0, 0],
            "chance_samples": 0,
        }
        self._state_counts: dict[str, int] = {}
        self._last_estimated_table_bytes = 0
        self._cached_regret_diagnostics = {
            "positive_sum": 0.0,
            "negative_sum": 0.0,
            "max_positive": 0.0,
            "min_negative": 0.0,
        }
        self._regret_diagnostics_as_of_iteration = 0

    @staticmethod
    def _sum_counts(reports: list[MCCFRIterationMetrics]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for report in reports:
            for label, count in report.information_states_by_seat_phase.items():
                merged[label] = merged.get(label, 0) + int(count)
        return merged

    def train_iteration(self) -> MCCFRIterationMetrics:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(self.replicas)) as executor:
            reports = list(executor.map(lambda replica: replica.train_iteration(), self.replicas))
        iteration = reports[0].iteration
        if any(report.iteration != iteration for report in reports):
            raise RuntimeError("ensemble replicas lost iteration synchronization")
        self.iteration = iteration
        iteration_nodes = sum(report.nodes_visited for report in reports)
        self.nodes_visited += iteration_nodes
        self.information_states = sum(report.information_states for report in reports)
        self._state_counts = self._sum_counts(reports)
        self._last_estimated_table_bytes = sum(report.estimated_table_bytes for report in reports)
        traversal_counts = tuple(
            sum(report.traversals_by_player[player] for report in reports) for player in (0, 1)
        )
        starting_counts = tuple(
            sum(report.traversals_by_starting_player[player] for report in reports)
            for player in (0, 1)
        )
        chance_samples = sum(report.chance_samples for report in reports)
        cast_player = self.diagnostics["traversals_by_player"]
        cast_start = self.diagnostics["traversals_by_starting_player"]
        assert isinstance(cast_player, list) and isinstance(cast_start, list)
        for player in (0, 1):
            cast_player[player] += traversal_counts[player]
            cast_start[player] += starting_counts[player]
        self.diagnostics["chance_samples"] = (
            int(self.diagnostics["chance_samples"]) + chance_samples
        )
        if any(report.regret_diagnostics_as_of_iteration == iteration for report in reports):
            diagnostics = [report.regret_diagnostics for report in reports]
            self._cached_regret_diagnostics = {
                "positive_sum": sum(float(item["positive_sum"]) for item in diagnostics),
                "negative_sum": sum(float(item["negative_sum"]) for item in diagnostics),
                "max_positive": max(float(item["max_positive"]) for item in diagnostics),
                "min_negative": min(float(item["min_negative"]) for item in diagnostics),
            }
            self._regret_diagnostics_as_of_iteration = iteration
        elapsed = time.perf_counter() - started
        return MCCFRIterationMetrics(
            iteration=iteration,
            elapsed_seconds=elapsed,
            traversal_seconds=max(report.traversal_seconds for report in reports),
            diagnostics_seconds=max(report.diagnostics_seconds for report in reports),
            traversals=sum(report.traversals for report in reports),
            traversals_by_player=traversal_counts,
            traversals_by_starting_player=starting_counts,
            chance_samples=chance_samples,
            nodes_visited=iteration_nodes,
            cumulative_nodes_visited=self.nodes_visited,
            information_states=self.information_states,
            information_states_by_seat_phase=dict(self._state_counts),
            regret_diagnostics=dict(self._cached_regret_diagnostics),
            regret_diagnostics_as_of_iteration=self._regret_diagnostics_as_of_iteration,
            estimated_table_bytes=self._last_estimated_table_bytes,
        )

    def _replica_root(self, output_dir: Path, replica: int) -> Path:
        return output_dir / ".internal" / "replicas" / f"replica-{replica}"

    def update_training_config(self, training_config: MCCFRConfig) -> None:
        """Apply allowed resume settings while preserving replica RNG streams."""
        if training_config.replicas != len(self.replicas):
            raise ValueError("resume cannot change the number of ensemble replicas")
        self.training_config = training_config
        for replica in self.replicas:
            replica.training_config = replace(
                training_config,
                replicas=1,
                seed=replica.training_config.seed,
            )

    def _save_all(self, output_dir: Path, method: str) -> list[Path]:
        roots = [self._replica_root(output_dir, index) for index in range(len(self.replicas))]
        with ThreadPoolExecutor(max_workers=len(self.replicas)) as executor:
            futures = [
                executor.submit(getattr(replica, method), root)
                for replica, root in zip(self.replicas, roots)
            ]
            for future in futures:
                future.result()
        return roots

    def save_training(self, output_dir: Path) -> None:
        roots = self._save_all(output_dir, "save_training")
        payload = {
            "format": ENSEMBLE_CHECKPOINT_FORMAT,
            "kind": ENSEMBLE_TRAINING_KIND,
            "artifact_role": (
                "recoverable mutable training manifest; all replica manifests, "
                "shards and mmap stores are required together to resume"
            ),
            "game": self.game_config.to_dict(),
            "mccfr": self.training_config.to_dict(),
            "iteration": self.iteration,
            "nodes_visited": self.nodes_visited,
            "diagnostics": self.diagnostics,
            "information_states": self.information_states,
            "information_states_by_seat_phase": self._state_counts,
            "replicas": [
                (root / "training.pt").relative_to(output_dir).as_posix() for root in roots
            ],
        }
        save_sharded_manifest(payload, output_dir / "training.pt")

    def save_policy(self, output_dir: Path) -> None:
        roots = self._save_all(output_dir, "save_policy")
        payload = {
            "format": ENSEMBLE_CHECKPOINT_FORMAT,
            "kind": ENSEMBLE_POLICY_KIND,
            "artifact_role": (
                "read-only average-policy manifest; replica policies are combined "
                "with equal weight at lookup"
            ),
            "game": self.game_config.to_dict(),
            "mccfr": self.training_config.to_dict(),
            "iteration": self.iteration,
            "diagnostics": self.diagnostics,
            "replicas": [(root / "policy.pt").relative_to(output_dir).as_posix() for root in roots],
            "warning": (
                "independent replica policies are averaged only at lookup; "
                "this is not a NashConv or exploitability certificate"
            ),
        }
        save_sharded_manifest(payload, output_dir / "policy.pt")

    def estimated_table_bytes(self) -> int:
        return self._last_estimated_table_bytes

    def train(self, output_dir: Path) -> list[MCCFRIterationMetrics]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / ".internal" / "metrics.jsonl"
        checkpoints_path = output_dir / ".internal" / "checkpoints.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.training_config.storage_backend == "mmap":
            for index, replica in enumerate(self.replicas):
                replica.set_storage_root(
                    output_dir / ".internal" / "mmap-replicas" / f"replica-{index}"
                )
        reports: list[MCCFRIterationMetrics] = []
        wall_started = time.monotonic()
        self.stop_reason: str | None = None
        try:
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
                    started = time.perf_counter()
                    self.save_training(output_dir)
                    bytes_written = sum(
                        path.stat().st_size
                        for root in (
                            self._replica_root(output_dir, index)
                            for index in range(len(self.replicas))
                        )
                        for path in (root / ".internal" / "shards").glob("training-*.pt")
                    )
                    checkpoint_seconds = time.perf_counter() - started
                    with checkpoints_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "iteration": report.iteration,
                                    "checkpoint_seconds": checkpoint_seconds,
                                    "checkpoint_bytes": bytes_written,
                                    "checkpoint_bytes_scope": (
                                        "serialized checkpoint shard files only; live mmap "
                                        "numeric slabs are excluded"
                                    ),
                                    "artifact_role": (
                                        "recoverable mutable ensemble training checkpoint"
                                    ),
                                    "information_states": self.information_states,
                                    "replicas": len(self.replicas),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    log_training_checkpoint(
                        report.iteration,
                        checkpoint_seconds,
                        bytes_written,
                        mmap_storage=self.training_config.storage_backend == "mmap",
                    )
            if self.iteration % self.training_config.checkpoint_every != 0:
                self.save_training(output_dir)
            if self.training_config.export_policy_on_finish:
                self.save_policy(output_dir)
            return reports
        finally:
            self.close()

    def close(self) -> None:
        for replica in self.replicas:
            replica.close()

    @classmethod
    def restore(cls, path: Path) -> MCCFREnsembleTrainer:
        payload = load_sampled_training_payload(path)
        if payload.get("kind") != ENSEMBLE_TRAINING_KIND:
            raise ValueError("not an ensemble sampled-tabular checkpoint")
        trainer = cls(
            GameConfig.from_dict(payload["game"]),
            MCCFRConfig.from_dict(payload["mccfr"]),
        )
        restored = [
            MCCFRParallelTrainer.restore(path.parent / str(relative))
            for relative in payload["replicas"]
        ]
        if len(restored) != trainer.training_config.replicas:
            raise ValueError("ensemble checkpoint replica count differs from configuration")
        trainer.replicas = restored
        trainer.iteration = int(payload["iteration"])
        trainer.nodes_visited = int(payload["nodes_visited"])
        trainer.diagnostics = payload["diagnostics"]
        trainer.information_states = int(payload["information_states"])
        trainer._state_counts = {
            str(key): int(value)
            for key, value in payload["information_states_by_seat_phase"].items()
        }
        trainer._last_estimated_table_bytes = sum(
            replica.estimated_table_bytes() for replica in restored
        )
        return trainer


def create_mccfr_trainer(
    game_config: GameConfig, training_config: MCCFRConfig
) -> MCCFRTrainer | MCCFRParallelTrainer | MCCFREnsembleTrainer:
    if training_config.replicas > 1:
        return MCCFREnsembleTrainer(game_config, training_config)
    if training_config.workers == 2:
        return MCCFRParallelTrainer(game_config, training_config)
    return MCCFRTrainer(game_config, training_config)


def restore_mccfr_trainer(path: Path) -> MCCFRTrainer | MCCFRParallelTrainer | MCCFREnsembleTrainer:
    payload = load_sampled_training_payload(path)
    if payload.get("kind") == ENSEMBLE_TRAINING_KIND:
        return MCCFREnsembleTrainer.restore(path)
    if payload.get("kind") == SHARDED_TRAINING_KIND:
        return MCCFRParallelTrainer.restore(path)
    return MCCFRTrainer.restore(path)
