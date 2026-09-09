from __future__ import annotations

"""Bounded expert iteration for the table-free linear blueprint."""

import gzip
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

try:
    import resource
except ImportError:  # pragma: no cover - Windows development environments
    resource = None  # type: ignore[assignment]

import numpy as np

from presine.evaluation.matches import evaluate_round
from presine.evaluation.multiplayer import play_multiplayer_round
from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.policies.abstraction_resolving.resolving import ResolvingPolicy
from presine.policies.heuristic import HeuristicPolicy
from presine.policies.linear import LinearBlueprintPolicy, LinearFeatureConfig
from presine.policies.random import RandomPolicy
from presine.search.config import SearchConfig

OpponentKind = Literal["self", "heuristic", "random", "snapshot"]


@dataclass(frozen=True, slots=True)
class ExpertIterationConfig:
    iterations: int = 3
    games_per_iteration: int = 12
    epochs_per_iteration: int = 4
    learning_rate: float = 0.08
    l2: float = 1e-4
    memory_examples: int = 20_000
    strength_buckets: int = 8
    tile_buckets: int = 32_768
    teacher_blueprint_weight: float = 0.20
    compact_prior: bool = True
    seed: int = 20260823

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.games_per_iteration <= 0 or self.epochs_per_iteration <= 0:
            raise ValueError("expert iteration counts must be positive")
        if self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("learning_rate must be positive and l2 non-negative")
        if self.memory_examples <= 0:
            raise ValueError("memory_examples must be positive")
        if self.tile_buckets and (
            self.tile_buckets < 64 or self.tile_buckets & (self.tile_buckets - 1)
        ):
            raise ValueError("tile_buckets must be zero or a power of two of at least 64")
        if not 0.0 <= self.teacher_blueprint_weight <= 1.0:
            raise ValueError("teacher_blueprint_weight must be in [0, 1]")


@dataclass(slots=True)
class ExpertExample:
    state: np.ndarray
    actions: np.ndarray
    tiles: np.ndarray
    target: np.ndarray
    base_logits: np.ndarray


class ExpertMemory:
    """Reservoir memory bounded independently of the game state space."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.rng = rng
        self.examples: list[ExpertExample] = []
        self.seen = 0

    def add(self, example: ExpertExample) -> None:
        self.seen += 1
        if len(self.examples) < self.capacity:
            self.examples.append(example)
            return
        replace = self.rng.randrange(self.seen)
        if replace < self.capacity:
            self.examples[replace] = example


class ExpertIterationTrainer:
    """Imitate a strong bounded search while sampling varied opponent states."""

    def __init__(
        self,
        game: GameConfig,
        search: SearchConfig,
        config: ExpertIterationConfig,
        *,
        initial_policy: LinearBlueprintPolicy | None = None,
    ) -> None:
        if not 2 <= game.players <= 6:
            raise ValueError("expert iteration supports 2..6 players")
        self.game = game
        self.search = search
        self.config = config
        self.rng = random.Random(config.seed)
        expected_policy_config = LinearFeatureConfig(
            config.strength_buckets,
            tile_buckets=config.tile_buckets,
            players=game.players,
            # The prior is configurable: neutral training must not inherit
            # the hand-crafted heads-up heuristic.
            compact_prior=(config.compact_prior and game.players == 2),
        )
        if initial_policy is not None and initial_policy.config != expected_policy_config:
            raise ValueError(
                "initial blueprint configuration does not match training configuration"
            )
        self.policy = initial_policy or LinearBlueprintPolicy(expected_policy_config)
        self.memory = ExpertMemory(config.memory_examples, self.rng)
        # In a continuation run the previous policy is also the first
        # snapshot, so validation can compare the new policy with its actual
        # starting point rather than with an unrelated fresh model.
        self.snapshots: list[LinearBlueprintPolicy] = (
            [self.policy.copy()] if initial_policy is not None else []
        )
        self.initial_policy_loaded = initial_policy is not None
        self.information_state_keys: set[tuple[object, ...]] = set()
        self.information_state_visits = 0
        self.new_information_states = 0
        self.repeated_information_states = 0
        self.fit_update_events = 0
        self.iteration = 0
        self.stop_requested = False
        self.stop_signal: int | None = None
        self.training_started = time.perf_counter()
        self.teacher = ResolvingPolicy(
            None,
            self.policy,
            self.search,
            blueprint_weight=self.config.teacher_blueprint_weight,
        )

    def _opponent(self, kind: OpponentKind):
        if kind == "self":
            return self.policy
        if kind == "heuristic":
            return HeuristicPolicy()
        if kind == "random":
            return RandomPolicy()
        return self.snapshots[-1] if self.snapshots else self.policy

    def _collect_game(self, opponent_kind: OpponentKind, seed: int) -> int:
        rng = random.Random(seed)
        state = RoundState(self.game)
        state.apply_chance(state.sample_chance(rng))
        opponent = self._opponent(opponent_kind)
        # One seat is the learner.  All other seats use the current opponent
        # from the rotating pool; this keeps the examples tied to a single
        # information set while still exposing the policy to multiplayer
        # histories and relative seats.
        learner_seat = seed % self.game.players
        examples = 0
        while not state.is_terminal:
            info = state.information_state()
            if state.current_player == learner_seat:
                key = info.key()
                self.information_state_visits += 1
                if key in self.information_state_keys:
                    self.repeated_information_states += 1
                else:
                    self.information_state_keys.add(key)
                    self.new_information_states += 1
                target = np.asarray(self.teacher.probabilities_state(state), dtype=np.float64)
                self.memory.add(
                    ExpertExample(
                        self.policy.state_features(info).astype(np.float32),
                        self.policy.action_features(info).astype(np.float32),
                        self.policy.tile_indices(info).astype(np.int32),
                        target.astype(np.float32),
                        self.policy.base_logits(info).astype(np.float32),
                    )
                )
                examples += 1
            # Only the learner is trained; the remaining seats form the
            # environment.  The resolver itself still observes the full state
            # and evaluates the learner's current action distribution.
            actor_policy = self.policy if state.current_player == learner_seat else opponent
            probabilities = actor_policy.probabilities(info)
            action = rng.choices(info.legal_actions, weights=probabilities, k=1)[0]
            state.apply_action(action)
        return examples

    def _fit(self) -> float:
        if not self.memory.examples:
            return 0.0
        losses: list[float] = []
        indices = list(range(len(self.memory.examples)))
        for _ in range(self.config.epochs_per_iteration):
            self.fit_update_events += len(self.memory.examples)
            self.rng.shuffle(indices)
            for index in indices:
                example = self.memory.examples[index]
                # Gli esempi sono conservati in float32 per risparmiare memoria;
                # qui usiamo float64 solo durante il piccolo aggiornamento numerico.
                state = example.state.astype(np.float64, copy=False)
                actions = example.actions.astype(np.float64, copy=False)
                base_count = self.policy.base_parameter_count
                tiles = example.tiles.astype(np.int64, copy=False)

                # Anche durante l'addestramento calcoliamo direttamente i
                # punteggi: non serve costruire la matrice completa delle feature.
                logits = self.policy.linear_logits_from_features(state, actions)

                # Se le tile non ci sono, evitiamo la creazione del loro contributo
                # nullo. Con le tile attive aggiungiamo invece i loro pesi normalmente.
                if tiles.shape[1]:
                    logits += self.policy.tile_logits_from_indices(tiles)
                if self.policy.config.compact_prior:
                    logits += example.base_logits.astype(np.float64, copy=False)
                probabilities = np.exp(
                    np.clip(logits / self.policy.config.temperature, -60.0, 60.0)
                )
                probabilities /= float(np.sum(probabilities))
                target = example.target.astype(np.float64, copy=False)
                losses.append(float(-np.sum(target * np.log(np.maximum(probabilities, 1e-12)))))
                # Il gradiente della parte che unisce stato e azione può essere
                # scritto come un prodotto tra due vettori. Così evitiamo di
                # creare e poi moltiplicare una matrice temporanea per ogni esempio.
                error = probabilities - target
                action_gradient = error @ actions
                gradient = np.concatenate(
                    (
                        np.outer(state, action_gradient).reshape(-1),
                        action_gradient,
                        np.asarray((float(np.sum(error)),), dtype=np.float64),
                    )
                )
                gradient += self.config.l2 * self.policy.weights[:base_count]
                self.policy.weights[:base_count] -= self.config.learning_rate * gradient
                if tiles.shape[1]:
                    addresses, inverse = np.unique(tiles.ravel(), return_inverse=True)
                    tile_gradient = np.zeros(len(addresses), dtype=np.float64)
                    np.add.at(
                        tile_gradient,
                        inverse,
                        np.repeat(probabilities - target, tiles.shape[1]),
                    )
                    tile_weights = self.policy.weights[base_count:]
                    tile_weights[addresses] -= self.config.learning_rate * (
                        tile_gradient + self.config.l2 * tile_weights[addresses]
                    )
        return float(sum(losses) / len(losses)) if losses else 0.0

    def train(self) -> dict[str, object]:
        pool: tuple[OpponentKind, ...] = ("self", "heuristic", "random", "snapshot")
        reports: list[dict[str, object]] = []
        previous_visits = 0
        previous_new = 0
        previous_repeated = 0
        previous_updates = 0
        try:
            while self.iteration < self.config.iterations:
                self.iteration += 1
                iteration = self.iteration
                collected = 0
                by_opponent: dict[str, int] = {}
                for game_index in range(self.config.games_per_iteration):
                    kind = pool[(iteration + game_index) % len(pool)]
                    count = self._collect_game(kind, self.rng.randrange(2**63))
                    collected += count
                    by_opponent[kind] = by_opponent.get(kind, 0) + count
                loss = self._fit()
                self.snapshots.append(self.policy.copy())
                reports.append(
                    {
                        "iteration": iteration,
                        "collected_examples": collected,
                        "memory_examples": len(self.memory.examples),
                        "memory_seen": self.memory.seen,
                        "by_opponent": by_opponent,
                        "cross_entropy": loss,
                        "information_state_visits": self.information_state_visits - previous_visits,
                        "new_information_states": self.new_information_states - previous_new,
                        "repeated_information_states": self.repeated_information_states
                        - previous_repeated,
                        "new_state_percentage": (
                            (self.new_information_states - previous_new) / collected * 100.0
                            if collected
                            else 0.0
                        ),
                        "fit_update_events": self.fit_update_events - previous_updates,
                    }
                )
                previous_visits = self.information_state_visits
                previous_new = self.new_information_states
                previous_repeated = self.repeated_information_states
                previous_updates = self.fit_update_events
                if self.stop_requested:
                    break
        finally:
            teacher_stats = self.teacher.stats()
            self.teacher.close()
        if resource is None:
            rss_bytes = 0
        else:
            rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_bytes = rss_kib * (1024 if sys.platform != "darwin" else 1024 * 1024)
        elapsed = time.perf_counter() - self.training_started
        return {
            "kind": "table_free_expert_iteration_v1",
            "game": self.game.to_dict(),
            "config": asdict(self.config),
            "policy": self.policy.stats(),
            "iterations": reports,
            "continuation": {
                "initial_policy_loaded": self.initial_policy_loaded,
                "initial_snapshot_available": bool(self.snapshots),
            },
            "training_statistics": {
                "elapsed_seconds": elapsed,
                "max_rss_bytes": rss_bytes,
                "max_rss_gib": rss_bytes / (1024**3),
                "information_states_visited_total": self.information_state_visits,
                "information_states_unique": len(self.information_state_keys),
                # A factorized blueprint deliberately has no row per
                # information set: its strategy domain is the full game
                # state space and is not finitely enumerated by training.
                "information_states_total": None,
                "information_state_coverage_percent": None,
                "information_state_space_note": (
                    "not enumerable for the factorized policy; no explicit information-state table"
                ),
                "new_information_states": self.new_information_states,
                "repeated_information_states": self.repeated_information_states,
                "new_state_percentage": (
                    self.new_information_states / self.information_state_visits * 100.0
                    if self.information_state_visits
                    else 0.0
                ),
                "repeated_state_percentage": (
                    self.repeated_information_states / self.information_state_visits * 100.0
                    if self.information_state_visits
                    else 0.0
                ),
                "fit_update_events": self.fit_update_events,
                "updates_per_unique_information_state": (
                    self.fit_update_events / len(self.information_state_keys)
                    if self.information_state_keys
                    else 0.0
                ),
                "teacher": teacher_stats,
            },
            "warning": (
                "teacher search and the opponent pool improve robustness, but this is "
                "not an exploitability certificate or an exact equilibrium solver"
            ),
        }

    def request_stop(self, signal_number: int | None = None) -> None:
        """Ask the trainer to checkpoint after the current simulated game."""
        self.stop_requested = True
        self.stop_signal = signal_number

    def save_checkpoint(self, path: Path) -> None:
        """Persist policy, reservoir and counters for a later long run.

        A pilot is useful only if its visited examples survive the Slurm job.
        The checkpoint is deliberately independent from the final JSON
        blueprint so production can resume the teacher memory as well as the
        learned weights.
        """
        payload = {
            "format": "presine-expert-iteration-checkpoint-v2",
            "game": self.game.to_dict(),
            "search": self.search.to_dict(),
            "config": asdict(self.config),
            "policy_config": asdict(self.policy.config),
            "policy_weights": self.policy.weights,
            "snapshots": [snapshot.weights for snapshot in self.snapshots],
            "memory_examples": self.memory.examples,
            "memory_seen": self.memory.seen,
            "iteration": self.iteration,
            "rng_state": self.rng.getstate(),
            "information_state_keys": tuple(self.information_state_keys),
            "information_state_visits": self.information_state_visits,
            "new_information_states": self.new_information_states,
            "repeated_information_states": self.repeated_information_states,
            "fit_update_events": self.fit_update_events,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        with gzip.open(temporary, "wb", compresslevel=1) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def restore(cls, path: Path) -> ExpertIterationTrainer:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("format") != "presine-expert-iteration-checkpoint-v2":
            raise ValueError("unsupported expert iteration checkpoint format")
        game = GameConfig.from_dict(dict(payload["game"]))
        search = SearchConfig.from_dict(dict(payload["search"]))
        config = ExpertIterationConfig(**dict(payload["config"]))
        policy = LinearBlueprintPolicy(
            LinearFeatureConfig(**dict(payload["policy_config"])),
            weights=np.asarray(payload["policy_weights"], dtype=np.float64),
        )
        trainer = cls(game, search, config, initial_policy=policy)
        trainer.snapshots = [
            LinearBlueprintPolicy(policy.config, weights=np.asarray(weights, dtype=np.float64))
            for weights in payload.get("snapshots", ())
        ] or [policy.copy()]
        trainer.memory.examples = list(payload.get("memory_examples", ()))
        trainer.memory.seen = int(payload.get("memory_seen", len(trainer.memory.examples)))
        trainer.iteration = int(payload.get("iteration", 0))
        trainer.rng.setstate(payload["rng_state"])
        trainer.information_state_keys = set(payload.get("information_state_keys", ()))
        trainer.information_state_visits = int(payload.get("information_state_visits", 0))
        trainer.new_information_states = int(payload.get("new_information_states", 0))
        trainer.repeated_information_states = int(payload.get("repeated_information_states", 0))
        trainer.fit_update_events = int(payload.get("fit_update_events", 0))
        trainer.initial_policy_loaded = True
        return trainer

    def _validation_matrix(self, candidate: object, *, games: int, seed: int) -> dict[str, object]:
        opponents = {
            "heuristic": HeuristicPolicy(),
            "random": RandomPolicy(),
            "snapshot": self.snapshots[0] if self.snapshots else self.policy,
        }
        report: dict[str, object] = {}
        for index, (name, opponent) in enumerate(opponents.items()):
            if self.game.players == 2:
                report[name] = evaluate_round(
                    self.game,
                    candidate,
                    opponent,
                    games=games,
                    seed=seed + index * 1_000_003,
                    rotate_seats=True,
                    rotate_starting_player=True,
                ).to_dict()
            else:
                report[name] = self._multiplayer_validation(
                    candidate,
                    opponent,
                    games=games,
                    seed=seed + index * 1_000_003,
                )
        return report

    def _multiplayer_validation(
        self, candidate: object, opponent: object, *, games: int, seed: int
    ) -> dict[str, object]:
        margins: list[float] = []
        candidate_errors: list[float] = []
        best_opponent_errors: list[float] = []
        wins = ties = losses = 0
        for game_index in range(games):
            candidate_seat = game_index % self.game.players
            config = GameConfig(
                players=self.game.players,
                hand_size=self.game.hand_size,
                starting_player=(self.game.starting_player + game_index) % self.game.players,
                payoff=self.game.payoff,
            )
            policies = tuple(
                candidate if seat == candidate_seat else opponent
                for seat in range(self.game.players)
            )
            result = play_multiplayer_round(config, policies, seed=seed + game_index * 104_729)
            own = float(result.errors[candidate_seat])
            best_other = float(
                min(value for seat, value in enumerate(result.errors) if seat != candidate_seat)
            )
            margin = best_other - own
            margins.append(margin)
            candidate_errors.append(own)
            best_opponent_errors.append(best_other)
            if margin > 0:
                wins += 1
            elif margin < 0:
                losses += 1
            else:
                ties += 1
        mean = sum(margins) / games
        variance = sum((value - mean) ** 2 for value in margins) / (games - 1) if games > 1 else 0.0
        half_width = 1.96 * (variance / games) ** 0.5
        return {
            "games": games,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / games,
            "tie_rate": ties / games,
            "loss_rate": losses / games,
            "mean_error_margin": mean,
            "error_margin_ci95": (mean - half_width, mean + half_width),
            "mean_candidate_errors": sum(candidate_errors) / games,
            "mean_best_opponent_errors": sum(best_opponent_errors) / games,
            "players": self.game.players,
        }

    def validation(self, *, games: int, seed: int) -> dict[str, object]:
        standalone = self._validation_matrix(self.policy, games=games, seed=seed)
        with ResolvingPolicy(None, self.policy, self.search, blueprint_weight=0.20) as resolved:
            resolver = self._validation_matrix(resolved, games=games, seed=seed ^ 0x9E3779B9)
            resolver_stats = resolved.stats()
        return {
            "standalone": standalone,
            "resolved": resolver,
            "resolved_stats": resolver_stats,
        }
