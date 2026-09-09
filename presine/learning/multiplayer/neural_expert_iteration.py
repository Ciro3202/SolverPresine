from __future__ import annotations

"""Resource-bounded neural expert iteration for multiplayer Presine.

Independent complete games are the unit of CPU scheduling.  Search labels are
generated from a frozen policy snapshot, then the shared policy/value network
is fitted in the parent process.  This keeps parallel workers independent and
makes CPU reallocation between R1..R5 immediate at every iteration.
"""

import gzip
import hashlib
import io
import math
import os
import pickle
import random
import signal
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

from presine.evaluation.multiplayer import play_multiplayer_round
from presine.evaluation.statistics import adaptive_sample_complete, mean_ci95
from presine.game.cards import ACE_OF_DENARI, NUM_CARDS
from presine.game.config import GameConfig
from presine.game.state import Phase, RoundState
from presine.learning.deep_cfr.encoding import ACTION_DIM, action_id
from presine.policies.heuristic import HeuristicPolicy
from presine.policies.random import RandomPolicy
from presine.search.config import SearchConfig
from presine.search.ismcts import visit_probabilities
from presine.search.policy import _run_presine_search
from presine.search.presine_adapter import TREE_INFORMATION_SEMANTICS

from .decision_focus import decision_categories, focus_multiplier
from .multiplayer_encoding import encode_information_state, encoding_version
from .neural_blueprint import (
    NeuralBlueprintArchitecture,
    NeuralMultiplayerBlueprint,
    SharedRoundPolicyValueNetwork,
)

OpponentKind = Literal["self", "heuristic", "random", "snapshot"]
CHECKPOINT_FORMAT = "presine-neural-multiplayer-expert-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class AdaptiveTeacherConfig:
    min_simulations: int = 512
    simulation_chunk: int = 512
    max_simulations: int = 2_048
    hard_max_simulations: int = 4_096
    min_particles: int = 128
    max_particles: int = 256
    min_ess_ratio: float = 0.35
    stability_js: float = 0.012
    stable_checks: int = 2
    hard_entropy_ratio: float = 0.72
    max_neural_guidance: float = 0.65
    max_neural_rollout_guidance: float = 0.10
    guidance_warmup_iterations: int = 20
    guided_rollout_depth: int = 32

    def __post_init__(self) -> None:
        positive = (
            self.min_simulations,
            self.simulation_chunk,
            self.max_simulations,
            self.hard_max_simulations,
            self.min_particles,
            self.max_particles,
            self.stable_checks,
            self.guidance_warmup_iterations,
            self.guided_rollout_depth,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("adaptive teacher counts must be positive")
        if not self.min_simulations <= self.max_simulations <= self.hard_max_simulations:
            raise ValueError("teacher simulation limits are inconsistent")
        if self.min_particles > self.max_particles:
            raise ValueError("teacher particle limits are inconsistent")
        if not 0 < self.min_ess_ratio <= 1:
            raise ValueError("min_ess_ratio must be in (0, 1]")
        if self.stability_js < 0 or not 0 <= self.hard_entropy_ratio <= 1:
            raise ValueError("invalid teacher stability thresholds")
        if not 0.0 <= self.max_neural_guidance <= 1.0:
            raise ValueError("max_neural_guidance must be in [0, 1]")
        if not 0.0 <= self.max_neural_rollout_guidance <= 1.0:
            raise ValueError("max_neural_rollout_guidance must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SatisfactionConfig:
    min_examples_per_round: int = 20_000
    min_examples_per_observed_stratum: int = 64
    max_validation_cross_entropy: float = 1.35
    max_policy_kl: float = 0.020
    min_error_margin_ci95: float = -0.25
    min_mean_error_margin: float | None = None
    stable_checks: int = 3
    validation_every: int = 4
    validation_games: int = 32
    validation_games_max: int = 512
    validation_ci_half_width: float = 0.10
    validation_batch_size: int = 32
    maintenance_fraction: float = 0.05
    max_mean_decision_regret: float = 0.08
    min_focus_examples: int = 256

    def __post_init__(self) -> None:
        if self.min_examples_per_round <= 0 or self.min_examples_per_observed_stratum <= 0:
            raise ValueError("satisfaction coverage thresholds must be positive")
        if self.max_validation_cross_entropy <= 0 or self.max_policy_kl < 0:
            raise ValueError("invalid satisfaction loss thresholds")
        if self.stable_checks <= 0 or self.validation_every <= 0 or self.validation_games <= 0:
            raise ValueError("satisfaction check counts must be positive")
        if self.validation_games_max < self.validation_games:
            raise ValueError("validation_games_max must be at least validation_games")
        if self.validation_ci_half_width <= 0 or self.validation_batch_size <= 0:
            raise ValueError("invalid adaptive validation settings")
        if not 0 < self.maintenance_fraction <= 1:
            raise ValueError("maintenance_fraction must be in (0, 1]")
        if self.max_mean_decision_regret < 0 or self.min_focus_examples < 0:
            raise ValueError("invalid decision-quality thresholds")


@dataclass(frozen=True, slots=True)
class NeuralExpertIterationConfig:
    iterations: int = 1_000_000
    games_per_iteration: int = 32
    cpu_workers: int = 8
    train_steps_per_iteration: int = 256
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    value_loss_weight: float = 0.35
    memory_examples: int = 50_000
    validation_examples: int = 5_000
    validation_fraction: float = 0.10
    round_weights: tuple[float, ...] = (1.0, 1.0, 2.0, 2.0, 2.0)
    ace_loss_weight: float = 2.0
    pre_ace_loss_weight: float = 1.5
    focus_replay_weight: float = 3.0
    focus_loss_weight: float = 1.5
    force_ace_deal: bool = False
    repair_round: int | None = None
    repair_continue: bool = False
    repair_plateau_patience: int = 0
    repair_plateau_min_delta: float = 0.005
    plateau_patience: int = 0
    plateau_min_delta: float = 0.005
    snapshot_every: int = 4
    max_snapshots: int = 8
    checkpoint_minutes: float = 25.0
    max_runtime_minutes: float | None = None
    report_history: int = 512
    unique_counter_limit: int = 2_000_000
    seed: int = 20260826
    device: str = "cpu"
    architecture: NeuralBlueprintArchitecture = field(default_factory=NeuralBlueprintArchitecture)
    teacher: AdaptiveTeacherConfig = field(default_factory=AdaptiveTeacherConfig)
    satisfaction: SatisfactionConfig = field(default_factory=SatisfactionConfig)

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.games_per_iteration,
            self.cpu_workers,
            self.train_steps_per_iteration,
            self.batch_size,
            self.memory_examples,
            self.validation_examples,
            self.snapshot_every,
            self.max_snapshots,
            self.report_history,
            self.unique_counter_limit,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("neural expert iteration counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.value_loss_weight < 0:
            raise ValueError("invalid neural optimization parameters")
        if self.ace_loss_weight < 1.0 or self.pre_ace_loss_weight < 1.0:
            raise ValueError("Ace loss weights must be at least 1")
        if self.focus_replay_weight < 1.0 or self.focus_loss_weight < 1.0:
            raise ValueError("focus weights must be at least 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if len(self.round_weights) != 5 or any(value < 0 for value in self.round_weights):
            raise ValueError("round_weights must contain five non-negative values")
        if not any(value > 0 for value in self.round_weights):
            raise ValueError("at least one neural round weight must be positive")
        if self.repair_round is not None and not 1 <= self.repair_round <= 5:
            raise ValueError("repair_round must be in 1..5")
        if self.repair_continue and self.repair_round is None:
            raise ValueError("repair_continue requires repair_round")
        if (
            self.repair_plateau_patience < 0
            or self.repair_plateau_min_delta < 0
            or self.plateau_patience < 0
            or self.plateau_min_delta < 0
        ):
            raise ValueError("invalid repair plateau parameters")
        if self.checkpoint_minutes <= 0:
            raise ValueError("checkpoint_minutes must be positive")
        if self.max_runtime_minutes is not None and self.max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be positive when set")


@dataclass(slots=True)
class NeuralExpertExample:
    state: np.ndarray
    policy_target: np.ndarray
    legal_mask: np.ndarray
    value_target: np.ndarray
    hand_size: int
    phase: int
    relative_starter: int
    opponent: str
    loss_weight: float = 1.0
    replay_weight: float = 1.0
    categories: tuple[str, ...] = ("all",)
    action_value_target: np.ndarray | None = None


def _sample_ace_deal(game: GameConfig, rng: random.Random) -> tuple[tuple[int, ...], ...]:
    """Sample a normal deal conditioned on the 30 being dealt.

    Ownership and hand slot are random; all remaining cards retain the normal
    without-replacement distribution.
    """
    count = game.players * game.hand_size
    cards = [
        ACE_OF_DENARI,
        *rng.sample([c for c in range(NUM_CARDS) if c != ACE_OF_DENARI], count - 1),
    ]
    rng.shuffle(cards)
    n = game.hand_size
    return tuple(
        tuple(sorted(cards[player * n : (player + 1) * n])) for player in range(game.players)
    )


class _Reservoir:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.rng = rng
        self.examples: list[NeuralExpertExample] = []
        self.seen = 0

    def add(self, example: NeuralExpertExample) -> None:
        self.seen += 1
        if len(self.examples) < self.capacity:
            self.examples.append(example)
            return
        replacement = self.rng.randrange(self.seen)
        if replacement < self.capacity:
            self.examples[replacement] = example


class _FixedValidationMemory:
    """Stable held-out anchors: once full, examples are never replaced."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.examples: list[NeuralExpertExample] = []

    def add(self, example: NeuralExpertExample) -> bool:
        if len(self.examples) >= self.capacity:
            return False
        self.examples.append(example)
        return True


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    p = np.maximum(np.asarray(first, dtype=np.float64), 1e-12)
    q = np.maximum(np.asarray(second, dtype=np.float64), 1e-12)
    p /= p.sum()
    q /= q.sum()
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


class AdaptiveSearchTeacher:
    """Spend search and belief budget only until a root distribution stabilizes."""

    def __init__(
        self,
        search: SearchConfig,
        config: AdaptiveTeacherConfig,
        *,
        seed: int,
        guidance: NeuralMultiplayerBlueprint | None = None,
        guidance_fraction: float = 0.0,
        rollout_guidance_fraction: float = 0.0,
    ) -> None:
        if search.mode != "ismcts":
            raise ValueError("the adaptive multiplayer teacher currently requires ISMCTS")
        self.search = search
        self.config = config
        self.rng = random.Random(seed)
        self.guidance = guidance
        self.guidance_fraction = guidance_fraction
        self.rollout_guidance_fraction = rollout_guidance_fraction

    def targets_state(
        self, state: RoundState
    ) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, float | int]]:
        """Return visit policy and searched action values for one root.

        Values are retained for held-out decision-regret monitoring.  They do
        not expose hidden cards: every search starts from the acting player's
        information-set belief.
        """
        actions = state.information_state().legal_actions
        visits = np.zeros(len(actions), dtype=np.int64)
        value_sums = np.zeros(len(actions), dtype=np.float64)
        simulations = 0
        searches = 0
        stable = 0
        particles = min(self.config.min_particles, self.config.max_particles)
        previous: np.ndarray | None = None
        last_js = math.inf
        ess_total = 0.0
        # Ace decisions are evaluated with the deeper teacher budget.  This is
        # a position-level focus, not an artificial deal distribution.
        root_categories = decision_categories(state.information_state())
        hard_position = (
            state.phase == Phase.ACE_CHOICE
            or (state.phase == Phase.PLAY and ACE_OF_DENARI in state.legal_actions())
            or "high_cups" in root_categories
            or "middle_cups" in root_categories
        )
        focused_root = hard_position

        while simulations < self.config.hard_max_simulations:
            normal_limit = self.config.max_simulations
            active_limit = self.config.hard_max_simulations if hard_position else normal_limit
            if simulations >= active_limit:
                break
            chunk = (
                self.config.min_simulations if simulations == 0 else self.config.simulation_chunk
            )
            chunk = min(chunk, active_limit - simulations)
            progress = (
                self.guidance_fraction / self.config.max_neural_guidance
                if self.config.max_neural_guidance
                else 0.0
            )
            guided_depth = round(
                self.search.ismcts.rollout_depth * (1.0 - progress)
                + self.config.guided_rollout_depth * progress
            )
            run_config = replace(
                self.search,
                workers=1,
                belief=replace(self.search.belief, particles=particles),
                ismcts=replace(
                    self.search.ismcts,
                    simulations=chunk,
                    rollout_depth=max(1, guided_depth),
                ),
            )
            result, diagnostics = _run_presine_search(
                state,
                state.current_player,
                run_config,
                self.rng.randrange(2**63),
                self.guidance,
                self.rollout_guidance_fraction,
                self.guidance_fraction,
            )
            if result.actions != actions:
                raise RuntimeError("adaptive teacher returned different root actions")
            chunk_visits = np.asarray(result.visits, dtype=np.int64)
            visits += chunk_visits
            value_sums += np.asarray(result.values, dtype=np.float64) * chunk_visits
            simulations += result.simulations
            searches += 1
            ess = float(diagnostics.get("effective_sample_size", 0.0))
            ess_total += ess
            if (
                particles < self.config.max_particles
                and ess / max(particles, 1) < self.config.min_ess_ratio
            ):
                particles = self.config.max_particles

            current = np.asarray(
                visit_probabilities(visits.tolist(), self.search.ismcts.temperature),
                dtype=np.float64,
            )
            if previous is not None:
                last_js = _js_divergence(previous, current)
                stable = stable + 1 if last_js <= self.config.stability_js else 0
            previous = current
            required_minimum = (
                self.config.max_simulations if focused_root else self.config.min_simulations
            )
            if simulations >= required_minimum and stable >= self.config.stable_checks:
                break
            if simulations >= normal_limit and len(current) > 1:
                entropy = -float(np.sum(current * np.log(np.maximum(current, 1e-12))))
                ratio = entropy / math.log(len(current))
                hard_position = ratio >= self.config.hard_entropy_ratio
                if not hard_position:
                    break

        if previous is None:
            previous = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
        action_values = np.divide(
            value_sums,
            visits,
            out=np.zeros_like(value_sums),
            where=visits > 0,
        )
        return (
            tuple(float(value) for value in previous),
            tuple(action_values),
            {
                "simulations": simulations,
                "searches": searches,
                "particles": particles,
                "mean_ess": ess_total / searches if searches else 0.0,
                "last_js": last_js,
                "hard_position": int(hard_position),
            },
        )

    def probabilities_state(
        self, state: RoundState
    ) -> tuple[tuple[float, ...], dict[str, float | int]]:
        """Compatibility wrapper for callers that need only policy targets."""

        probabilities, _, diagnostics = self.targets_state(state)
        return probabilities, diagnostics


_WORKER_CURRENT: NeuralMultiplayerBlueprint | None = None
_WORKER_SNAPSHOT: NeuralMultiplayerBlueprint | None = None
_WORKER_SEARCH: SearchConfig | None = None
_WORKER_TEACHER: AdaptiveTeacherConfig | None = None
_WORKER_GUIDANCE_FRACTION = 0.0
_WORKER_ROLLOUT_GUIDANCE_FRACTION = 0.0
_WORKER_FORCE_ACE = False
_WORKER_ACE_LOSS_WEIGHT = 2.0
_WORKER_PRE_ACE_LOSS_WEIGHT = 1.5
_WORKER_FOCUS_REPLAY_WEIGHT = 3.0
_WORKER_FOCUS_LOSS_WEIGHT = 1.5


def _worker_initializer(
    players: int,
    architecture: NeuralBlueprintArchitecture,
    current_state: dict[str, torch.Tensor],
    snapshot_state: dict[str, torch.Tensor] | None,
    search: SearchConfig,
    teacher: AdaptiveTeacherConfig,
    iteration: int,
    force_ace: bool = True,
    ace_loss_weight: float = 4.0,
    pre_ace_loss_weight: float = 2.0,
    focus_replay_weight: float = 3.0,
    focus_loss_weight: float = 1.5,
) -> None:
    global _WORKER_CURRENT, _WORKER_SNAPSHOT, _WORKER_SEARCH, _WORKER_TEACHER
    global _WORKER_GUIDANCE_FRACTION
    global _WORKER_ROLLOUT_GUIDANCE_FRACTION, _WORKER_FORCE_ACE
    global _WORKER_ACE_LOSS_WEIGHT, _WORKER_PRE_ACE_LOSS_WEIGHT
    global _WORKER_FOCUS_REPLAY_WEIGHT, _WORKER_FOCUS_LOSS_WEIGHT
    torch.set_num_threads(1)
    current_network = SharedRoundPolicyValueNetwork(players, architecture)
    current_network.load_state_dict(current_state)
    _WORKER_CURRENT = NeuralMultiplayerBlueprint(players, current_network)
    if snapshot_state is not None:
        snapshot_network = SharedRoundPolicyValueNetwork(players, architecture)
        snapshot_network.load_state_dict(snapshot_state)
        _WORKER_SNAPSHOT = NeuralMultiplayerBlueprint(players, snapshot_network)
    else:
        _WORKER_SNAPSHOT = None
    _WORKER_SEARCH = search
    _WORKER_TEACHER = teacher
    _WORKER_FORCE_ACE = force_ace
    _WORKER_ACE_LOSS_WEIGHT = ace_loss_weight
    _WORKER_PRE_ACE_LOSS_WEIGHT = pre_ace_loss_weight
    _WORKER_FOCUS_REPLAY_WEIGHT = focus_replay_weight
    _WORKER_FOCUS_LOSS_WEIGHT = focus_loss_weight
    progress = min(1.0, iteration / teacher.guidance_warmup_iterations)
    _WORKER_GUIDANCE_FRACTION = teacher.max_neural_guidance * progress
    _WORKER_ROLLOUT_GUIDANCE_FRACTION = teacher.max_neural_rollout_guidance * progress


def _collect_game_worker(task: tuple[int, OpponentKind, int, int]) -> dict[str, object]:
    hand_size, opponent_kind, seed, players = task
    if _WORKER_CURRENT is None or _WORKER_SEARCH is None or _WORKER_TEACHER is None:
        raise RuntimeError("multiplayer collection worker was not initialized")
    rng = random.Random(seed)
    learner = seed % players
    game = GameConfig(
        players=players,
        hand_size=hand_size,
        starting_player=(seed // players) % players,
    )
    state = RoundState(game)
    deal = _sample_ace_deal(game, rng) if _WORKER_FORCE_ACE else state.sample_chance(rng)
    state.apply_chance(deal)
    if opponent_kind == "heuristic":
        opponent: object = HeuristicPolicy()
    elif opponent_kind == "random":
        opponent = RandomPolicy()
    elif opponent_kind == "snapshot" and _WORKER_SNAPSHOT is not None:
        opponent = _WORKER_SNAPSHOT
    else:
        opponent = _WORKER_CURRENT
    teacher = AdaptiveSearchTeacher(
        _WORKER_SEARCH,
        _WORKER_TEACHER,
        seed=seed ^ 0x5DEECE66D,
        guidance=_WORKER_CURRENT,
        guidance_fraction=_WORKER_GUIDANCE_FRACTION,
        rollout_guidance_fraction=_WORKER_ROLLOUT_GUIDANCE_FRACTION,
    )
    pending: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            int,
            int,
            str,
            float,
            float,
            tuple[str, ...],
        ]
    ] = []
    totals: Counter[str] = Counter()
    scalar_totals: Counter[str] = Counter()
    while not state.is_terminal:
        info = state.information_state()
        if state.current_player == learner:
            target, action_values, stats = teacher.targets_state(state)
            full_target = np.zeros(ACTION_DIM, dtype=np.float16)
            full_action_values = np.zeros(ACTION_DIM, dtype=np.float16)
            mask = np.zeros(ACTION_DIM, dtype=np.bool_)
            for action, probability, action_value in zip(info.legal_actions, target, action_values):
                index = action_id(info, action)
                full_target[index] = probability
                full_action_values[index] = action_value
                mask[index] = True
            categories = decision_categories(info)
            focus = focus_multiplier(info, _WORKER_FOCUS_LOSS_WEIGHT)
            loss_weight = focus * (
                _WORKER_ACE_LOSS_WEIGHT
                if state.phase == Phase.ACE_CHOICE
                else _WORKER_PRE_ACE_LOSS_WEIGHT
                if state.phase == Phase.PLAY
                and state.current_trick
                and ACE_OF_DENARI in state.hands[state.current_player]
                else 1.0
            )
            replay_weight = focus_multiplier(info, _WORKER_FOCUS_REPLAY_WEIGHT)
            pending.append(
                (
                    encode_information_state(info).astype(np.float16),
                    full_target,
                    mask,
                    full_action_values,
                    hand_size,
                    int(info.phase),
                    (info.starting_player - info.player) % players,
                    opponent_kind,
                    loss_weight,
                    replay_weight,
                    categories,
                )
            )
            totals["decisions"] += 1
            totals["hard_positions"] += int(stats["hard_position"])
            scalar_totals["simulations"] += int(stats["simulations"])
            scalar_totals["searches"] += int(stats["searches"])
            scalar_totals["particles"] += int(stats["particles"])
            scalar_totals["effective_particles_sum"] += float(stats["mean_ess"])
        actor_policy = _WORKER_CURRENT if state.current_player == learner else opponent
        probabilities = actor_policy.probabilities(info)  # type: ignore[attr-defined]
        action = rng.choices(info.legal_actions, weights=probabilities, k=1)[0]
        state.apply_action(action)
    utilities = state.utilities()
    relative_values = np.asarray(
        [utilities[(learner + relative) % players] for relative in range(players)],
        dtype=np.float16,
    )
    examples = [
        NeuralExpertExample(
            state=encoded,
            policy_target=target,
            legal_mask=mask,
            value_target=relative_values.copy(),
            hand_size=round_size,
            phase=phase,
            relative_starter=relative_starter,
            opponent=kind,
            loss_weight=loss_weight,
            replay_weight=replay_weight,
            categories=categories,
            action_value_target=action_values,
        )
        for (
            encoded,
            target,
            mask,
            action_values,
            round_size,
            phase,
            relative_starter,
            kind,
            loss_weight,
            replay_weight,
            categories,
        ) in pending
    ]
    worker_rss_bytes = 0
    if resource is not None:
        rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        worker_rss_bytes = rss_kib * (
            1024 if os.name != "posix" or os.uname().sysname != "Darwin" else 1024 * 1024
        )
    totals["max_worker_rss_bytes"] = worker_rss_bytes
    return {
        "hand_size": hand_size,
        "examples": examples,
        "teacher": dict(totals | scalar_totals),
    }


def _tensors_on_cpu(value: object) -> object:
    """Copy nested optimizer/checkpoint tensors to CPU before pickling."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensors_on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensors_on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensors_on_cpu(item) for item in value)
    return value


class DynamicRoundScheduler:
    """Weighted deficit scheduler with a small post-satisfaction maintenance share."""

    def __init__(self, weights: tuple[float, ...], maintenance_fraction: float) -> None:
        self.weights = {hand_size: weights[hand_size - 1] for hand_size in range(1, 6)}
        self.maintenance_fraction = maintenance_fraction
        self.satisfied = {hand_size: False for hand_size in range(1, 6)}
        self.stable_checks = {hand_size: 0 for hand_size in range(1, 6)}

    def allocation(
        self, tasks: int, example_counts: dict[int, int], target_examples: int
    ) -> list[int]:
        scores: dict[int, float] = {}
        for hand_size in range(1, 6):
            if self.satisfied[hand_size]:
                scores[hand_size] = self.weights[hand_size] * self.maintenance_fraction
            else:
                deficit = max(0.10, 1.0 - example_counts.get(hand_size, 0) / target_examples)
                # Larger rounds retain their configured 2x cost/importance weight.
                scores[hand_size] = self.weights[hand_size] * (0.5 + deficit)
        assigned = {hand_size: 0 for hand_size in range(1, 6)}
        result: list[int] = []
        # Never let a satisfied round disappear completely: one maintenance
        # game is the minimum needed to detect later drift and reopen it.
        # The weighted scheduler receives the remaining capacity afterwards.
        for hand_size in range(1, 6):
            if self.satisfied[hand_size] and len(result) < tasks:
                assigned[hand_size] += 1
                result.append(hand_size)
        for _ in range(tasks - len(result)):
            chosen = max(
                scores,
                key=lambda size: scores[size] / (assigned[size] + 1),
            )
            assigned[chosen] += 1
            result.append(chosen)
        return result

    @property
    def all_satisfied(self) -> bool:
        return all(self.satisfied.values())


class NeuralMultiplayerExpertTrainer:
    def __init__(
        self,
        players: int,
        search: SearchConfig,
        config: NeuralExpertIterationConfig,
    ) -> None:
        if not 3 <= players <= 6:
            raise ValueError("multiplayer neural expert iteration requires 3..6 players")
        self.players = players
        self.search = search
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.network = SharedRoundPolicyValueNetwork(players, config.architecture).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.memory = _Reservoir(config.memory_examples, self.rng)
        self.validation_memory = _FixedValidationMemory(config.validation_examples)
        self.strata: Counter[tuple[int, int, int, str]] = Counter()
        self.category_counts: Counter[tuple[int, str]] = Counter()
        self.unique_state_digests: dict[int, set[bytes]] = {
            hand_size: set() for hand_size in range(1, 6)
        }
        self.examples_by_round: Counter[int] = Counter()
        self.scheduler = DynamicRoundScheduler(
            config.round_weights, config.satisfaction.maintenance_fraction
        )
        self.snapshots: list[dict[str, torch.Tensor]] = []
        self.previous_anchor_predictions: dict[int, np.ndarray] = {}
        self.iteration = 0
        self.stop_requested = False
        self.stop_signal: int | None = None
        self.started = time.perf_counter()
        self.last_checkpoint = self.started
        self.reports: list[dict[str, object]] = []
        self.teacher_totals: Counter[str] = Counter()
        self.timing_totals: Counter[str] = Counter()
        self.max_worker_rss_bytes = 0
        self.search_semantics = TREE_INFORMATION_SEMANTICS
        self.repair_prepared_rounds: set[int] = set()
        self.repair_best_mean_margin: float | None = None
        self.repair_plateau_checks = 0
        self.repair_plateau_reached = False
        self.best_monitor_score: dict[int, float] = {}
        self.plateau_checks: Counter[int] = Counter()
        self.plateau_reached: set[int] = set()

    def policy(self) -> NeuralMultiplayerBlueprint:
        return NeuralMultiplayerBlueprint(
            self.players,
            self.network,
            device=str(self.device),
            metadata={"iteration": self.iteration},
        )

    def request_stop(self, signal_number: int | None = None) -> None:
        self.stop_requested = True
        self.stop_signal = signal_number

    def _cpu_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.network.state_dict().items()}

    def _active_rounds(self) -> tuple[int, ...]:
        return (
            (self.config.repair_round,)
            if self.config.repair_round is not None
            else tuple(
                hand_size
                for hand_size, weight in enumerate(self.config.round_weights, start=1)
                if weight > 0
            )
        )

    def _training_complete(self) -> bool:
        if self.config.repair_round is not None and self.config.repair_plateau_patience > 0:
            return self.repair_plateau_reached
        return all(
            self.scheduler.satisfied[size] or size in self.plateau_reached
            for size in self._active_rounds()
        )

    def _configure_repair_optimizer(self, hand_size: int) -> None:
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        trainable = [
            *self.network.policy_heads[str(hand_size)].parameters(),
            *self.network.value_heads[str(hand_size)].parameters(),
        ]
        for parameter in trainable:
            parameter.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def prepare_round_repair(self, hand_size: int) -> None:
        """Start a clean, head-only repair without mutating the other rounds."""
        if hand_size in self.repair_prepared_rounds:
            self._configure_repair_optimizer(hand_size)
            return
        if self.config.repair_round != hand_size:
            raise ValueError("repair round and trainer configuration differ")

        # All old examples for this round were labelled by the previous
        # teacher.  A repair must not mix them with corrected targets, and a
        # head-only optimizer has no use for examples from frozen rounds.
        self.memory = _Reservoir(self.config.memory_examples, self.rng)
        self.validation_memory = _FixedValidationMemory(self.config.validation_examples)
        self.strata = Counter(
            {key: count for key, count in self.strata.items() if key[0] != hand_size}
        )
        self.category_counts = Counter(
            {key: count for key, count in self.category_counts.items() if key[0] != hand_size}
        )
        self.unique_state_digests[hand_size] = set()
        self.examples_by_round[hand_size] = 0
        self.scheduler.satisfied[hand_size] = False
        self.scheduler.stable_checks[hand_size] = 0
        self.previous_anchor_predictions.pop(hand_size, None)
        self.best_monitor_score.pop(hand_size, None)
        self.plateau_checks[hand_size] = 0
        self.plateau_reached.discard(hand_size)
        self.snapshots = []
        self.reports = []
        self.teacher_totals = Counter()
        self.timing_totals = Counter()
        self.max_worker_rss_bytes = 0
        self.stop_requested = False
        self.stop_signal = None
        self.started = time.perf_counter()
        self.last_checkpoint = self.started

        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed ^ (hand_size * 0x9E3779B1))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.seed ^ (hand_size * 0x9E3779B1))
            self.network.policy_heads[str(hand_size)].reset_parameters()
            self.network.value_heads[str(hand_size)].reset_parameters()
        self._configure_repair_optimizer(hand_size)
        self.repair_prepared_rounds.add(hand_size)

    def continue_round_repair(self, hand_size: int) -> None:
        """Continue an existing clean repair without resetting its R1 data."""
        if self.config.repair_round != hand_size:
            raise ValueError("repair round and trainer configuration differ")
        if hand_size not in self.repair_prepared_rounds:
            self.prepare_round_repair(hand_size)
            return
        self.scheduler.satisfied[hand_size] = False
        self.scheduler.stable_checks[hand_size] = 0
        self.stop_requested = False
        self.stop_signal = None
        self.repair_plateau_reached = False
        self.repair_plateau_checks = 0
        self.started = time.perf_counter()
        self.last_checkpoint = self.started
        self._configure_repair_optimizer(hand_size)

    def _tasks(self) -> tuple[list[tuple[int, OpponentKind, int, int]], dict[int, int]]:
        if self.config.repair_round is not None:
            sizes = [self.config.repair_round] * self.config.games_per_iteration
        else:
            sizes = self.scheduler.allocation(
                self.config.games_per_iteration,
                dict(self.examples_by_round),
                self.config.satisfaction.min_examples_per_round,
            )
        pool: tuple[OpponentKind, ...] = ("self", "heuristic", "random", "snapshot")
        tasks: list[tuple[int, OpponentKind, int, int]] = []
        allocation = Counter(sizes)
        for index, hand_size in enumerate(sizes):
            opponent = pool[(self.iteration + index) % len(pool)]
            tasks.append((hand_size, opponent, self.rng.randrange(2**63), self.players))
        return tasks, dict(allocation)

    def _collect(self) -> tuple[int, dict[int, int], dict[str, float | int]]:
        tasks, allocation = self._tasks()
        current_state = self._cpu_state()
        snapshot = self.snapshots[-1] if self.snapshots else None
        if self.config.cpu_workers == 1:
            _worker_initializer(
                self.players,
                self.config.architecture,
                current_state,
                snapshot,
                self.search,
                self.config.teacher,
                self.iteration,
                self.config.force_ace_deal,
                self.config.ace_loss_weight,
                self.config.pre_ace_loss_weight,
                self.config.focus_replay_weight,
                self.config.focus_loss_weight,
            )
            results = [_collect_game_worker(task) for task in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=self.config.cpu_workers,
                initializer=_worker_initializer,
                initargs=(
                    self.players,
                    self.config.architecture,
                    current_state,
                    snapshot,
                    self.search,
                    self.config.teacher,
                    self.iteration,
                    self.config.force_ace_deal,
                    self.config.ace_loss_weight,
                    self.config.pre_ace_loss_weight,
                    self.config.focus_replay_weight,
                    self.config.focus_loss_weight,
                ),
            ) as executor:
                results = list(executor.map(_collect_game_worker, tasks, chunksize=1))
        collected = 0
        iteration_teacher: Counter[str] = Counter()
        for result in results:
            for key, value in dict(result["teacher"]).items():
                if key == "max_worker_rss_bytes":
                    self.max_worker_rss_bytes = max(self.max_worker_rss_bytes, int(value))
                    iteration_teacher[key] = max(iteration_teacher[key], int(value))
                elif key == "effective_particles_sum":
                    iteration_teacher[key] += float(value)
                    self.teacher_totals[key] += float(value)
                else:
                    iteration_teacher[key] += int(value)
                    self.teacher_totals[key] += int(value)
            for example in result["examples"]:  # type: ignore[assignment]
                assert isinstance(example, NeuralExpertExample)
                collected += 1
                self.examples_by_round[example.hand_size] += 1
                self.strata[
                    (
                        example.hand_size,
                        example.phase,
                        example.relative_starter,
                        example.opponent,
                    )
                ] += 1
                for category in getattr(example, "categories", ("all",)):
                    self.category_counts[(example.hand_size, category)] += 1
                digests = self.unique_state_digests[example.hand_size]
                if len(digests) < self.config.unique_counter_limit:
                    digests.add(hashlib.blake2b(example.state.tobytes(), digest_size=8).digest())
                if (
                    len(self.validation_memory.examples) < self.validation_memory.capacity
                    and self.rng.random() < self.config.validation_fraction
                ):
                    self.validation_memory.add(example)
                else:
                    self.memory.add(example)
        effective_sum = float(iteration_teacher.pop("effective_particles_sum", 0.0))
        decisions = int(iteration_teacher.get("decisions", 0))
        iteration_teacher["mean_effective_particles"] = (
            effective_sum / decisions if decisions else 0.0
        )
        return collected, allocation, dict(iteration_teacher)

    def _sample_batch(self) -> list[NeuralExpertExample]:
        eligible = self.memory.examples
        if self.config.repair_round is not None:
            eligible = [
                example for example in eligible if example.hand_size == self.config.repair_round
            ]
        count = min(self.config.batch_size, len(eligible))
        if not eligible:
            return []
        # Stratified replay: naturally occurring Ace positions are sampled
        # more often, while the game/deal distribution itself remains natural.
        weights = [max(1.0, float(getattr(example, "replay_weight", 1.0))) for example in eligible]
        return self.rng.choices(eligible, weights=weights, k=count)

    def _loss_for_examples(
        self, examples: list[NeuralExpertExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        for hand_size in range(1, 6):
            selected = [example for example in examples if example.hand_size == hand_size]
            if not selected:
                continue
            states = torch.from_numpy(
                np.stack([example.state for example in selected]).astype(np.float32)
            ).to(self.device)
            targets = torch.from_numpy(
                np.stack([example.policy_target for example in selected]).astype(np.float32)
            ).to(self.device)
            masks = torch.from_numpy(np.stack([example.legal_mask for example in selected])).to(
                self.device
            )
            values = torch.from_numpy(
                np.stack([example.value_target for example in selected]).astype(np.float32)
            ).to(self.device)
            weights = torch.from_numpy(
                np.asarray(
                    [getattr(example, "loss_weight", 1.0) for example in selected], dtype=np.float32
                )
            ).to(self.device)
            logits, predicted_values = self.network(states, hand_size)
            masked_logits = logits.masked_fill(~masks, -1e9)
            per_policy = -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)
            policy_losses.append((per_policy * weights).sum() / weights.sum().clamp_min(1.0))
            per_value = ((predicted_values - values) ** 2).mean(dim=1)
            value_losses.append((per_value * weights).sum() / weights.sum().clamp_min(1.0))
        if not policy_losses:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return zero, zero, zero
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        total = policy_loss + self.config.value_loss_weight * value_loss
        return total, policy_loss, value_loss

    def _fit(self) -> dict[str, float]:
        if not self.memory.examples:
            return {"total": math.nan, "policy": math.nan, "value": math.nan}
        self.network.train()
        totals = np.zeros(3, dtype=np.float64)
        for _ in range(self.config.train_steps_per_iteration):
            batch = self._sample_batch()
            self.optimizer.zero_grad(set_to_none=True)
            total, policy, value = self._loss_for_examples(batch)
            total.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in self.network.parameters() if parameter.requires_grad],
                5.0,
            )
            self.optimizer.step()
            totals += (float(total.detach()), float(policy.detach()), float(value.detach()))
        self.network.eval()
        totals /= self.config.train_steps_per_iteration
        return {"total": float(totals[0]), "policy": float(totals[1]), "value": float(totals[2])}

    @torch.inference_mode()
    def _validation_metrics(self, hand_size: int) -> tuple[float, float, np.ndarray]:
        examples = [
            example for example in self.validation_memory.examples if example.hand_size == hand_size
        ]
        if not examples:
            return math.inf, math.inf, np.empty((0, ACTION_DIM), dtype=np.float32)
        predictions: list[np.ndarray] = []
        losses: list[float] = []
        for start in range(0, len(examples), self.config.batch_size):
            batch = examples[start : start + self.config.batch_size]
            states = torch.from_numpy(
                np.stack([example.state for example in batch]).astype(np.float32)
            ).to(self.device)
            targets = torch.from_numpy(
                np.stack([example.policy_target for example in batch]).astype(np.float32)
            ).to(self.device)
            masks = torch.from_numpy(np.stack([example.legal_mask for example in batch])).to(
                self.device
            )
            logits, _ = self.network(states, hand_size)
            masked = logits.masked_fill(~masks, -1e9)
            probabilities = torch.softmax(masked, dim=1)
            loss = -(targets * torch.log_softmax(masked, dim=1)).sum(dim=1)
            predictions.append(probabilities.cpu().numpy())
            losses.extend(float(value) for value in loss.cpu())
        current = np.concatenate(predictions, axis=0)
        previous = self.previous_anchor_predictions.get(hand_size)
        policy_kl = math.inf
        if previous is not None and previous.shape == current.shape:
            p = np.maximum(previous, 1e-9)
            q = np.maximum(current, 1e-9)
            policy_kl = float(np.mean(np.sum(p * np.log(p / q), axis=1)))
        self.previous_anchor_predictions[hand_size] = current
        return float(np.mean(losses)), policy_kl, current

    def _decision_regret_metrics(
        self, hand_size: int, predictions: np.ndarray
    ) -> dict[str, object]:
        """Compare the policy with searched action values on fixed held-out roots.

        This is a local opportunity-cost diagnostic, not game-wide NashConv.
        Forced actions are reported but the stopping gate uses genuine choices.
        """

        examples = [
            example for example in self.validation_memory.examples if example.hand_size == hand_size
        ]
        grouped: dict[str, list[float]] = {}
        for example, probabilities in zip(examples, predictions):
            action_values = getattr(example, "action_value_target", None)
            if action_values is None:
                continue
            mask = np.asarray(example.legal_mask, dtype=np.bool_)
            if not bool(mask.any()):
                continue
            legal_probabilities = np.asarray(probabilities[mask], dtype=np.float64)
            legal_probabilities /= max(float(legal_probabilities.sum()), 1e-12)
            legal_values = np.asarray(action_values[mask], dtype=np.float64)
            regret = max(0.0, float(legal_values.max() - legal_probabilities @ legal_values))
            for category in getattr(example, "categories", ("all",)):
                grouped.setdefault(category, []).append(regret)

        def summary(values: list[float]) -> dict[str, float | int]:
            array = np.asarray(values, dtype=np.float64)
            return {
                "examples": len(values),
                "mean": float(array.mean()),
                "p90": float(np.quantile(array, 0.90)),
                "maximum": float(array.max()),
            }

        by_category = {name: summary(values) for name, values in sorted(grouped.items())}
        real_choices = by_category.get("real_choice")
        return {
            "mean_real_choice": (
                float(real_choices["mean"]) if real_choices is not None else math.inf
            ),
            "by_category": by_category,
        }

    def _cross_play(self, hand_size: int) -> dict[str, object]:
        candidate = self.policy()
        margins: list[float] = []
        threshold = self.config.satisfaction
        for game_index in range(threshold.validation_games_max):
            seat = game_index % self.players
            opponent = HeuristicPolicy() if game_index % 2 == 0 else RandomPolicy()
            policies = tuple(
                candidate if player == seat else opponent for player in range(self.players)
            )
            result = play_multiplayer_round(
                GameConfig(
                    players=self.players,
                    hand_size=hand_size,
                    starting_player=(game_index // self.players) % self.players,
                ),
                policies,
                seed=self.config.seed
                ^ (self.iteration * 1_000_003 + hand_size * 8191 + game_index),
            )
            best_other = min(error for player, error in enumerate(result.errors) if player != seat)
            margins.append(float(best_other - result.errors[seat]))
            completed_batch = (game_index + 1) % math.lcm(
                threshold.validation_batch_size, self.players * self.players
            ) == 0
            if completed_batch and adaptive_sample_complete(
                margins,
                minimum=threshold.validation_games,
                maximum=threshold.validation_games_max,
                target_half_width=threshold.validation_ci_half_width,
            ):
                break
        mean, interval, half_width = mean_ci95(margins)
        return {
            "games": len(margins),
            "mean_error_margin": mean,
            "error_margin_ci95": interval,
            "error_margin_ci95_half_width": half_width,
            "precision_target_reached": half_width <= threshold.validation_ci_half_width,
        }

    def _check_satisfaction(self) -> dict[str, object]:
        result: dict[str, object] = {}
        threshold = self.config.satisfaction
        for hand_size in self._active_rounds():
            cross_entropy, policy_kl, predictions = self._validation_metrics(hand_size)
            decision_regret = self._decision_regret_metrics(hand_size, predictions)
            cross_play = self._cross_play(hand_size)
            observed = [count for key, count in self.strata.items() if key[0] == hand_size]
            min_stratum = min(observed) if observed else 0
            focus_examples = self.category_counts[(hand_size, "difficult_cups")]
            checks = {
                "examples": self.examples_by_round[hand_size] >= threshold.min_examples_per_round,
                "strata": min_stratum >= threshold.min_examples_per_observed_stratum,
                "policy_stability": policy_kl <= threshold.max_policy_kl,
                "focus_coverage": focus_examples >= threshold.min_focus_examples,
                "decision_regret": decision_regret["mean_real_choice"]
                <= threshold.max_mean_decision_regret,
                "cross_play_precision": bool(cross_play["precision_target_reached"]),
                "cross_play": cross_play["error_margin_ci95"][0] >= threshold.min_error_margin_ci95,  # type: ignore[index]
            }
            if (
                self.config.repair_round == hand_size
                and threshold.min_mean_error_margin is not None
            ):
                checks["cross_play_mean"] = (
                    cross_play["mean_error_margin"] >= threshold.min_mean_error_margin
                )  # type: ignore[operator]
            stable_now = all(checks.values())
            self.scheduler.stable_checks[hand_size] = (
                self.scheduler.stable_checks[hand_size] + 1 if stable_now else 0
            )
            if not stable_now:
                # Maintenance examples exist precisely to detect shared-trunk
                # drift. Reopen a previously satisfied round as soon as one
                # complete validation gate regresses.
                self.scheduler.satisfied[hand_size] = False
            if self.scheduler.stable_checks[hand_size] >= threshold.stable_checks:
                self.scheduler.satisfied[hand_size] = True
            if (
                self.config.plateau_patience > 0
                and self.examples_by_round[hand_size] >= threshold.min_examples_per_round
                and bool(cross_play["precision_target_reached"])
                and math.isfinite(float(decision_regret["mean_real_choice"]))
            ):
                monitor_score = float(cross_play["mean_error_margin"]) - float(
                    decision_regret["mean_real_choice"]
                )
                best = self.best_monitor_score.get(hand_size)
                if best is None or monitor_score > best + self.config.plateau_min_delta:
                    self.best_monitor_score[hand_size] = monitor_score
                    self.plateau_checks[hand_size] = 0
                else:
                    self.plateau_checks[hand_size] += 1
                if self.plateau_checks[hand_size] >= self.config.plateau_patience:
                    self.plateau_reached.add(hand_size)
            if (
                self.config.repair_round == hand_size
                and self.config.repair_plateau_patience > 0
                and self.examples_by_round[hand_size] >= threshold.min_examples_per_round
                and min_stratum >= threshold.min_examples_per_observed_stratum
                and cross_entropy <= threshold.max_validation_cross_entropy
                and math.isfinite(policy_kl)
            ):
                mean_margin = float(cross_play["mean_error_margin"])
                if (
                    self.repair_best_mean_margin is None
                    or mean_margin
                    > self.repair_best_mean_margin + self.config.repair_plateau_min_delta
                ):
                    self.repair_best_mean_margin = mean_margin
                    self.repair_plateau_checks = 0
                else:
                    self.repair_plateau_checks += 1
                if self.repair_plateau_checks >= self.config.repair_plateau_patience:
                    self.repair_plateau_reached = True
            result[str(hand_size)] = {
                "examples": self.examples_by_round[hand_size],
                "minimum_observed_stratum": min_stratum,
                "validation_cross_entropy": cross_entropy,
                "policy_kl": policy_kl,
                "decision_regret": decision_regret,
                "focus_examples": focus_examples,
                "category_examples": {
                    category: count
                    for (round_size, category), count in sorted(self.category_counts.items())
                    if round_size == hand_size
                },
                **cross_play,
                "checks": checks,
                "stable_checks": self.scheduler.stable_checks[hand_size],
                "satisfied": self.scheduler.satisfied[hand_size],
                "monitor_score": self.best_monitor_score.get(hand_size),
                "plateau_checks": self.plateau_checks[hand_size],
                "plateau_reached": hand_size in self.plateau_reached,
                "repair_best_mean_margin": self.repair_best_mean_margin,
                "repair_plateau_checks": self.repair_plateau_checks,
                "repair_plateau_reached": self.repair_plateau_reached,
            }
        return result

    def train(self, *, checkpoint_path: Path | None = None) -> dict[str, object]:
        while self.iteration < self.config.iterations and not self._training_complete():
            self.iteration += 1
            iteration_started = time.perf_counter()
            collect_started = time.perf_counter()
            collected, allocation, teacher = self._collect()
            collect_elapsed = time.perf_counter() - collect_started
            self.timing_totals["collect_seconds"] += collect_elapsed
            fit_started = time.perf_counter()
            losses = self._fit()
            fit_elapsed = time.perf_counter() - fit_started
            self.timing_totals["fit_seconds"] += fit_elapsed
            satisfaction: dict[str, object] | None = None
            validation_elapsed = 0.0
            if self.iteration % self.config.satisfaction.validation_every == 0:
                validation_started = time.perf_counter()
                satisfaction = self._check_satisfaction()
                validation_elapsed = time.perf_counter() - validation_started
                self.timing_totals["validation_seconds"] += validation_elapsed
            if self.iteration % self.config.snapshot_every == 0:
                self.snapshots.append(self._cpu_state())
                self.snapshots = self.snapshots[-self.config.max_snapshots :]
            report = {
                "iteration": self.iteration,
                "elapsed_seconds": time.perf_counter() - iteration_started,
                "allocation_games": {str(key): value for key, value in sorted(allocation.items())},
                "collected_examples": collected,
                "memory_examples": len(self.memory.examples),
                "validation_examples": len(self.validation_memory.examples),
                "examples_by_round": {str(key): self.examples_by_round[key] for key in range(1, 6)},
                "unique_information_states_observed": {
                    str(key): len(self.unique_state_digests[key]) for key in range(1, 6)
                },
                "loss": losses,
                "teacher": teacher,
                "satisfaction": satisfaction,
                "timing_seconds": {
                    "collect": collect_elapsed,
                    "fit": fit_elapsed,
                    "validation": validation_elapsed,
                },
            }
            self.reports.append(report)
            self.reports = self.reports[-self.config.report_history :]
            print({"multiplayer_neural_iteration": report}, flush=True)
            now = time.perf_counter()
            if (
                self.config.max_runtime_minutes is not None
                and (now - self.started) / 60.0 >= self.config.max_runtime_minutes
            ):
                self.request_stop()
            if checkpoint_path is not None and (
                (now - self.last_checkpoint) / 60.0 >= self.config.checkpoint_minutes
                or self.stop_requested
            ):
                self.save_checkpoint(checkpoint_path)
                self.last_checkpoint = now
            if self.stop_requested:
                break
        return self.report()

    def report(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started
        rss_bytes = 0
        if resource is not None:
            rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_bytes = rss_kib * (
                1024 if os.name != "posix" or os.uname().sysname != "Darwin" else 1024 * 1024
            )
        decisions = self.teacher_totals["decisions"]
        return {
            "kind": "neural_multiplayer_expert_iteration_v1",
            "players": self.players,
            "encoding": encoding_version(self.players),
            "config": asdict(self.config),
            "search": self.search.to_dict(),
            "search_semantics": self.search_semantics,
            "iteration": self.iteration,
            "status": (
                "runtime_limit"
                if self.stop_requested and self.stop_signal is None
                else "stopped"
                if self.stop_requested
                else "plateau"
                if self.plateau_reached and self._training_complete()
                else "plateau"
                if self.repair_plateau_reached
                and not self.scheduler.satisfied.get(self.config.repair_round or 0, False)
                else "satisfied"
                if self._training_complete()
                else "iteration_limit"
            ),
            "requires_continuation": self.stop_requested or not self._training_complete(),
            "satisfied_rounds": [
                size for size in self._active_rounds() if self.scheduler.satisfied[size]
            ],
            "repair_round": self.config.repair_round,
            "repair_plateau": {
                "best_mean_margin": self.repair_best_mean_margin,
                "checks": self.repair_plateau_checks,
                "patience": self.config.repair_plateau_patience,
                "min_delta": self.config.repair_plateau_min_delta,
                "reached": self.repair_plateau_reached,
            },
            "round_plateau": {
                "best_monitor_score": self.best_monitor_score,
                "checks": dict(self.plateau_checks),
                "reached": sorted(self.plateau_reached),
                "patience": self.config.plateau_patience,
                "min_delta": self.config.plateau_min_delta,
            },
            "examples_by_round": {str(key): self.examples_by_round[key] for key in range(1, 6)},
            "unique_information_states_observed": {
                str(key): len(self.unique_state_digests[key]) for key in range(1, 6)
            },
            "unique_counter_saturated_rounds": [
                key
                for key in range(1, 6)
                if len(self.unique_state_digests[key]) >= self.config.unique_counter_limit
            ],
            "teacher": {
                **{
                    key: value
                    for key, value in self.teacher_totals.items()
                    if key != "effective_particles_sum"
                },
                "mean_simulations_per_decision": self.teacher_totals["simulations"] / decisions
                if decisions
                else 0.0,
                "mean_effective_particles": self.teacher_totals["effective_particles_sum"]
                / decisions
                if decisions
                else 0.0,
                "hard_position_rate": self.teacher_totals["hard_positions"] / decisions
                if decisions
                else 0.0,
            },
            "training_statistics": {
                "elapsed_seconds": elapsed,
                "max_rss_bytes": rss_bytes,
                "max_rss_gib": rss_bytes / (1024**3),
                "max_worker_rss_bytes": self.max_worker_rss_bytes,
                "estimated_all_processes_peak_bytes": (
                    rss_bytes + self.config.cpu_workers * self.max_worker_rss_bytes
                ),
                "estimated_all_processes_peak_gib": (
                    rss_bytes + self.config.cpu_workers * self.max_worker_rss_bytes
                )
                / (1024**3),
                "model_parameters": self.network.parameter_count,
                "model_fp32_bytes": self.network.parameter_count * 4,
                "stored_state_rows": 0,
                "timing_seconds": dict(self.timing_totals),
            },
            "iterations": self.reports,
            "warning": "ISMCTS Max-N is a strong practical teacher, not a multiplayer equilibrium certificate",
        }

    def save_checkpoint(self, path: Path) -> None:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "players": self.players,
            "search": self.search.to_dict(),
            "search_semantics": self.search_semantics,
            "config": asdict(self.config),
            "network": self._cpu_state(),
            # Keep resumable checkpoints portable between CUDA HPC workers
            # and CPU-only laptops.  The network is already copied to CPU;
            # optimizer tensors need the same treatment.
            "optimizer": _tensors_on_cpu(self.optimizer.state_dict()),
            "memory": self.memory,
            "validation_memory": self.validation_memory,
            "strata": self.strata,
            "category_counts": self.category_counts,
            "unique_state_digests": self.unique_state_digests,
            "examples_by_round": self.examples_by_round,
            "scheduler": self.scheduler,
            "snapshots": self.snapshots,
            "previous_anchor_predictions": self.previous_anchor_predictions,
            "iteration": self.iteration,
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "reports": self.reports,
            "teacher_totals": self.teacher_totals,
            "timing_totals": self.timing_totals,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "repair_prepared_rounds": sorted(self.repair_prepared_rounds),
            "repair_best_mean_margin": self.repair_best_mean_margin,
            "repair_plateau_checks": self.repair_plateau_checks,
            "repair_plateau_reached": self.repair_plateau_reached,
            "best_monitor_score": self.best_monitor_score,
            "plateau_checks": self.plateau_checks,
            "plateau_reached": sorted(self.plateau_reached),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        with gzip.open(temporary, "wb", compresslevel=1) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def restore(
        cls,
        path: Path,
        *,
        config: NeuralExpertIterationConfig | None = None,
        device: str | None = None,
    ) -> NeuralMultiplayerExpertTrainer:
        # Older production checkpoints were written with ``pickle.dump`` while
        # Adam's state tensors lived on CUDA.  PyTorch's pickle reducer then
        # calls ``torch.storage._load_from_bytes`` without a map_location,
        # which fails on a CPU-only machine.  Temporarily route that reducer
        # through torch.load(map_location="cpu") so both old and new files
        # remain loadable offline.
        original_loader = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda data: torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=False
        )
        try:
            with gzip.open(path, "rb") as handle:
                payload = pickle.load(handle)
        finally:
            torch.storage._load_from_bytes = original_loader
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported neural multiplayer checkpoint format")
        stored_config = _config_from_dict(dict(payload["config"]))
        active_config = config or stored_config
        if device is not None and active_config.device != device:
            active_config = replace(active_config, device=device)
        if active_config.architecture != stored_config.architecture:
            raise ValueError("cannot change neural architecture when resuming")
        search = SearchConfig.from_dict(dict(payload["search"]))
        trainer = cls(int(payload["players"]), search, active_config)
        trainer.search_semantics = str(
            payload.get("search_semantics", "root-observer-information-sets-v0")
        )
        trainer.network.load_state_dict(payload["network"])
        trainer.memory = payload["memory"]
        trainer.memory.capacity = active_config.memory_examples
        trainer.memory.rng = trainer.rng
        trainer.validation_memory = payload["validation_memory"]
        trainer.validation_memory.capacity = active_config.validation_examples
        trainer.strata = Counter(payload["strata"])
        trainer.category_counts = Counter(payload.get("category_counts", {}))
        trainer.unique_state_digests = {
            hand_size: set(payload.get("unique_state_digests", {}).get(hand_size, ()))
            for hand_size in range(1, 6)
        }
        trainer.examples_by_round = Counter(payload["examples_by_round"])
        trainer.scheduler = payload["scheduler"]
        trainer.snapshots = list(payload["snapshots"])[-active_config.max_snapshots :]
        trainer.previous_anchor_predictions = dict(payload["previous_anchor_predictions"])
        trainer.iteration = int(payload["iteration"])
        trainer.rng.setstate(payload["rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        trainer.reports = list(payload.get("reports", ()))
        trainer.teacher_totals = Counter(payload.get("teacher_totals", {}))
        trainer.timing_totals = Counter(payload.get("timing_totals", {}))
        trainer.max_worker_rss_bytes = int(payload.get("max_worker_rss_bytes", 0))
        trainer.repair_prepared_rounds = set(
            int(value) for value in payload.get("repair_prepared_rounds", ())
        )
        trainer.repair_best_mean_margin = payload.get("repair_best_mean_margin")
        if trainer.repair_best_mean_margin is not None:
            trainer.repair_best_mean_margin = float(trainer.repair_best_mean_margin)
        trainer.repair_plateau_checks = int(payload.get("repair_plateau_checks", 0))
        trainer.repair_plateau_reached = bool(payload.get("repair_plateau_reached", False))
        trainer.best_monitor_score = {
            int(key): float(value) for key, value in payload.get("best_monitor_score", {}).items()
        }
        trainer.plateau_checks = Counter(payload.get("plateau_checks", {}))
        trainer.plateau_reached = set(int(value) for value in payload.get("plateau_reached", ()))
        if active_config.repair_round is not None:
            if active_config.repair_continue:
                trainer.continue_round_repair(active_config.repair_round)
            elif active_config.repair_round in trainer.repair_prepared_rounds:
                trainer._configure_repair_optimizer(active_config.repair_round)
                trainer.optimizer.load_state_dict(payload["optimizer"])
            else:
                trainer.prepare_round_repair(active_config.repair_round)
        else:
            if trainer.repair_prepared_rounds:
                raise ValueError("a repair checkpoint must be resumed with its repair_round")
            trainer.optimizer.load_state_dict(payload["optimizer"])
        # Allow the next slice to change optimizer hyperparameters explicitly.
        for group in trainer.optimizer.param_groups:
            group["lr"] = active_config.learning_rate
            group["weight_decay"] = active_config.weight_decay
        return trainer


def _config_from_dict(values: dict[str, object]) -> NeuralExpertIterationConfig:
    raw = dict(values)
    raw["round_weights"] = tuple(raw.get("round_weights", (0, 1, 2, 2, 2)))
    raw["architecture"] = NeuralBlueprintArchitecture.from_dict(dict(raw.get("architecture", {})))
    raw["teacher"] = AdaptiveTeacherConfig(**dict(raw.get("teacher", {})))
    raw["satisfaction"] = SatisfactionConfig(**dict(raw.get("satisfaction", {})))
    return NeuralExpertIterationConfig(**raw)


def install_stop_handlers(trainer: NeuralMultiplayerExpertTrainer) -> None:
    def stop(signum: int, frame: object) -> None:
        del frame
        trainer.request_stop(signum)

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
