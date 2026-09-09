from __future__ import annotations

"""Distill a large tabular policy into a bounded linear blueprint.

The source table is used only as a teacher.  We sample reachable trajectories,
read the teacher distribution at every visited information state, and update a
fixed-size :class:`LinearBlueprintPolicy`.  This deliberately avoids building
an intermediate ``dict`` of all table rows, which is the part that makes R2/R3
uncomfortable on a normal desktop.
"""

import gzip
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.policies.linear import LinearBlueprintPolicy, LinearFeatureConfig


@dataclass(frozen=True, slots=True)
class TabularDistillationConfig:
    """Bound the amount of game sampling and the size of the exported model."""

    games: int = 100_000
    epochs: int = 2
    learning_rate: float = 0.035
    l2: float = 1e-5
    strength_buckets: int = 12
    tile_buckets: int = 16_384
    validation_games: int = 5_000
    seed: int = 20260824

    def __post_init__(self) -> None:
        if self.games <= 0 or self.epochs <= 0 or self.validation_games <= 0:
            raise ValueError("games, epochs and validation_games must be positive")
        if self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("learning_rate must be positive and l2 non-negative")


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (logits - float(np.max(logits))) / temperature
    values = np.exp(np.clip(shifted, -60.0, 60.0))
    return values / float(np.sum(values))


class TabularBlueprintDistiller:
    """Fit a compact teacher imitation policy from reachable table states."""

    def __init__(
        self,
        game: GameConfig,
        teacher: object,
        config: TabularDistillationConfig,
        *,
        policy: LinearBlueprintPolicy | None = None,
    ) -> None:
        self.game = game
        self.teacher = teacher
        self.config = config
        self.rng = random.Random(config.seed)
        self.policy = policy or LinearBlueprintPolicy(
            LinearFeatureConfig(
                strength_buckets=config.strength_buckets,
                temperature=0.35,
                tile_buckets=config.tile_buckets,
            )
        )
        self.examples = 0
        self.updates = 0
        self.teacher_exact = 0
        self.teacher_fallback = 0
        self.loss_sum = 0.0
        self.started = time.perf_counter()

    def _teacher_entry(self, info: Any) -> tuple[float, ...] | None:
        entries = getattr(self.teacher, "entries", None)
        if entries is None:
            raise TypeError("distillation teacher must expose tabular entries")
        entry = entries.get(info.key())
        if entry is None:
            self.teacher_fallback += 1
            return None
        actions, probabilities = entry
        if tuple(actions) != tuple(info.legal_actions):
            raise ValueError("teacher action order differs from the current game")
        self.teacher_exact += 1
        return tuple(float(value) for value in probabilities)

    def _update(self, info: Any, target: tuple[float, ...], learning_rate: float) -> float:
        state = self.policy.state_features(info)
        actions = self.policy.action_features(info)
        tiles = self.policy.tile_indices(info)
        logits = self.policy.linear_logits_from_features(state, actions)
        if tiles.shape[1]:
            logits += self.policy.tile_logits_from_indices(tiles)
        logits += self.policy.base_logits(info)
        probabilities = _softmax(logits, self.policy.config.temperature)
        target_array = np.asarray(target, dtype=np.float64)
        loss = float(-np.sum(target_array * np.log(np.maximum(probabilities, 1e-12))))
        error = probabilities - target_array
        action_gradient = error @ actions
        base_count = self.policy.base_parameter_count
        gradient = np.concatenate(
            (
                np.outer(state, action_gradient).reshape(-1),
                action_gradient,
                np.asarray((float(np.sum(error)),), dtype=np.float64),
            )
        )
        self.policy.weights[:base_count] -= learning_rate * (
            gradient + self.config.l2 * self.policy.weights[:base_count]
        )
        if tiles.shape[1]:
            addresses, inverse = np.unique(tiles.ravel(), return_inverse=True)
            tile_gradient = np.zeros(len(addresses), dtype=np.float64)
            np.add.at(tile_gradient, inverse, np.repeat(error, tiles.shape[1]))
            tile_weights = self.policy.weights[base_count:]
            tile_weights[addresses] -= learning_rate * (
                tile_gradient + self.config.l2 * tile_weights[addresses]
            )
        self.examples += 1
        self.updates += 1
        self.loss_sum += loss
        return loss

    def _sample_games(
        self, count: int, *, training: bool, seed_offset: int = 0
    ) -> dict[str, float | int]:
        rng = random.Random(self.config.seed + seed_offset)
        exact_before = self.teacher_exact
        fallback_before = self.teacher_fallback
        examples_before = self.examples
        losses_before = self.loss_sum
        game_count = 0
        for _ in range(count):
            starter = rng.randrange(self.game.players)
            game = GameConfig(
                players=self.game.players,
                hand_size=self.game.hand_size,
                starting_player=starter,
                payoff=self.game.payoff,
            )
            state = RoundState(game)
            state.apply_chance_fast(state.sample_chance(rng))
            while not state.is_terminal:
                info = state.information_state()
                target = self._teacher_entry(info)
                if target is not None and training:
                    # A slowly decreasing rate makes later epochs corrections
                    # rather than wholesale rewrites of the learned policy.
                    rate = self.config.learning_rate / math.sqrt(1.0 + self.updates / 100_000.0)
                    self._update(info, target, rate)
                if target is None:
                    weights = (1.0,) * len(info.legal_actions)
                else:
                    weights = target
                action = rng.choices(info.legal_actions, weights=weights, k=1)[0]
                state.apply_action_fast(action)
            game_count += 1
        return {
            "games": game_count,
            "examples": self.examples - examples_before,
            "teacher_exact": self.teacher_exact - exact_before,
            "teacher_fallback": self.teacher_fallback - fallback_before,
            "cross_entropy": (
                (self.loss_sum - losses_before) / max(1, self.examples - examples_before)
                if training
                else 0.0
            ),
        }

    def _validation(self) -> dict[str, float | int]:
        rng = random.Random(self.config.seed ^ 0x5F3759DF)
        exact = fallback = states = top1 = 0
        kl_sum = 0.0
        for _ in range(self.config.validation_games):
            starter = rng.randrange(self.game.players)
            game = GameConfig(
                players=self.game.players,
                hand_size=self.game.hand_size,
                starting_player=starter,
                payoff=self.game.payoff,
            )
            state = RoundState(game)
            state.apply_chance_fast(state.sample_chance(rng))
            while not state.is_terminal:
                info = state.information_state()
                target = self._teacher_entry(info)
                model = np.asarray(self.policy.probabilities(info), dtype=np.float64)
                if target is None:
                    fallback += 1
                    weights = (1.0,) * len(info.legal_actions)
                else:
                    exact += 1
                    teacher = np.asarray(target, dtype=np.float64)
                    kl_sum += float(
                        np.sum(
                            teacher * np.log(np.maximum(teacher, 1e-12) / np.maximum(model, 1e-12))
                        )
                    )
                    if int(np.argmax(teacher)) == int(np.argmax(model)):
                        top1 += 1
                    weights = target
                states += 1
                action = rng.choices(info.legal_actions, weights=weights, k=1)[0]
                state.apply_action_fast(action)
        return {
            "games": self.config.validation_games,
            "states": states,
            "teacher_exact": exact,
            "teacher_fallback": fallback,
            "exact_rate": exact / max(1, exact + fallback),
            "top1_agreement": top1 / max(1, exact),
            "mean_kl_teacher_to_blueprint": kl_sum / max(1, exact),
        }

    def train(self) -> dict[str, object]:
        epochs = []
        for epoch in range(1, self.config.epochs + 1):
            stats = self._sample_games(
                self.config.games, training=True, seed_offset=epoch * 1_000_003
            )
            stats["epoch"] = epoch
            stats["updates_total"] = self.updates
            epochs.append(stats)
        validation = self._validation()
        elapsed = time.perf_counter() - self.started
        return {
            "kind": "tabular-reachable-state-distillation-v1",
            "game": self.game.to_dict(),
            "config": asdict(self.config),
            "policy": self.policy.stats(),
            "epochs": epochs,
            "validation": validation,
            "training_statistics": {
                "elapsed_seconds": elapsed,
                "examples": self.examples,
                "updates": self.updates,
                "teacher_exact": self.teacher_exact,
                "teacher_fallback": self.teacher_fallback,
                "teacher_exact_rate": self.teacher_exact
                / max(1, self.teacher_exact + self.teacher_fallback),
                "source_note": "only reachable sampled states are retained; the full table is never copied",
            },
        }


def load_tabular_teacher(path: Path) -> tuple[GameConfig, object]:
    """Load a sampled policy, including the legacy R3 table as a teacher.

    Legacy MCCFR+ shards are intentionally rejected by the normal deployment
    loader because their strategy-sum semantics are not safe to deploy as-is.
    Distillation is different: it only queries the old table as a temporary
    teacher and writes a new fixed-size blueprint.  Keep the compatibility
    path local to this command so the legacy table can never silently become a
    game-time policy.
    """

    path = Path(path)
    if path.name == "training.pt":
        from presine.learning.checkpoint_deployment.sampled_checkpoint import (
            load_sampled_training_policy,
        )

        return load_sampled_training_policy(path)
    from presine.learning.checkpoint_deployment.sampled_checkpoint import load_sampled_policy

    try:
        return load_sampled_policy(path)
    except ValueError as error:
        # The R3 artifact is a current sharded manifest whose two shards use
        # the pre-v2 ``mccfr-plus-average-policy`` payload.  Only fall back
        # when the manifest explicitly advertises that sharded format; other
        # loader errors should retain their original diagnostic.
        try:
            manifest = _read_pickle_relaxed(path)
        except Exception:
            raise error
        if (
            manifest.get("format") != 3
            or manifest.get("kind") != "external-sampling-tabular-sharded-average-policy"
        ):
            raise error
        sources: list[object] = []
        for relative in manifest.get("shards", []):
            shard = _read_pickle_relaxed(path.parent / str(relative))
            if "entries" in shard:
                sources.append(shard["entries"])
            elif "entries_compact" in shard:
                from presine.learning.checkpoint_deployment.compact_store import (
                    CompactPolicyEntries,
                )

                sources.append(CompactPolicyEntries(shard["entries_compact"]))
            else:
                raise error
        if not sources:
            raise error
        return GameConfig.from_dict(manifest["game"]), SimpleNamespace(
            entries=_LegacyTeacherEntries(sources)
        )


class _LegacyTeacherEntries:
    """Read-only lookup over legacy starting-player shards.

    The distiller only needs ``get`` for sampled reachable states.  Avoid
    merging both huge shard dictionaries into a third giant dictionary.
    """

    def __init__(self, sources: list[object]) -> None:
        self._sources = tuple(sources)

    def get(self, key: object, default: object = None) -> object:
        for source in self._sources:
            getter = getattr(source, "get", None)
            if getter is None:
                continue
            value = getter(key, None)
            if value is not None:
                return value
        return default


def _read_pickle_relaxed(path: Path) -> dict[str, Any]:
    """Read a gzip or plain pickle without applying deploy-format guards."""

    with path.open("rb") as raw:
        magic = raw.read(2)
        raw.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=raw, mode="rb") as handle:
                payload = pickle.load(handle)
        else:
            payload = pickle.load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid legacy teacher payload: {path}")
    return payload
