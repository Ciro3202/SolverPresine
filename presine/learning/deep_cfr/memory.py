from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

from .encoding import ACTION_DIM, STATE_DIM

VALIDATION_MODULUS = 10
DISTINCT_TRACKING_LIMIT = 50_000


@dataclass(frozen=True, slots=True)
class MemoryBatch:
    states: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    weights: np.ndarray


class ReservoirBuffer:
    """Bounded uniform reservoir with compact float16 state storage."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        depth_index: int = 10,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.depth_index = depth_index
        self.seen = 0
        self._rng = random.Random(seed)
        self._states: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._weights: list[float] = []
        self._validation: list[bool] = []
        self._distinct_hashes: set[int] = set()
        self._distinct_groups: dict[tuple[int, int], set[int]] = {}
        self._distinct_saturated = False
        self._partition_cache: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self._states)

    def add(
        self,
        state: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        weight: float,
    ) -> None:
        if state.shape != (self.state_dim,):
            raise ValueError("state has the wrong shape")
        if target.shape != (self.action_dim,) or mask.shape != (self.action_dim,):
            raise ValueError("target or mask has the wrong shape")
        if weight <= 0:
            raise ValueError("sample weight must be positive")
        compact_state = np.asarray(state, dtype=np.float16)
        state_hash = self._state_hash(compact_state)
        is_validation = state_hash % VALIDATION_MODULUS == 0
        self._track_distinct(state_hash, compact_state)
        self.seen += 1
        if len(self) < self.capacity:
            index = len(self)
            self._states.append(compact_state)
            self._targets.append(np.asarray(target, dtype=np.float16))
            self._masks.append(np.asarray(mask, dtype=np.bool_))
            self._weights.append(float(weight))
            self._validation.append(is_validation)
        else:
            index = self._rng.randrange(self.seen)
            if index >= self.capacity:
                return
            self._states[index] = compact_state
            self._targets[index] = np.asarray(target, dtype=np.float16)
            self._masks[index] = np.asarray(mask, dtype=np.bool_)
            self._weights[index] = float(weight)
            self._validation[index] = is_validation
        self._partition_cache.clear()

    def sample(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        partition: str = "all",
    ) -> MemoryBatch:
        if not self:
            raise ValueError("cannot sample an empty reservoir")
        available = self._indices(partition)
        if not available:
            raise ValueError(f"reservoir partition is empty: {partition}")
        indices = [available[rng.randrange(len(available))] for _ in range(batch_size)]
        return self._batch(indices)

    def validation_batch(self, max_samples: int) -> MemoryBatch | None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        indices = self._indices("validation")
        if not indices:
            return None
        if len(indices) > max_samples:
            positions = np.linspace(0, len(indices) - 1, max_samples, dtype=np.int64)
            indices = [indices[int(position)] for position in positions]
        return self._batch(indices)

    def partition_size(self, partition: str) -> int:
        return len(self._indices(partition))

    def coverage(self) -> dict[str, object]:
        return {
            "seen": self.seen,
            "retained": len(self),
            "train_retained": self.partition_size("train"),
            "validation_retained": self.partition_size("validation"),
            "distinct_information_states_lower_bound": len(self._distinct_hashes),
            "distinct_tracking_limit": DISTINCT_TRACKING_LIMIT,
            "distinct_tracking_saturated": self._distinct_saturated,
            "by_phase_depth_lower_bound": {
                f"phase_{phase}_depth_{depth}": len(hashes)
                for (phase, depth), hashes in sorted(self._distinct_groups.items())
            },
        }

    def _batch(self, indices: list[int]) -> MemoryBatch:
        return MemoryBatch(
            states=np.stack([self._states[i] for i in indices]).astype(np.float32),
            targets=np.stack([self._targets[i] for i in indices]).astype(np.float32),
            masks=np.stack([self._masks[i] for i in indices]),
            weights=np.asarray([self._weights[i] for i in indices], dtype=np.float32),
        )

    def _indices(self, partition: str) -> list[int]:
        if partition not in ("all", "train", "validation"):
            raise ValueError(f"unknown reservoir partition: {partition}")
        cached = self._partition_cache.get(partition)
        if cached is not None:
            return cached
        if partition == "all":
            indices = list(range(len(self)))
        elif partition == "train":
            indices = [index for index, held_out in enumerate(self._validation) if not held_out]
        else:
            indices = [index for index, held_out in enumerate(self._validation) if held_out]
        self._partition_cache[partition] = indices
        return indices

    @staticmethod
    def _state_hash(state: np.ndarray) -> int:
        compact = np.asarray(state, dtype=np.float16)
        digest = hashlib.blake2b(compact.tobytes(), digest_size=8, person=b"presine").digest()
        return int.from_bytes(digest, "little")

    def _track_distinct(self, state_hash: int, state: np.ndarray) -> None:
        if state_hash in self._distinct_hashes:
            return
        if len(self._distinct_hashes) >= DISTINCT_TRACKING_LIMIT:
            self._distinct_saturated = True
            return
        self._distinct_hashes.add(state_hash)
        encoded = np.asarray(state)
        phase = int(np.argmax(encoded[:3]))
        depth = int(round(float(encoded[self.depth_index]) * 5.0))
        self._distinct_groups.setdefault((phase, depth), set()).add(state_hash)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "depth_index": self.depth_index,
            "seen": self.seen,
            "rng_state": self._rng.getstate(),
            "states": np.stack(self._states)
            if self._states
            else np.empty((0, self.state_dim), np.float16),
            "targets": np.stack(self._targets)
            if self._targets
            else np.empty((0, self.action_dim), np.float16),
            "masks": np.stack(self._masks)
            if self._masks
            else np.empty((0, self.action_dim), np.bool_),
            "weights": np.asarray(self._weights, dtype=np.float32),
            "validation": np.asarray(self._validation, dtype=np.bool_),
            "distinct_hashes": np.asarray(sorted(self._distinct_hashes), dtype=np.uint64),
            "distinct_groups": {
                key: np.asarray(sorted(hashes), dtype=np.uint64)
                for key, hashes in self._distinct_groups.items()
            },
            "distinct_saturated": self._distinct_saturated,
        }

    @classmethod
    def from_state_dict(cls, values: dict[str, object]) -> ReservoirBuffer:
        states = np.asarray(values["states"])
        targets = np.asarray(values["targets"])
        buffer = cls(
            int(values["capacity"]),
            seed=0,
            state_dim=int(
                values.get(
                    "state_dim",
                    states.shape[1] if states.ndim == 2 and states.shape[1] else STATE_DIM,
                )
            ),
            action_dim=int(
                values.get(
                    "action_dim",
                    targets.shape[1] if targets.ndim == 2 and targets.shape[1] else ACTION_DIM,
                )
            ),
            depth_index=int(values.get("depth_index", 10)),
        )
        buffer.seen = int(values["seen"])
        buffer._rng.setstate(values["rng_state"])  # type: ignore[arg-type]
        buffer._states = [row.copy() for row in states]
        buffer._targets = [row.copy() for row in targets]
        buffer._masks = [row.copy() for row in np.asarray(values["masks"])]
        buffer._weights = [float(value) for value in np.asarray(values["weights"])]
        validation = values.get("validation")
        if validation is None:
            buffer._validation = [
                buffer._state_hash(state) % VALIDATION_MODULUS == 0 for state in buffer._states
            ]
        else:
            buffer._validation = [bool(value) for value in np.asarray(validation)]
        distinct = values.get("distinct_hashes")
        if distinct is None:
            for state in buffer._states:
                state_hash = buffer._state_hash(state)
                buffer._track_distinct(state_hash, state)
        else:
            buffer._distinct_hashes = {
                int(value) for value in np.asarray(distinct, dtype=np.uint64)
            }
            groups = values.get("distinct_groups", {})
            buffer._distinct_groups = {
                tuple(key): {int(value) for value in np.asarray(hashes, dtype=np.uint64)}
                for key, hashes in groups.items()  # type: ignore[union-attr]
            }
            buffer._distinct_saturated = bool(values.get("distinct_saturated", False))
        return buffer
