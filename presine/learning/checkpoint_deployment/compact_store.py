from __future__ import annotations

import json
import os
import pickle
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from presine.policies.tabular import InformationKey, PolicyEntry

from ..key_codec import decode_information_key, encode_information_key

if TYPE_CHECKING:
    from ..tabular_mccfr.mccfr_trainer import MCCFRNode

# Heads-up round-five has at most six legal actions (the bid phase).  Fixed
# width numeric slabs avoid one Python list and one Python float object per
# action and are the main memory saving over the legacy node dictionary.
MAX_ACTIONS = 6
CHUNK_SIZE = 1_048_576
COMPACT_STORE_FORMAT = "packed-mccfr-v2"
LEGACY_COMPACT_STORE_FORMAT = "packed-mccfr-v1"


class CompactMCCFRStore:
    """Sparse MCCFR table backed by a byte-key index and numeric slabs.

    The index keeps only one compact byte serialization per information state;
    regrets, strategy sums, actions, and DCFR timestamps live in chunked NumPy
    arrays.  Chunks grow independently, so adding states does not copy the
    whole table or create a second multi-hundred-GB allocation.
    """

    CHUNK_SIZE = CHUNK_SIZE

    def __init__(self) -> None:
        self._index: dict[bytes, int] = {}
        self._keys: dict[bytes, int] = self._index
        self._regrets: list[np.ndarray] = []
        self._strategy: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._lengths: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._count = 0
        self._legacy_pickle_keys = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CompactMCCFRStore:
        if payload.get("format") not in {
            COMPACT_STORE_FORMAT,
            LEGACY_COMPACT_STORE_FORMAT,
        }:
            raise ValueError("unsupported compact MCCFR store format")
        store = cls()
        store._legacy_pickle_keys = payload.get("format") == LEGACY_COMPACT_STORE_FORMAT
        # Reuse the unpickled index object; copying it would briefly recreate
        # the very multi-GB allocation this store is designed to avoid.
        store._index = payload["index"]
        store._keys = store._index
        store._regrets = [np.asarray(item, dtype=np.float64) for item in payload["regrets"]]
        store._strategy = [np.asarray(item, dtype=np.float64) for item in payload["strategy"]]
        store._actions = [np.asarray(item, dtype=np.int16) for item in payload["actions"]]
        store._lengths = [np.asarray(item, dtype=np.uint8) for item in payload["lengths"]]
        store._timestamps = [np.asarray(item, dtype=np.int32) for item in payload["timestamps"]]
        store._count = int(payload["count"])
        return store

    def _encode_key(self, key: InformationKey) -> bytes:
        return (
            pickle.dumps(key, protocol=5)
            if self._legacy_pickle_keys
            else encode_information_key(key)
        )

    def _decode_key(self, encoded: bytes) -> InformationKey:
        return (
            pickle.loads(encoded) if self._legacy_pickle_keys else decode_information_key(encoded)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": (
                LEGACY_COMPACT_STORE_FORMAT if self._legacy_pickle_keys else COMPACT_STORE_FORMAT
            ),
            "index": self._index,
            "regrets": self._regrets,
            "strategy": self._strategy,
            "actions": self._actions,
            "lengths": self._lengths,
            "timestamps": self._timestamps,
            "count": self._count,
        }

    def _ensure_chunk(self) -> None:
        if self._count % CHUNK_SIZE:
            return
        shape = (CHUNK_SIZE, MAX_ACTIONS)
        self._regrets.append(np.zeros(shape, dtype=np.float64))
        self._strategy.append(np.zeros(shape, dtype=np.float64))
        self._actions.append(np.zeros(shape, dtype=np.int16))
        self._lengths.append(np.zeros(CHUNK_SIZE, dtype=np.uint8))
        self._timestamps.append(np.zeros(CHUNK_SIZE, dtype=np.int32))

    @staticmethod
    def _position(index: int) -> tuple[int, int]:
        return divmod(index, CHUNK_SIZE)

    def lookup(
        self, key: InformationKey, actions: tuple[int, ...], iteration: int
    ) -> tuple[int, bool]:
        if not actions or len(actions) > MAX_ACTIONS:
            raise ValueError("unsupported legal-action width for compact MCCFR store")
        encoded = self._encode_key(key)
        index = self._index.get(encoded)
        if index is not None:
            stored_actions = self.actions(index)
            if stored_actions != actions:
                raise ValueError("legal action order changed for an information state")
            return index, False
        self._ensure_chunk()
        index = self._count
        chunk, row = self._position(index)
        self._index[encoded] = index
        self._actions[chunk][row, : len(actions)] = actions
        self._lengths[chunk][row] = len(actions)
        self._timestamps[chunk][row] = iteration
        self._count += 1
        return index, True

    def __len__(self) -> int:
        return self._count

    def actions(self, index: int) -> tuple[int, ...]:
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        return tuple(int(value) for value in self._actions[chunk][row, :length])

    def regrets(self, index: int) -> tuple[float, ...]:
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        return tuple(float(value) for value in self._regrets[chunk][row, :length])

    def strategy_sum(self, index: int) -> tuple[float, ...]:
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        return tuple(float(value) for value in self._strategy[chunk][row, :length])

    def timestamp(self, index: int) -> int:
        chunk, row = self._position(index)
        return int(self._timestamps[chunk][row])

    def set_timestamp(self, index: int, value: int) -> None:
        chunk, row = self._position(index)
        self._timestamps[chunk][row] = value

    def set_node(
        self,
        index: int,
        actions: tuple[int, ...],
        regrets: tuple[float, ...] | list[float],
        strategy_sum: tuple[float, ...] | list[float],
        timestamp: int,
    ) -> None:
        if len(actions) > MAX_ACTIONS:
            raise ValueError("unsupported legal-action width for compact MCCFR store")
        chunk, row = self._position(index)
        length = len(actions)
        self._actions[chunk][row, :] = 0
        self._actions[chunk][row, :length] = actions
        self._lengths[chunk][row] = length
        self._regrets[chunk][row, :] = 0.0
        self._strategy[chunk][row, :] = 0.0
        self._regrets[chunk][row, :length] = regrets
        self._strategy[chunk][row, :length] = strategy_sum
        self._timestamps[chunk][row] = timestamp

    def scale_regrets(self, index: int, positive: float, negative: float) -> None:
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        values = self._regrets[chunk][row, :length]
        for action_index, value in enumerate(values):
            values[action_index] = value * (positive if value >= 0.0 else negative)

    def add_strategy(self, index: int, probabilities: tuple[float, ...], weight: float) -> None:
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        self._strategy[chunk][row, :length] += weight * np.asarray(probabilities, dtype=np.float64)

    def add_regrets(self, index: int, values: list[float], node_value: float) -> None:
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        self._regrets[chunk][row, :length] += np.asarray(values) - node_value

    def add_regrets_plus(self, index: int, values: list[float], node_value: float) -> None:
        """CFR+ regret update, stored without a temporary Python vector."""
        self.add_regrets(index, values, node_value)
        chunk, row = self._position(index)
        length = int(self._lengths[chunk][row])
        np.maximum(self._regrets[chunk][row, :length], 0.0, out=self._regrets[chunk][row, :length])

    def iter_indices(self) -> range:
        return range(self._count)

    def key_at(self, index: int) -> InformationKey:
        for encoded, stored in self._index.items():
            if stored == index:
                return self._decode_key(encoded)
        raise KeyError(index)

    def estimated_bytes(self) -> int:
        if not self._count:
            return sys.getsizeof(self._index)
        sample = list(self._index.keys())[: min(256, self._count)]
        key_bytes = sum(sys.getsizeof(key) for key in sample) / len(sample)
        array_bytes = sum(
            array.nbytes
            for group in (
                self._regrets,
                self._strategy,
                self._actions,
                self._lengths,
                self._timestamps,
            )
            for array in group
        )
        return int(sys.getsizeof(self._index) + key_bytes * self._count + array_bytes)

    def _materialize(self, index: int) -> MCCFRNode:
        from ..tabular_mccfr.mccfr_trainer import MCCFRNode

        return MCCFRNode(
            self.actions(index),
            list(self.regrets(index)),
            list(self.strategy_sum(index)),
            self.timestamp(index),
        )

    # Compatibility mapping used by small tests and legacy callers.  Training
    # hot paths use integer indices and never materialize these nodes.
    def __getitem__(self, key: InformationKey) -> MCCFRNode:
        index = self._index[self._encode_key(key)]
        return self._materialize(index)

    def __setitem__(self, key: InformationKey, node: MCCFRNode) -> None:
        encoded = self._encode_key(key)
        index = self._index.get(encoded)
        if index is None:
            index, _ = self.lookup(key, tuple(node.actions), node.last_discount_iteration)
        self.set_node(
            index,
            tuple(node.actions),
            node.regrets,
            node.strategy_sum,
            node.last_discount_iteration,
        )

    def get(self, key: InformationKey, default: Any = None) -> Any:
        index = self._index.get(self._encode_key(key))
        return default if index is None else self._materialize(index)

    def items(self) -> Iterator[tuple[InformationKey, MCCFRNode]]:
        for encoded, index in self._index.items():
            yield self._decode_key(encoded), self._materialize(index)

    def values(self) -> Iterator[MCCFRNode]:
        for index in self.iter_indices():
            yield self._materialize(index)

    def keys(self) -> list[InformationKey]:
        return [self._decode_key(encoded) for encoded in self._index]

    def __iter__(self) -> Iterator[InformationKey]:
        return iter(self.keys())

    def policy_payload(self) -> dict[str, Any]:
        probabilities: list[np.ndarray] = []
        for chunk_index, strategy in enumerate(self._strategy):
            result = np.zeros_like(strategy)
            lengths = self._lengths[chunk_index]
            for length in range(1, MAX_ACTIONS + 1):
                rows = np.flatnonzero(lengths == length)
                if not len(rows):
                    continue
                values = strategy[rows, :length]
                totals = values.sum(axis=1)
                positive = totals > 1e-15
                if np.any(positive):
                    result[rows[positive], :length] = values[positive] / totals[positive, None]
                if np.any(~positive):
                    result[rows[~positive], :length] = 1.0 / length
            probabilities.append(result)
        return {
            "format": (
                LEGACY_COMPACT_STORE_FORMAT if self._legacy_pickle_keys else COMPACT_STORE_FORMAT
            ),
            "index": self._index,
            "actions": self._actions,
            "lengths": self._lengths,
            "probabilities": probabilities,
            "count": self._count,
        }


class MMapCompactMCCFRStore(CompactMCCFRStore):
    """Numeric MCCFR slabs backed by durable mmap files.

    The key index remains a compact byte dictionary (fast state lookup), while
    the large numeric part of the table never has to be copied for a checkpoint
    or held as Python objects.  The OS page cache uses available RAM naturally;
    unmapped pages remain on SSD.  This is a mutable training representation,
    separate from the read-only packed deployment policy.
    """

    FORMAT = "mmap-mccfr-v1"

    def __init__(self, directory: Path, numeric_dtype: str = "float32") -> None:
        self.directory = Path(directory)
        self.numeric_dtype = np.dtype(numeric_dtype)
        self.directory.mkdir(parents=True, exist_ok=True)
        super().__init__()

    @property
    def _manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def _chunk_path(self, name: str, chunk: int) -> Path:
        return self.directory / f"{name}-{chunk:05d}.bin"

    def _map(self, name: str, chunk: int, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
        return np.memmap(self._chunk_path(name, chunk), mode="w+", dtype=dtype, shape=shape)

    def _ensure_chunk(self) -> None:
        if self._count % CHUNK_SIZE:
            return
        chunk = len(self._regrets)
        shape = (CHUNK_SIZE, MAX_ACTIONS)
        self._regrets.append(self._map("regrets", chunk, shape, self.numeric_dtype))
        self._strategy.append(self._map("strategy", chunk, shape, self.numeric_dtype))
        self._actions.append(self._map("actions", chunk, shape, np.dtype(np.int16)))
        self._lengths.append(self._map("lengths", chunk, (CHUNK_SIZE,), np.dtype(np.uint8)))
        self._timestamps.append(self._map("timestamps", chunk, (CHUNK_SIZE,), np.dtype(np.int32)))

    def flush(self) -> None:
        for group in (
            self._regrets,
            self._strategy,
            self._actions,
            self._lengths,
            self._timestamps,
        ):
            for array in group:
                array.flush()

    def close(self) -> None:
        """Release file handles explicitly (required before cleanup on Windows)."""
        self.flush()
        for group in (
            self._regrets,
            self._strategy,
            self._actions,
            self._lengths,
            self._timestamps,
        ):
            for array in group:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        self._regrets.clear()
        self._strategy.clear()
        self._actions.clear()
        self._lengths.clear()
        self._timestamps.clear()

    def save_manifest(self) -> dict[str, Any]:
        """Atomically persist the small index and slab metadata."""
        self.flush()
        index_path = self.directory / "index.pkl"
        temporary_index = index_path.with_suffix(".tmp")
        with temporary_index.open("wb") as handle:
            pickle.dump(self._index, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_index, index_path)
        payload = {
            "format": self.FORMAT,
            "count": self._count,
            "numeric_dtype": self.numeric_dtype.name,
            "chunk_size": CHUNK_SIZE,
            "max_actions": MAX_ACTIONS,
            "index": index_path.name,
        }
        temporary_manifest = self._manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, self._manifest_path)
        return payload

    @classmethod
    def restore(cls, directory: Path) -> MMapCompactMCCFRStore:
        directory = Path(directory)
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported mmap MCCFR store: {directory}")
        if int(payload["chunk_size"]) != CHUNK_SIZE or int(payload["max_actions"]) != MAX_ACTIONS:
            raise ValueError("mmap MCCFR layout differs from this solver build")
        store = cls(directory, str(payload["numeric_dtype"]))
        with (directory / str(payload["index"])).open("rb") as handle:
            store._index = pickle.load(handle)
        store._keys = store._index
        store._count = int(payload["count"])
        chunks = (store._count + CHUNK_SIZE - 1) // CHUNK_SIZE
        shape = (CHUNK_SIZE, MAX_ACTIONS)

        def open_map(
            name: str, chunk: int, mapped_shape: tuple[int, ...], dtype: np.dtype
        ) -> np.memmap:
            return np.memmap(
                store._chunk_path(name, chunk), mode="r+", dtype=dtype, shape=mapped_shape
            )

        for chunk in range(chunks):
            store._regrets.append(open_map("regrets", chunk, shape, store.numeric_dtype))
            store._strategy.append(open_map("strategy", chunk, shape, store.numeric_dtype))
            store._actions.append(open_map("actions", chunk, shape, np.dtype(np.int16)))
            store._lengths.append(open_map("lengths", chunk, (CHUNK_SIZE,), np.dtype(np.uint8)))
            store._timestamps.append(
                open_map("timestamps", chunk, (CHUNK_SIZE,), np.dtype(np.int32))
            )
        return store


class CompactPolicyEntries(Mapping[InformationKey, PolicyEntry]):
    """Lazy Mapping facade over a compact policy payload."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._legacy_pickle_keys = payload.get("format") == LEGACY_COMPACT_STORE_FORMAT
        self._index = payload["index"]
        self._actions = [np.asarray(item, dtype=np.int16) for item in payload["actions"]]
        self._lengths = [np.asarray(item, dtype=np.uint8) for item in payload["lengths"]]
        self._probabilities = [
            np.asarray(item, dtype=np.float64) for item in payload["probabilities"]
        ]
        self._count = int(payload["count"])

    def __len__(self) -> int:
        return self._count

    def _lookup(self, key: InformationKey) -> PolicyEntry | None:
        encoded = (
            pickle.dumps(key, protocol=5)
            if self._legacy_pickle_keys
            else encode_information_key(key)
        )
        index = self._index.get(encoded)
        if index is None:
            return None
        chunk, row = divmod(index, CHUNK_SIZE)
        length = int(self._lengths[chunk][row])
        actions = tuple(int(value) for value in self._actions[chunk][row, :length])
        probabilities = tuple(float(value) for value in self._probabilities[chunk][row, :length])
        return actions, probabilities

    def __getitem__(self, key: InformationKey) -> PolicyEntry:
        result = self._lookup(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, key: InformationKey, default: Any = None) -> Any:
        result = self._lookup(key)
        return default if result is None else result

    def __iter__(self) -> Iterator[InformationKey]:
        for encoded in self._index:
            yield (
                pickle.loads(encoded)
                if self._legacy_pickle_keys
                else decode_information_key(encoded)
            )

    def iter_entries(self) -> Iterator[tuple[InformationKey, PolicyEntry]]:
        """Stream keys and values without re-encoding every key for lookup."""
        for encoded, index in self._index.items():
            chunk, row = divmod(index, CHUNK_SIZE)
            length = int(self._lengths[chunk][row])
            actions = tuple(int(value) for value in self._actions[chunk][row, :length])
            probabilities = tuple(
                float(value) for value in self._probabilities[chunk][row, :length]
            )
            key = (
                pickle.loads(encoded)
                if self._legacy_pickle_keys
                else decode_information_key(encoded)
            )
            yield key, (actions, probabilities)


class AverageStrategyPolicyEntries(Mapping[InformationKey, PolicyEntry]):
    """Read-only, zero-copy average-policy view over a mutable training store.

    This adapter is intentionally used only for evaluation/export tooling.  It
    normalizes one ``strategy_sum`` row on demand and never allocates a second
    table containing probabilities for every information state.
    """

    def __init__(self, store: CompactMCCFRStore) -> None:
        self._store = store

    def __len__(self) -> int:
        return len(self._store)

    def _entry_at(self, index: int) -> PolicyEntry:
        actions = self._store.actions(index)
        strategy_sum = self._store.strategy_sum(index)
        total = sum(strategy_sum)
        probabilities = (
            tuple(value / total for value in strategy_sum)
            if total > 1e-15
            else (1.0 / len(actions),) * len(actions)
        )
        return actions, probabilities

    def get(self, key: InformationKey, default: Any = None) -> Any:
        index = self._store._index.get(self._store._encode_key(key))
        return default if index is None else self._entry_at(index)

    def __getitem__(self, key: InformationKey) -> PolicyEntry:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self) -> Iterator[InformationKey]:
        for encoded in self._store._index:
            yield self._store._decode_key(encoded)

    def iter_entries(self) -> Iterator[tuple[InformationKey, PolicyEntry]]:
        for encoded, index in self._store._index.items():
            yield self._store._decode_key(encoded), self._entry_at(index)

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if callable(close):
            close()


class CompactNodeMapping(Mapping[InformationKey, tuple[object, ...]]):
    """Compatibility view used when reading compact training payloads."""

    def __init__(self, store: CompactMCCFRStore) -> None:
        self._store = store

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, key: InformationKey) -> tuple[object, ...]:
        node = self._store[key]
        return (
            node.actions,
            tuple(node.regrets),
            tuple(node.strategy_sum),
            node.last_discount_iteration,
        )

    def __iter__(self) -> Iterator[InformationKey]:
        return iter(self._store)

    def keys(self) -> list[InformationKey]:
        return self._store.keys()

    def items(self) -> Iterator[tuple[InformationKey, tuple[object, ...]]]:
        for key in self._store:
            yield key, self[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())


class ShardedCompactPolicyEntries(Mapping[InformationKey, PolicyEntry]):
    """Lookup facade that keeps starting-player policy shards separate."""

    def __init__(self, shards: list[Mapping[InformationKey, PolicyEntry]]) -> None:
        self._shards = shards

    def __len__(self) -> int:
        return sum(len(shard) for shard in self._shards)

    def __getitem__(self, key: InformationKey) -> PolicyEntry:
        for shard in self._shards:
            result = shard.get(key)
            if result is not None:
                return result
        raise KeyError(key)

    def get(self, key: InformationKey, default: Any = None) -> Any:
        for shard in self._shards:
            result = shard.get(key)
            if result is not None:
                return result
        return default

    def __iter__(self) -> Iterator[InformationKey]:
        for shard in self._shards:
            yield from shard

    def iter_entries(self) -> Iterator[tuple[InformationKey, PolicyEntry]]:
        for shard in self._shards:
            streaming = getattr(shard, "iter_entries", None)
            yield from (streaming() if streaming is not None else shard.items())

    def iter_entry_sources(
        self,
    ) -> Iterator[Iterator[tuple[InformationKey, PolicyEntry]]]:
        """Expose shards separately for balanced bounded sampling."""
        for shard in self._shards:
            streaming = getattr(shard, "iter_entries", None)
            yield streaming() if streaming is not None else iter(shard.items())

    def close(self) -> None:
        for shard in self._shards:
            close = getattr(shard, "close", None)
            if callable(close):
                close()


class EnsembleCompactPolicyEntries(Mapping[InformationKey, PolicyEntry]):
    """Average independently trained policy replicas at lookup time.

    A replica that never visited a state is omitted from that state's average;
    treating it as uniform would systematically dilute the replicas that have
    actual samples.  The underlying starting-player shards stay lazy.
    """

    def __init__(self, replicas: list[Mapping[InformationKey, PolicyEntry]]) -> None:
        if not replicas:
            raise ValueError("an ensemble needs at least one policy replica")
        self._replicas = replicas
        self.requires_aggregation = True

    @property
    def replicas(self) -> tuple[Mapping[InformationKey, PolicyEntry], ...]:
        """Replica mappings without constructing their (potentially huge) union."""
        return tuple(self._replicas)

    def __len__(self) -> int:
        # A union count would require a second full key set.  This is an upper
        # bound used only for reporting and keeps deploy loading streaming.
        return sum(len(replica) for replica in self._replicas)

    def __getitem__(self, key: InformationKey) -> PolicyEntry:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, key: InformationKey, default: Any = None) -> Any:
        entries = [replica.get(key) for replica in self._replicas]
        present = [entry for entry in entries if entry is not None]
        if not present:
            return default
        actions = present[0][0]
        if any(entry[0] != actions for entry in present[1:]):
            raise ValueError("ensemble replica action order differs")
        count = len(present)
        probabilities = tuple(
            sum(entry[1][index] for entry in present) / count for index in range(len(actions))
        )
        return actions, probabilities

    def __iter__(self) -> Iterator[InformationKey]:
        # Iteration is intentionally source-streaming and may repeat a key.
        # Lookup is the deploy path; aggregation consumers should use
        # iter_entry_sources to preserve replica weights without a giant union.
        for replica in self._replicas:
            yield from replica

    def iter_entry_sources(
        self,
    ) -> Iterator[Iterator[tuple[InformationKey, PolicyEntry]]]:
        for replica in self._replicas:
            streaming = getattr(replica, "iter_entries", None)
            yield streaming() if streaming is not None else iter(replica.items())

    def iter_entries(self) -> Iterator[tuple[InformationKey, PolicyEntry]]:
        """Stream raw rows so abstraction sees every replica equally."""
        for source in self.iter_entry_sources():
            yield from source

    def close(self) -> None:
        """Release mmap-backed replicas when this is a deploy-time ensemble."""
        for replica in self._replicas:
            close = getattr(replica, "close", None)
            if callable(close):
                close()
