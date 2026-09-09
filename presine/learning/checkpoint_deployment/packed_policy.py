from __future__ import annotations

"""Read-only, sharded and memory-mapped policy deployment format."""

import hashlib
import json
import mmap
import os
import shutil
import struct
import tempfile
import zlib
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from presine.game.config import GameConfig
from presine.policies.tabular import InformationKey, PolicyEntry, TabularPolicy

from ..key_codec import PACKED_KEY_CODEC, encode_information_key

FORMAT_V1 = "presine-packed-policy-v1"
FORMAT = "presine-packed-policy-v2"
ENSEMBLE_FORMAT_V1 = "presine-packed-policy-ensemble-v1"
ENSEMBLE_FORMAT = "presine-packed-policy-ensemble-v2"
MAGIC = b"PSPOL01\0"
HEADER = struct.Struct("<8sQ")
INDEX = struct.Struct("<QI")  # hash, record offset (<4 GiB per shard)
COMPRESSED_MAGIC = b"PSPOL02\0"
COMPRESSED_HEADER = struct.Struct("<8sQQ")  # magic, entries, compressed blocks
COMPRESSED_INDEX = struct.Struct("<QII")  # hash, block, offset in raw block
BLOCK_INDEX = struct.Struct("<QII")  # compressed offset, compressed bytes, raw bytes
BLOCK_TARGET_BYTES = 64 * 1024
RECORD_HEADER = struct.Struct("<IB")  # key length, action count
PROBABILITY = struct.Struct("<H")


def adaptive_shard_count(entries: object) -> int:
    """Choose a power-of-two shard count without creating micro-shards."""
    try:
        count = len(entries)  # type: ignore[arg-type]
    except TypeError:
        # Streaming/ensemble mappings may not expose their size before export.
        return 256
    target_entries_per_shard = 200_000
    required = max(1, (count + target_entries_per_shard - 1) // target_entries_per_shard)
    return min(256, 1 << (required - 1).bit_length())


def _hash_key(key: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def _quantize(probabilities: tuple[float, ...]) -> tuple[int, ...]:
    if not probabilities:
        raise ValueError("policy entry has no probabilities")
    total = sum(probabilities)
    if total <= 0 or any(value < 0 for value in probabilities):
        raise ValueError("invalid policy probabilities")
    scaled = [value / total * 65535.0 for value in probabilities]
    result = [int(value) for value in scaled]
    remainder = 65535 - sum(result)
    order = sorted(
        range(len(result)), key=lambda index: scaled[index] - result[index], reverse=True
    )
    for index in order[:remainder]:
        result[index] += 1
    return tuple(result)


def _entry_iterator(
    entries: Mapping[InformationKey, PolicyEntry],
) -> Iterator[tuple[InformationKey, PolicyEntry]]:
    streaming = getattr(entries, "iter_entries", None)
    yield from (streaming() if streaming is not None else entries.items())


def export_packed_policy(
    game: GameConfig,
    entries: Mapping[InformationKey, PolicyEntry],
    output_dir: Path,
    *,
    shards: int = 256,
    staging_dir: Path | None = None,
) -> dict[str, object]:
    """Write an atomic deploy directory without materializing another table.

    Entries are partitioned first, then each partition is sorted independently.
    Peak exporter memory is therefore bounded by the largest shard rather than
    by the complete strategy.
    """
    if shards <= 0 or shards & (shards - 1):
        raise ValueError("packed policy shard count must be a power of two")
    if getattr(entries, "requires_aggregation", False):
        replicas = getattr(entries, "replicas", None)
        if replicas is None:
            raise ValueError("packed ensemble export requires explicit replica mappings")
        return _export_packed_ensemble(
            game, tuple(replicas), output_dir, shards=shards, staging_dir=staging_dir
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if staging_dir is None:
        staging = output_dir / ".packing"
        staging.mkdir(exist_ok=True)
    else:
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="presine-policy-pack-", dir=staging_dir))
    raw_paths = [staging / f"raw-{index:04d}.bin" for index in range(shards)]
    handles: dict[int, Any] = {}
    count = 0
    try:
        for key, (actions, probabilities) in _entry_iterator(entries):
            if tuple(key[-1]) != tuple(actions):
                raise ValueError("policy actions differ from information-key legal actions")
            encoded = encode_information_key(key)
            digest = _hash_key(encoded)
            shard = digest & (shards - 1)
            handle = handles.get(shard)
            if handle is None:
                handle = raw_paths[shard].open("ab")
                handles[shard] = handle
            quantized = _quantize(tuple(probabilities))
            record = bytearray(RECORD_HEADER.pack(len(encoded), len(actions)))
            record.extend(encoded)
            for probability in quantized:
                record.extend(PROBABILITY.pack(probability))
            handle.write(struct.pack("<QI", digest, len(record)))
            handle.write(record)
            count += 1
    finally:
        for handle in handles.values():
            handle.close()

    shard_files: list[str] = []
    for shard, raw_path in enumerate(raw_paths):
        records: list[tuple[int, bytes]] = []
        if raw_path.exists():
            with raw_path.open("rb") as handle:
                while header := handle.read(12):
                    digest, length = struct.unpack("<QI", header)
                    record = handle.read(length)
                    if len(record) != length:
                        raise ValueError("truncated packed-policy staging record")
                    records.append((digest, record))
        records.sort(key=lambda item: item[0])
        blocks: list[tuple[bytes, int]] = []
        compressed_index: list[tuple[int, int, int]] = []
        raw_block = bytearray()
        for digest, record in records:
            if raw_block and len(raw_block) + len(record) > BLOCK_TARGET_BYTES:
                blocks.append((zlib.compress(raw_block, level=1), len(raw_block)))
                raw_block = bytearray()
            compressed_index.append((digest, len(blocks), len(raw_block)))
            raw_block.extend(record)
        if raw_block:
            blocks.append((zlib.compress(raw_block, level=1), len(raw_block)))
        name = f"policy-{shard:04d}.bin"
        temporary = output_dir / (name + f".tmp-{os.getpid()}")
        final = output_dir / name
        with temporary.open("wb") as handle:
            handle.write(COMPRESSED_HEADER.pack(COMPRESSED_MAGIC, len(records), len(blocks)))
            for digest, block, offset in compressed_index:
                handle.write(COMPRESSED_INDEX.pack(digest, block, offset))
            data_offset = (
                COMPRESSED_HEADER.size
                + COMPRESSED_INDEX.size * len(compressed_index)
                + BLOCK_INDEX.size * len(blocks)
            )
            for compressed, raw_size in blocks:
                handle.write(BLOCK_INDEX.pack(data_offset, len(compressed), raw_size))
                data_offset += len(compressed)
            for compressed, _ in blocks:
                handle.write(compressed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        shard_files.append(name)
        if raw_path.exists():
            raw_path.unlink()
    shutil.rmtree(staging)

    manifest: dict[str, object] = {
        "format": FORMAT,
        "key_codec": PACKED_KEY_CODEC,
        "probability_codec": "uint16-sum-65535",
        "record_codec": "zlib-block-v1",
        "block_bytes": BLOCK_TARGET_BYTES,
        "game": game.to_dict(),
        "entries": count,
        "shards": shard_files,
    }
    temporary_manifest = output_dir / f"manifest.json.tmp-{os.getpid()}"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, output_dir / "manifest.json")
    return manifest


def _export_packed_ensemble(
    game: GameConfig,
    replicas: tuple[Mapping[InformationKey, PolicyEntry], ...],
    output_dir: Path,
    *,
    shards: int,
    staging_dir: Path | None,
) -> dict[str, object]:
    """Pack replicas independently and average only at query time.

    Partially overlapping replica tables cannot be merged without a second
    giant key set.  Separate deploy directories retain the original
    "average only present rows" semantics without materializing that union.
    """
    if not replicas:
        raise ValueError("packed ensemble has no replicas")
    output_dir.mkdir(parents=True, exist_ok=True)
    replica_dirs: list[str] = []
    entries_upper_bound = 0
    for index, replica in enumerate(replicas):
        if getattr(replica, "requires_aggregation", False):
            raise ValueError("nested packed ensembles are not supported")
        name = f"replica-{index:04d}"
        manifest = export_packed_policy(
            game, replica, output_dir / name, shards=shards, staging_dir=staging_dir
        )
        replica_dirs.append(name)
        entries_upper_bound += int(manifest["entries"])
    payload: dict[str, object] = {
        "format": ENSEMBLE_FORMAT,
        "game": game.to_dict(),
        "replicas": replica_dirs,
        "entries_upper_bound": entries_upper_bound,
        "shards_per_replica": shards,
        "aggregation": "mean-over-present-replica-rows",
    }
    temporary = output_dir / f"manifest.json.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "manifest.json")
    return payload


class _MappedShard:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.count = HEADER.unpack_from(self._map)
        if magic != MAGIC:
            raise ValueError(f"invalid packed policy shard: {path}")

    def close(self) -> None:
        self._map.close()
        self._handle.close()

    def get(self, digest: int, encoded: bytes, actions: tuple[int, ...]) -> PolicyEntry | None:
        low, high = 0, self.count
        while low < high:
            middle = (low + high) // 2
            candidate, _ = INDEX.unpack_from(self._map, HEADER.size + middle * INDEX.size)
            if candidate < digest:
                low = middle + 1
            else:
                high = middle
        index = low
        while index < self.count:
            candidate, offset = INDEX.unpack_from(self._map, HEADER.size + index * INDEX.size)
            if candidate != digest:
                break
            key_length, action_count = RECORD_HEADER.unpack_from(self._map, offset)
            cursor = offset + RECORD_HEADER.size
            if self._map[cursor : cursor + key_length] == encoded:
                cursor += key_length
                probabilities: list[float] = []
                for _ in range(action_count):
                    (probability,) = PROBABILITY.unpack_from(self._map, cursor)
                    cursor += PROBABILITY.size
                    probabilities.append(probability / 65535.0)
                if len(actions) != action_count:
                    raise ValueError("packed policy record action count is corrupt")
                return actions, tuple(probabilities)
            index += 1
        return None


class MMapPolicyEntries(Mapping[InformationKey, PolicyEntry]):
    """Lookup-only mapping whose index and records remain OS-page-backed."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self._count = int(manifest["entries"])
        self._shards = [_MappedShard(root / str(name)) for name in manifest["shards"]]

    def __len__(self) -> int:
        return self._count

    def get(self, key: InformationKey, default: Any = None) -> Any:
        encoded = encode_information_key(key)
        digest = _hash_key(encoded)
        actions = tuple(int(action) for action in key[-1])
        result = self._shards[digest & (len(self._shards) - 1)].get(digest, encoded, actions)
        return default if result is None else result

    def __getitem__(self, key: InformationKey) -> PolicyEntry:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self) -> Iterator[InformationKey]:
        raise TypeError("packed deploy policies are lookup-only")

    def close(self) -> None:
        for shard in self._shards:
            shard.close()


class _CompressedMappedShard:
    """Mmap index plus a bounded cache of decompressed policy-record blocks."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.count, self.block_count = COMPRESSED_HEADER.unpack_from(self._map)
        if magic != COMPRESSED_MAGIC:
            raise ValueError(f"invalid compressed packed policy shard: {path}")
        self._index_start = COMPRESSED_HEADER.size
        self._blocks_start = self._index_start + self.count * COMPRESSED_INDEX.size
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def close(self) -> None:
        self._cache.clear()
        self._map.close()
        self._handle.close()

    def _block(self, block: int) -> bytes:
        cached = self._cache.get(block)
        if cached is not None:
            self._cache.move_to_end(block)
            return cached
        if block >= self.block_count:
            raise ValueError("packed policy block index is corrupt")
        offset, compressed_size, raw_size = BLOCK_INDEX.unpack_from(
            self._map, self._blocks_start + block * BLOCK_INDEX.size
        )
        raw = zlib.decompress(self._map[offset : offset + compressed_size])
        if len(raw) != raw_size:
            raise ValueError("packed policy block length is corrupt")
        self._cache[block] = raw
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        return raw

    def get(self, digest: int, encoded: bytes, actions: tuple[int, ...]) -> PolicyEntry | None:
        low, high = 0, self.count
        while low < high:
            middle = (low + high) // 2
            candidate, _, _ = COMPRESSED_INDEX.unpack_from(
                self._map, self._index_start + middle * COMPRESSED_INDEX.size
            )
            if candidate < digest:
                low = middle + 1
            else:
                high = middle
        index = low
        while index < self.count:
            candidate, block, offset = COMPRESSED_INDEX.unpack_from(
                self._map, self._index_start + index * COMPRESSED_INDEX.size
            )
            if candidate != digest:
                break
            raw = self._block(block)
            key_length, action_count = RECORD_HEADER.unpack_from(raw, offset)
            cursor = offset + RECORD_HEADER.size
            if raw[cursor : cursor + key_length] == encoded:
                cursor += key_length
                probabilities = tuple(
                    PROBABILITY.unpack_from(raw, cursor + position * PROBABILITY.size)[0] / 65535.0
                    for position in range(action_count)
                )
                if len(actions) != action_count:
                    raise ValueError("packed policy record action count is corrupt")
                return actions, probabilities
            index += 1
        return None


class CompressedMMapPolicyEntries(Mapping[InformationKey, PolicyEntry]):
    """Lookup-only deploy mapping with mmap indexes and compressed record blocks."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self._count = int(manifest["entries"])
        self._shards = [_CompressedMappedShard(root / str(name)) for name in manifest["shards"]]

    def __len__(self) -> int:
        return self._count

    def get(self, key: InformationKey, default: Any = None) -> Any:
        encoded = encode_information_key(key)
        digest = _hash_key(encoded)
        actions = tuple(int(action) for action in key[-1])
        result = self._shards[digest & (len(self._shards) - 1)].get(digest, encoded, actions)
        return default if result is None else result

    def __getitem__(self, key: InformationKey) -> PolicyEntry:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self) -> Iterator[InformationKey]:
        raise TypeError("packed deploy policies are lookup-only")

    def close(self) -> None:
        for shard in self._shards:
            shard.close()


def load_packed_policy(path: Path) -> tuple[GameConfig, TabularPolicy]:
    root = path if path.is_dir() else path.parent
    manifest_path = root / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") in {ENSEMBLE_FORMAT_V1, ENSEMBLE_FORMAT}:
        from .compact_store import EnsembleCompactPolicyEntries

        game = GameConfig.from_dict(manifest["game"])
        replica_entries: list[Mapping[InformationKey, PolicyEntry]] = []
        try:
            for relative in manifest.get("replicas", []):
                replica_path = root / str(relative)
                replica_game, replica_policy = load_packed_policy(replica_path)
                if replica_game != game:
                    raise ValueError(f"packed ensemble replica game differs: {replica_path}")
                replica_entries.append(replica_policy.entries)
        except Exception:
            for entries in replica_entries:
                close = getattr(entries, "close", None)
                if callable(close):
                    close()
            raise
        if not replica_entries:
            raise ValueError("packed ensemble has no replicas")
        return game, TabularPolicy(EnsembleCompactPolicyEntries(replica_entries))
    if manifest.get("format") not in {FORMAT_V1, FORMAT}:
        raise ValueError("unsupported packed policy format")
    shards = manifest.get("shards", [])
    if not shards or len(shards) & (len(shards) - 1):
        raise ValueError("packed policy must contain a power-of-two shard count")
    entries: Mapping[InformationKey, PolicyEntry]
    if manifest.get("format") == FORMAT:
        entries = CompressedMMapPolicyEntries(root, manifest)
    else:
        entries = MMapPolicyEntries(root, manifest)
    return GameConfig.from_dict(manifest["game"]), TabularPolicy(entries)
