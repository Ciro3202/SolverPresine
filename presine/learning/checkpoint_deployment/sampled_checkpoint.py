from __future__ import annotations

import gzip
import os
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from presine.game.config import GameConfig
from presine.policies.tabular import TabularPolicy

from .compact_store import (
    AverageStrategyPolicyEntries,
    CompactMCCFRStore,
    CompactNodeMapping,
    CompactPolicyEntries,
    EnsembleCompactPolicyEntries,
    MMapCompactMCCFRStore,
    ShardedCompactPolicyEntries,
)

if TYPE_CHECKING:
    from ..tabular_mccfr.mccfr_trainer import MCCFRTrainer


SAMPLED_CHECKPOINT_FORMAT = 2
SHARDED_CHECKPOINT_FORMAT = 3
ENSEMBLE_CHECKPOINT_FORMAT = 4
TRAINING_KIND = "external-sampling-tabular-training"
POLICY_KIND = "external-sampling-tabular-average-policy"
SHARDED_TRAINING_KIND = "external-sampling-tabular-sharded-training"
SHARDED_POLICY_KIND = "external-sampling-tabular-sharded-average-policy"
ENSEMBLE_TRAINING_KIND = "external-sampling-tabular-ensemble-training"
ENSEMBLE_POLICY_KIND = "external-sampling-tabular-ensemble-average-policy"
LEGACY_KINDS = {"mccfr-plus-training", "mccfr-plus-average-policy"}
CHECKPOINT_CODEC = "gzip"


def _atomic_pickle_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        # Compress while streaming directly to the temporary file.  This keeps
        # the serialized shard small without creating a second uncompressed
        # pickle in memory or on disk.
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=1, mtime=0) as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary, path)
    finally:
        # A failed checkpoint must not leave a multi-gigabyte staging file
        # behind and consume the next checkpoint's quota.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_pickle(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as raw_handle:
            # Keep backward compatibility with the existing uncompressed
            # checkpoints.  New checkpoints are gzip streams, identified by
            # their standard two-byte magic header.
            magic = raw_handle.read(2)
            raw_handle.seek(0)
            if magic == b"\x1f\x8b":
                with gzip.GzipFile(fileobj=raw_handle, mode="rb") as handle:
                    payload = pickle.load(handle)
            else:
                payload = pickle.load(raw_handle)
    except Exception as error:
        raise ValueError(
            "incompatible legacy MCCFR checkpoint; MCCFR+ strategy sums and "
            "neural fallback state cannot be migrated to sampled-tabular format v2"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid sampled-tabular checkpoint: {path}")
    if payload.get("kind") in LEGACY_KINDS:
        raise ValueError(
            "incompatible legacy MCCFR+ checkpoint: old strategy_sum semantics "
            "cannot be converted safely"
        )
    return payload


def save_sampled_training_checkpoint(trainer: MCCFRTrainer, path: Path) -> None:
    payload: dict[str, Any] = {
        "format": SAMPLED_CHECKPOINT_FORMAT,
        "kind": TRAINING_KIND,
        "artifact_role": (
            "recoverable mutable training state; resume training from this "
            "checkpoint, but do not use it directly for game-time policy lookup"
        ),
        "codec": CHECKPOINT_CODEC,
        "game": trainer.game_config.to_dict(),
        "mccfr": trainer.training_config.to_dict(),
        "iteration": trainer.iteration,
        "nodes_visited": trainer.nodes_visited,
        "rng_state": trainer.rng.getstate(),
        "diagnostics": trainer.diagnostics,
        "information_states": len(trainer.nodes),
        "information_states_by_seat_phase": getattr(trainer, "_state_counts", {}),
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    if isinstance(trainer.nodes, MMapCompactMCCFRStore):
        trainer.nodes.save_manifest()
        # Sharded checkpoints live one directory below the run root, therefore
        # this may intentionally contain ``..``.  It remains run-local and is
        # resolved relative to the checkpoint on restore.
        relative = Path(os.path.relpath(trainer.nodes.directory, path.parent))
        payload["persistent_store"] = {
            "format": MMapCompactMCCFRStore.FORMAT,
            "directory": relative.as_posix(),
        }
    else:
        compact = getattr(trainer.nodes, "to_payload", None)
        if compact is not None:
            payload["nodes_compact"] = compact()
        else:
            payload["nodes"] = {
                key: (
                    node.actions,
                    tuple(node.regrets),
                    tuple(node.strategy_sum),
                    node.last_discount_iteration,
                )
                for key, node in trainer.nodes.items()
            }
    _atomic_pickle_save(payload, path)


def save_sampled_policy_checkpoint(trainer: MCCFRTrainer, path: Path) -> None:
    payload: dict[str, Any] = {
        "format": SAMPLED_CHECKPOINT_FORMAT,
        "kind": POLICY_KIND,
        "artifact_role": (
            "read-only average strategy derived from training; this is a policy "
            "artifact, not a resumable training checkpoint"
        ),
        "codec": CHECKPOINT_CODEC,
        "game": trainer.game_config.to_dict(),
        "mccfr": trainer.training_config.to_dict(),
        "iteration": trainer.iteration,
        "diagnostics": trainer.diagnostics,
        "warning": (
            "external-sampling guarantees convergence in expectation/high "
            "probability under its assumptions; sampled regrets are not an "
            "exploitability certificate"
        ),
    }
    if isinstance(trainer.nodes, MMapCompactMCCFRStore):
        raise ValueError(
            "mmap MCCFR training tables are not serialised as policy.pt; "
            "run the streaming packed-policy exporter after training"
        )
    compact = getattr(trainer.nodes, "policy_payload", None)
    if compact is not None:
        payload["entries_compact"] = compact()
    else:
        payload["entries"] = trainer.policy_entries()
    _atomic_pickle_save(payload, path)


def load_sampled_training_payload(path: Path) -> dict[str, Any]:
    payload = _load_pickle(path)
    if (
        payload.get("format") == SHARDED_CHECKPOINT_FORMAT
        and payload.get("kind") == SHARDED_TRAINING_KIND
    ):
        return payload
    if (
        payload.get("format") == ENSEMBLE_CHECKPOINT_FORMAT
        and payload.get("kind") == ENSEMBLE_TRAINING_KIND
    ):
        return payload
    if payload.get("format") != SAMPLED_CHECKPOINT_FORMAT or payload.get("kind") != TRAINING_KIND:
        raise ValueError(
            "unsupported or legacy sampled-tabular training checkpoint; format v2 is required"
        )
    if "nodes_compact" in payload:
        payload["nodes"] = CompactNodeMapping(
            CompactMCCFRStore.from_payload(payload["nodes_compact"])
        )
    return payload


def load_sampled_training_policy(path: Path) -> tuple[GameConfig, TabularPolicy]:
    """Open a zero-copy average-policy view over a training checkpoint.

    The returned policy is for evaluation or streaming export only.  For mmap
    checkpoints it owns live mappings to the mutable run directory, which must
    remain intact until ``policy.entries.close()`` is called.
    """
    path = Path(path)
    payload = _load_pickle(path)
    game = GameConfig.from_dict(payload["game"])

    if (
        payload.get("format") == ENSEMBLE_CHECKPOINT_FORMAT
        and payload.get("kind") == ENSEMBLE_TRAINING_KIND
    ):
        replicas = []
        try:
            for relative in payload.get("replicas", []):
                replica_game, replica_policy = load_sampled_training_policy(
                    path.parent / str(relative)
                )
                if replica_game != game:
                    raise ValueError("ensemble training replica game differs")
                replicas.append(replica_policy.entries)
        except Exception:
            for entries in replicas:
                close = getattr(entries, "close", None)
                if callable(close):
                    close()
            raise
        if not replicas:
            raise ValueError("ensemble training checkpoint has no replicas")
        return game, TabularPolicy(EnsembleCompactPolicyEntries(replicas))

    if (
        payload.get("format") == SHARDED_CHECKPOINT_FORMAT
        and payload.get("kind") == SHARDED_TRAINING_KIND
    ):
        shards = []
        try:
            for relative in payload.get("shards", []):
                shard_game, shard_policy = load_sampled_training_policy(path.parent / str(relative))
                if replace(shard_game, starting_player=game.starting_player) != game:
                    raise ValueError("training shard game differs")
                shards.append(shard_policy.entries)
        except Exception:
            for entries in shards:
                close = getattr(entries, "close", None)
                if callable(close):
                    close()
            raise
        if not shards:
            raise ValueError("sharded training checkpoint has no shards")
        return game, TabularPolicy(ShardedCompactPolicyEntries(shards))

    if payload.get("format") != SAMPLED_CHECKPOINT_FORMAT or payload.get("kind") != TRAINING_KIND:
        raise ValueError("not a sampled-tabular training checkpoint")
    if "persistent_store" in payload:
        directory = path.parent / str(payload["persistent_store"]["directory"])
        store = MMapCompactMCCFRStore.restore(directory)
    elif "nodes_compact" in payload:
        store = CompactMCCFRStore.from_payload(payload["nodes_compact"])
    else:
        raise ValueError("training checkpoint does not expose a compact strategy-sum store")
    return game, TabularPolicy(AverageStrategyPolicyEntries(store))


def load_sampled_policy(path: Path) -> tuple[GameConfig, TabularPolicy]:
    payload = _load_pickle(path)
    if (
        payload.get("format") == ENSEMBLE_CHECKPOINT_FORMAT
        and payload.get("kind") == ENSEMBLE_POLICY_KIND
    ):
        replicas: list[ShardedCompactPolicyEntries] = []
        for relative in payload.get("replicas", []):
            replica_path = path.parent / str(relative)
            game, policy = load_sampled_policy(replica_path)
            if game != GameConfig.from_dict(payload["game"]):
                raise ValueError(f"ensemble replica game differs: {replica_path}")
            if not isinstance(policy.entries, ShardedCompactPolicyEntries):
                raise ValueError(f"invalid ensemble policy replica: {replica_path}")
            replicas.append(policy.entries)
        if not replicas:
            raise ValueError("ensemble policy has no replicas")
        return GameConfig.from_dict(payload["game"]), TabularPolicy(
            EnsembleCompactPolicyEntries(replicas)
        )
    if (
        payload.get("format") == SHARDED_CHECKPOINT_FORMAT
        and payload.get("kind") == SHARDED_POLICY_KIND
    ):
        compact_shards: list[CompactPolicyEntries] = []
        if payload.get("shards"):
            shard_payloads = [
                _load_pickle(path.parent / str(relative)) for relative in payload["shards"]
            ]
            if all("entries_compact" in shard for shard in shard_payloads):
                compact_shards = [
                    CompactPolicyEntries(shard["entries_compact"]) for shard in shard_payloads
                ]
                return GameConfig.from_dict(payload["game"]), TabularPolicy(
                    ShardedCompactPolicyEntries(compact_shards)
                )
        entries: dict[object, Any] = {}
        for relative, shard in zip(payload["shards"], shard_payloads):
            shard_path = path.parent / str(relative)
            if shard.get("format") != SAMPLED_CHECKPOINT_FORMAT or shard.get("kind") != POLICY_KIND:
                raise ValueError(f"invalid sampled policy shard: {shard_path}")
            if "entries_compact" in shard:
                compact_shard = CompactPolicyEntries(shard["entries_compact"])
                for key in compact_shard:
                    if key in entries:
                        raise ValueError("starting-player policy shards overlap")
                    entries[key] = compact_shard[key]
            else:
                overlap = entries.keys() & shard["entries"].keys()
                if overlap:
                    raise ValueError("starting-player policy shards overlap")
                entries.update(shard["entries"])
        normalized = {
            key: (tuple(actions), tuple(probabilities))
            for key, (actions, probabilities) in entries.items()
        }
        return GameConfig.from_dict(payload["game"]), TabularPolicy(normalized)
    if payload.get("format") != SAMPLED_CHECKPOINT_FORMAT or payload.get("kind") != POLICY_KIND:
        raise ValueError(
            "unsupported or legacy sampled-tabular policy checkpoint; format v2 is required"
        )
    if "entries_compact" in payload:
        return GameConfig.from_dict(payload["game"]), TabularPolicy(
            CompactPolicyEntries(payload["entries_compact"])
        )
    entries = {
        key: (tuple(actions), tuple(probabilities))
        for key, (actions, probabilities) in payload["entries"].items()
    }
    return GameConfig.from_dict(payload["game"]), TabularPolicy(entries)


def is_sampled_checkpoint(path: Path) -> bool:
    """Cheap signature check used before importing the neural checkpoint stack."""
    try:
        payload = _load_pickle(path)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("kind") in {
        TRAINING_KIND,
        POLICY_KIND,
        SHARDED_TRAINING_KIND,
        SHARDED_POLICY_KIND,
        ENSEMBLE_TRAINING_KIND,
        ENSEMBLE_POLICY_KIND,
    }


def save_sharded_manifest(payload: dict[str, Any], path: Path) -> None:
    """Atomically publish a small manifest after every shard is durable."""
    _atomic_pickle_save(payload, path)
