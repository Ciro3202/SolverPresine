from __future__ import annotations

"""Portable one-file bundle for the five round-specific multiplayer heads."""

import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from presine.game.config import GameConfig
from presine.game.observation import InformationState

from .linear import LinearBlueprintPolicy, LinearFeatureConfig

BUNDLE_FORMAT = "presine-multiplayer-bundle-v1"


def _payload_policy(payload: dict[str, Any]) -> LinearBlueprintPolicy:
    format_name = payload.get("format")
    if format_name not in {"presine-linear-blueprint-v1", "presine-tile-blueprint-v1"}:
        raise ValueError("bundle head is not a linear blueprint")
    config = LinearFeatureConfig(**dict(payload.get("config", {})))
    if "weights_f32_base64" in payload:
        raw = base64.b64decode(str(payload["weights_f32_base64"]))
        weights = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        weights = np.asarray(payload["weights"], dtype=np.float64)
    return LinearBlueprintPolicy(config, weights=weights)


def _read_head(path: Path) -> tuple[GameConfig, dict[str, Any]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    game = GameConfig.from_dict(dict(payload["game"]))
    _payload_policy(payload)
    return game, payload


class MultiplayerBlueprintBundle:
    """One deployable policy that selects a compact head by hand size."""

    def __init__(self, game: GameConfig, heads: dict[int, LinearBlueprintPolicy]) -> None:
        if not 3 <= game.players <= 6:
            raise ValueError("multiplayer bundles require 3..6 players")
        if set(heads) != {1, 2, 3, 4, 5}:
            raise ValueError("a multiplayer bundle needs heads for hand sizes 1..5")
        for hand_size, head in heads.items():
            if head.config.players != game.players:
                raise ValueError(f"head R{hand_size} has a different player count")
        self.game = game
        self.heads = dict(heads)

    def for_hand_size(self, hand_size: int) -> LinearBlueprintPolicy:
        try:
            return self.heads[hand_size]
        except KeyError as exc:
            raise ValueError(f"bundle has no head for hand size {hand_size}") from exc

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        return self.for_hand_size(info.hand_size).probabilities(info)

    def lookup(self, info: InformationState) -> tuple[float, ...]:
        return self.probabilities(info)

    def stats(self) -> dict[str, object]:
        return {
            "kind": BUNDLE_FORMAT,
            "players": self.game.players,
            "heads": {str(size): head.stats() for size, head in sorted(self.heads.items())},
        }


def save_multiplayer_bundle(
    game: GameConfig, heads: dict[int, LinearBlueprintPolicy], path: Path
) -> dict[str, object]:
    embedded: dict[str, object] = {}
    for hand_size, head in sorted(heads.items()):
        payload: dict[str, object] = {
            "format": "presine-tile-blueprint-v1"
            if head.config.tile_buckets
            else "presine-linear-blueprint-v1",
            "game": GameConfig(
                players=game.players,
                hand_size=hand_size,
                starting_player=game.starting_player,
                payoff=game.payoff,
            ).to_dict(),
            "config": asdict(head.config),
        }
        if head.config.tile_buckets:
            payload["weights_f32_base64"] = base64.b64encode(
                head.weights.astype("<f4", copy=False).tobytes()
            ).decode("ascii")
        else:
            payload["weights"] = head.weights.tolist()
        embedded[str(hand_size)] = payload
    payload = {
        "format": BUNDLE_FORMAT,
        "game": game.to_dict(),
        "heads": embedded,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {
        "format": BUNDLE_FORMAT,
        "players": game.players,
        "heads": 5,
        "path": str(path.resolve()),
    }


def load_multiplayer_bundle(path: Path) -> tuple[GameConfig, MultiplayerBlueprintBundle]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != BUNDLE_FORMAT:
        raise ValueError("unsupported multiplayer bundle format")
    game = GameConfig.from_dict(dict(payload["game"]))
    heads = {
        int(hand_size): _payload_policy(dict(head_payload))
        for hand_size, head_payload in dict(payload["heads"]).items()
    }
    return game, MultiplayerBlueprintBundle(game, heads)
