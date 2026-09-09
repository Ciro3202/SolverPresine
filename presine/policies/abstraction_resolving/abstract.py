from __future__ import annotations

"""Card-bucket blueprint used as the compact offline strategy layer."""

import gzip
import os
import pickle
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from presine.game.cards import ACE_OF_DENARI, NUM_CARDS
from presine.game.config import GameConfig
from presine.game.observation import EventKind, InformationState, PublicEvent
from presine.game.state import Phase

from ..tabular import InformationKey, PolicyEntry

ABSTRACT_POLICY_FORMAT = "presine-abstract-blueprint-v1"


def information_from_key(key: InformationKey) -> InformationState:
    if len(key) != 13:
        raise ValueError("unsupported information-key shape")
    return InformationState(
        player=int(key[0]),
        phase=int(key[1]),
        hand_size=int(key[2]),
        starting_player=int(key[3]),
        leader=int(key[4]),
        trick_index=int(key[5]),
        bids=tuple(key[6]),
        catches=tuple(key[7]),
        visible_cards=tuple(key[8]),
        initial_private_observation=tuple(key[9]),
        current_trick=tuple(key[10]),
        public_history=tuple(
            PublicEvent(EventKind(kind), actor, value) for kind, actor, value in key[11]
        ),
        legal_actions=tuple(key[12]),
    )


class CardAbstraction:
    """Order-aware buckets with the special Ace of Denari kept separate."""

    def __init__(self, card_buckets: int = 8) -> None:
        if not 2 <= card_buckets <= 40:
            raise ValueError("card_buckets must be in 2..40")
        self.card_buckets = card_buckets

    def card(self, card: int) -> int:
        if card == ACE_OF_DENARI:
            return self.card_buckets
        return min(self.card_buckets - 1, card * self.card_buckets // NUM_CARDS)

    def cards(self, cards: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(self.card(card) for card in cards))

    def key(self, info: InformationState) -> InformationKey:
        history = tuple(
            (
                int(event.kind),
                event.actor,
                self.card(event.value) if event.kind == EventKind.PLAY else event.value,
            )
            for event in info.public_history
        )
        return (
            info.player,
            info.phase,
            info.hand_size,
            info.starting_player,
            info.leader,
            info.trick_index,
            info.bids,
            info.catches,
            self.cards(info.visible_cards),
            self.cards(info.initial_private_observation),
            tuple((actor, self.card(card)) for actor, card in info.current_trick),
            history,
            len(info.legal_actions),
        )

    @staticmethod
    def action_tokens(info: InformationState) -> tuple[int, ...]:
        if info.phase == Phase.PLAY:
            order = {card: index for index, card in enumerate(sorted(info.legal_actions))}
            return tuple(order[action] for action in info.legal_actions)
        return info.legal_actions


class AbstractBlueprintPolicy:
    def __init__(
        self,
        entries: Mapping[InformationKey, PolicyEntry],
        *,
        card_buckets: int = 8,
    ) -> None:
        self.entries = entries
        self.abstraction = CardAbstraction(card_buckets)
        self.hits = 0
        self.misses = 0

    def lookup(self, info: InformationState) -> tuple[float, ...] | None:
        entry = self.entries.get(self.abstraction.key(info))
        if entry is None:
            self.misses += 1
            return None
        tokens, probabilities = entry
        expected = self.abstraction.action_tokens(info)
        if tokens != expected or len(probabilities) != len(expected):
            self.misses += 1
            return None
        self.hits += 1
        total = sum(probabilities)
        return tuple(value / total for value in probabilities)

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        result = self.lookup(info)
        return (
            result
            if result is not None
            else (1.0 / len(info.legal_actions),) * len(info.legal_actions)
        )

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "lookups": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "card_buckets": self.abstraction.card_buckets,
        }


def _iter_entries(
    entries: Mapping[InformationKey, PolicyEntry],
) -> Iterator[tuple[InformationKey, PolicyEntry]]:
    streaming = getattr(entries, "iter_entries", None)
    yield from (streaming() if streaming is not None else entries.items())


def build_abstract_blueprint(
    entries: Mapping[InformationKey, PolicyEntry], *, card_buckets: int = 8
) -> dict[InformationKey, PolicyEntry]:
    abstraction = CardAbstraction(card_buckets)
    totals: dict[InformationKey, tuple[tuple[int, ...], list[float], int]] = {}
    for key, (_, probabilities) in _iter_entries(entries):
        info = information_from_key(key)
        abstract_key = abstraction.key(info)
        tokens = abstraction.action_tokens(info)
        current = totals.get(abstract_key)
        if current is None:
            totals[abstract_key] = (tokens, list(probabilities), 1)
        else:
            old_tokens, sums, count = current
            if old_tokens != tokens:
                # Legal-action count is in the key, so this indicates an
                # abstraction bug rather than a bucket collision.
                raise ValueError("abstract action layouts collided")
            for index, probability in enumerate(probabilities):
                sums[index] += probability
            totals[abstract_key] = (tokens, sums, count + 1)
    return {
        key: (tokens, tuple(value / count for value in sums))
        for key, (tokens, sums, count) in totals.items()
    }


def save_abstract_blueprint(
    game: GameConfig,
    entries: Mapping[InformationKey, PolicyEntry],
    path: Path,
    *,
    card_buckets: int = 8,
) -> dict[str, object]:
    abstract = build_abstract_blueprint(entries, card_buckets=card_buckets)
    payload = {
        "format": ABSTRACT_POLICY_FORMAT,
        "game": game.to_dict(),
        "card_buckets": card_buckets,
        "source_entries": len(entries),
        "entries": abstract,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return {
        "source_entries": len(entries),
        "abstract_entries": len(abstract),
        "card_buckets": card_buckets,
    }


def load_abstract_blueprint(path: Path) -> tuple[GameConfig, AbstractBlueprintPolicy]:
    with gzip.open(path, "rb") as handle:
        payload: dict[str, Any] = pickle.load(handle)
    if payload.get("format") != ABSTRACT_POLICY_FORMAT:
        raise ValueError("unsupported abstract blueprint format")
    return GameConfig.from_dict(payload["game"]), AbstractBlueprintPolicy(
        payload["entries"], card_buckets=int(payload["card_buckets"])
    )


def load_blueprint(path: Path) -> tuple[GameConfig, object]:
    """Load either the legacy CFR-derived blueprint or the table-free one."""
    with path.open("rb") as handle:
        prefix = handle.read(2)
    if prefix == b"\x1f\x8b":
        return load_abstract_blueprint(path)
    import json

    with path.open("r", encoding="utf-8") as handle:
        format_name = json.load(handle).get("format")
    if format_name == "presine-multiplayer-bundle-v1":
        from ..multiplayer_bundle import load_multiplayer_bundle

        return load_multiplayer_bundle(path)
    if format_name == "presine-feature-blueprint-v1":
        from ..compact import load_compact_blueprint

        return load_compact_blueprint(path)
    if format_name in {"presine-linear-blueprint-v1", "presine-tile-blueprint-v1"}:
        from ..linear import load_linear_blueprint

        return load_linear_blueprint(path)
    raise ValueError("unsupported blueprint format")
