from __future__ import annotations

"""Small, table-free blueprint used by the online R4/R5 resolver.

This is intentionally not a compressed CFR table.  It contains a handful of
interpretable parameters and converts the current information state directly
into a distribution over the *currently legal* actions.  ISMCTS subsequently
improves that distribution with a local search.
"""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from presine.game.cards import ACE_OF_DENARI, NUM_CARDS
from presine.game.config import GameConfig
from presine.game.observation import InformationState
from presine.game.state import Phase

from .abstraction_resolving.canonical import canonical_information_key

COMPACT_BLUEPRINT_FORMAT = "presine-feature-blueprint-v1"


@dataclass(frozen=True, slots=True)
class CompactBlueprintConfig:
    """Interpretable, bounded policy parameters.

    ``strength_buckets`` is part of the feature signature and is deliberately
    small.  It can be swept later without changing the game model or retaining
    state rows.  Temperatures make this a distribution rather than a brittle
    deterministic heuristic, which gives local search useful exploration.
    """

    strength_buckets: int = 8
    bid_temperature: float = 0.70
    play_temperature: float = 0.42
    high_card_threshold: float = 0.58
    ace_strength_bonus: float = 0.80

    def __post_init__(self) -> None:
        if not 2 <= self.strength_buckets <= 20:
            raise ValueError("strength_buckets must be in 2..20")
        if self.bid_temperature <= 0 or self.play_temperature <= 0:
            raise ValueError("blueprint temperatures must be positive")
        if not 0.0 < self.high_card_threshold < 1.0:
            raise ValueError("high_card_threshold must be in (0, 1)")
        if self.ace_strength_bonus < 0:
            raise ValueError("ace_strength_bonus cannot be negative")


def _softmax(scores: list[float], temperature: float) -> tuple[float, ...]:
    if not scores:
        return ()
    maximum = max(scores)
    weights = [math.exp((score - maximum) / temperature) for score in scores]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


class CompactBlueprintPolicy:
    """Table-free action prior with semantic, action-relative features."""

    def __init__(self, config: CompactBlueprintConfig | None = None) -> None:
        self.config = config or CompactBlueprintConfig()
        self.calls = 0
        self.feature_signatures = 0

    def feature_key(self, info: InformationState) -> tuple[object, ...]:
        """Bounded abstraction used for diagnostics and future parameter fitting.

        The policy itself never keeps a mapping keyed by this value.  Therefore
        the number of visited states cannot make deployment memory grow.
        """
        exact = canonical_information_key(info)
        buckets = self.config.strength_buckets

        def bucket(card: int) -> int:
            return (
                buckets if card == ACE_OF_DENARI else min(buckets - 1, card * buckets // NUM_CARDS)
            )

        return (
            exact[0],  # phase
            exact[2:7],  # public relative position, trick and bids
            tuple(bucket(card) for card in exact[7]),  # current private hand
            tuple((actor, bucket(card)) for actor, card in exact[9]),  # current trick
            len(exact[10]),  # history length, not raw-card identity
            len(exact[11]),  # legal action width
        )

    def _hand_strength(self, cards: tuple[int, ...]) -> float:
        if not cards:
            return 0.0
        total = 0.0
        for card in cards:
            if card == ACE_OF_DENARI:
                total += 1.0 + self.config.ace_strength_bonus
            else:
                rank = self._rank(card)
                total += max(
                    0.0,
                    (rank - self.config.high_card_threshold)
                    / (1.0 - self.config.high_card_threshold),
                )
        return total

    def _rank(self, card: int) -> float:
        # Playing the Ace exposes a later high/low decision.  Treat it as a
        # high candidate here; the ACE_CHOICE branch applies the actual goal.
        if card == ACE_OF_DENARI:
            return 1.05
        bucket = min(
            self.config.strength_buckets - 1,
            card * self.config.strength_buckets // NUM_CARDS,
        )
        return (bucket + 0.5) / self.config.strength_buckets

    def _bid_probabilities(self, info: InformationState) -> tuple[float, ...]:
        desired = min(float(info.hand_size), self._hand_strength(info.visible_cards))
        scores = [-((float(action) - desired) ** 2) for action in info.legal_actions]
        return _softmax(scores, self.config.bid_temperature)

    def _play_probabilities(self, info: InformationState) -> tuple[float, ...]:
        needed = max(0, info.bids[info.player] - info.catches[info.player])
        remaining = max(1, len(info.visible_cards))
        urgency = min(1.0, needed / remaining)
        ranks = [self._rank(card) for card in info.legal_actions]
        if not info.current_trick:
            # Leading: spend strength only in proportion to how urgently a
            # trick is needed; otherwise conserve it by preferring a low card.
            scores = [(2.0 * urgency - 1.0) * rank for rank in ranks]
            return _softmax(scores, self.config.play_temperature)

        opposing = info.current_trick[0][1]
        opposing_rank = self._rank(opposing)
        winners = [rank > opposing_rank for rank in ranks]
        if urgency and any(winners):
            # Win economically: select the smallest available winning card.
            scores = [(-rank if wins else -2.0 - rank) for rank, wins in zip(ranks, winners)]
        elif urgency:
            # No win is available: discard the least valuable card.
            scores = [-rank for rank in ranks]
        else:
            # We have already met the declaration: avoid accidental wins.
            scores = [(-2.0 - rank if wins else -rank) for rank, wins in zip(ranks, winners)]
        return _softmax(scores, self.config.play_temperature)

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        self.calls += 1
        # Computing the signature validates the canonicalization in the hot
        # path but deliberately does not retain it.
        self.feature_key(info)
        self.feature_signatures += 1
        phase = Phase(info.phase)
        if phase == Phase.BID:
            return self._bid_probabilities(info)
        if phase == Phase.ACE_CHOICE:
            need = info.catches[info.player] < info.bids[info.player]
            return (0.05, 0.95) if need else (0.95, 0.05)
        if phase == Phase.PLAY:
            return self._play_probabilities(info)
        raise ValueError("compact blueprint requires a decision state")

    def lookup(self, info: InformationState) -> tuple[float, ...]:
        return self.probabilities(info)

    def stats(self) -> dict[str, object]:
        return {
            "kind": "feature_blueprint",
            "stored_state_rows": 0,
            "calls": self.calls,
            "feature_signatures": self.feature_signatures,
            "config": asdict(self.config),
        }


def save_compact_blueprint(
    game: GameConfig, path: Path, config: CompactBlueprintConfig | None = None
) -> dict[str, object]:
    policy = CompactBlueprintPolicy(config)
    payload = {
        "format": COMPACT_BLUEPRINT_FORMAT,
        "game": game.to_dict(),
        "config": asdict(policy.config),
        "description": "table-free feature blueprint; local ISMCTS is the policy improvement layer",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {"format": COMPACT_BLUEPRINT_FORMAT, "stored_state_rows": 0, "path": str(path.resolve())}


def load_compact_blueprint(path: Path) -> tuple[GameConfig, CompactBlueprintPolicy]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != COMPACT_BLUEPRINT_FORMAT:
        raise ValueError("unsupported compact blueprint format")
    return GameConfig.from_dict(dict(payload["game"])), CompactBlueprintPolicy(
        CompactBlueprintConfig(**dict(payload.get("config", {})))
    )


def audit_abstraction(
    game: GameConfig,
    *,
    samples: int,
    seed: int,
    config: CompactBlueprintConfig | None = None,
) -> dict[str, object]:
    """Measure actual state reduction on random legal round trajectories.

    This is deliberately a *measurement*, not a promise based on the nominal
    number of buckets.  It lets us reject an abstraction whose apparent
    compression comes only from collapsing behaviourally different states.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    from presine.game.state import RoundState

    rng = random.Random(seed)
    policy = CompactBlueprintPolicy(config)
    exact: set[tuple[object, ...]] = set()
    canonical: set[tuple[object, ...]] = set()
    features: set[tuple[object, ...]] = set()
    decisions = 0
    for _ in range(samples):
        state = RoundState(game)
        state.apply_chance(state.sample_chance(rng))
        while not state.is_terminal:
            info = state.information_state()
            exact.add(info.key())
            canonical.add(canonical_information_key(info))
            features.add(policy.feature_key(info))
            decisions += 1
            state.apply_action(rng.choice(info.legal_actions))

    def ratio(size: int) -> float:
        return len(exact) / size if size else 0.0

    return {
        "game": game.to_dict(),
        "samples": samples,
        "decisions": decisions,
        "unique_exact": len(exact),
        "unique_seat_canonical": len(canonical),
        "unique_feature_buckets": len(features),
        "seat_canonical_reduction": ratio(len(canonical)),
        "theoretical_seat_canonical_factor": 2,
        "feature_reduction": ratio(len(features)),
        "config": asdict(policy.config),
        "warning": (
            "seat canonicalization is lossless; feature buckets are an explicit "
            "approximation and must be evaluated with local search enabled"
        ),
    }
