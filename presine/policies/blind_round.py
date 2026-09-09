from __future__ import annotations

"""Closed deterministic policy for the one-card blind round.

The policy minimizes only the acting player's expected personal error. It
enumerates every still-possible own card and filters that posterior with all
previous bids. A previous bid is interpreted through the same direct
personal-error rule from the cards that bidder could see.
"""

from functools import lru_cache

from presine.game.cards import ACE_HIGH, ACE_LOW, ACE_OF_DENARI, NUM_CARDS, card_strength
from presine.game.observation import InformationState
from presine.game.state import Phase


def _position(player: int, starter: int, players: int) -> int:
    return (player - starter) % players


def _visible_from_deal(deal: tuple[int, ...], player: int) -> tuple[int, ...]:
    players = len(deal)
    return tuple(deal[(player + offset) % players] for offset in range(1, players))


def _deal_from_view(player: int, visible: tuple[int, ...], own: int) -> tuple[int, ...]:
    players = len(visible) + 1
    deal = [-1] * players
    deal[player] = own
    for offset, card in enumerate(visible, start=1):
        deal[(player + offset) % players] = card
    return tuple(deal)


@lru_cache(maxsize=100_000)
def _no_visible_ace_bid(visible: tuple[int, ...]) -> int:
    """Exact personal-error bid when the Ace is not visible."""

    excluded = set(visible)
    wins = losses = 0
    highest = max(visible)
    for own in range(NUM_CARDS):
        if own in excluded or own == ACE_OF_DENARI:
            # The hidden Ace makes either bid correct: low for 0, high for 1.
            continue
        if own > highest:
            wins += 1
        else:
            losses += 1
    return 1 if wins > losses else 0


@lru_cache(maxsize=100_000)
def _direct_bid(visible: tuple[int, ...]) -> int:
    """Direct closed bid from visible cards, including a visible Ace."""

    if ACE_OF_DENARI not in visible:
        return _no_visible_ace_bid(visible)

    actor = 0
    players = len(visible) + 1
    excluded = set(visible)
    errors = [0, 0]
    for own in range(NUM_CARDS):
        if own in excluded or own == ACE_OF_DENARI:
            continue
        deal = _deal_from_view(actor, visible, own)
        ace_owner = deal.index(ACE_OF_DENARI)
        # The Ace owner does not see their own Ace; their direct declaration
        # therefore uses the ordinary no-visible-Ace calculation.
        ace_bid = _no_visible_ace_bid(_visible_from_deal(deal, ace_owner))
        ace_high = ace_bid == 1
        winner = max(
            range(players),
            key=lambda seat: card_strength(
                deal[seat], ace_high if deal[seat] == ACE_OF_DENARI else None
            ),
        )
        for action in (0, 1):
            errors[action] += int(action != int(winner == actor))
    return min((0, 1), key=lambda action: (errors[action], action))


def _previous_bid_matches(
    deal: tuple[int, ...],
    bids: tuple[int, ...],
    *,
    starter: int,
    actor_position: int,
) -> bool:
    players = len(deal)
    for offset in range(actor_position):
        seat = (starter + offset) % players
        if _direct_bid(_visible_from_deal(deal, seat)) != bids[seat]:
            return False
    return True


def _winner_for_candidate(
    deal: tuple[int, ...],
    bids: tuple[int, ...],
    *,
    starter: int,
    actor_position: int,
) -> int:
    try:
        ace_owner = deal.index(ACE_OF_DENARI)
    except ValueError:
        return max(range(len(deal)), key=lambda seat: deal[seat])

    ace_position = _position(ace_owner, starter, len(deal))
    ace_bid = (
        bids[ace_owner]
        if ace_position < actor_position
        else _direct_bid(_visible_from_deal(deal, ace_owner))
    )
    ace_high = ace_bid == 1
    return max(
        range(len(deal)),
        key=lambda seat: card_strength(
            deal[seat], ace_high if deal[seat] == ACE_OF_DENARI else None
        ),
    )


@lru_cache(maxsize=250_000)
def _closed_bid(
    players: int,
    starter: int,
    actor: int,
    visible: tuple[int, ...],
    bids: tuple[int, ...],
) -> int:
    if len(visible) != players - 1 or len(bids) != players:
        raise ValueError("invalid blind-round closed-policy state")
    actor_position = _position(actor, starter, players)
    excluded = set(visible)
    candidates = [card for card in range(NUM_CARDS) if card not in excluded]
    compatible = [
        own
        for own in candidates
        if _previous_bid_matches(
            _deal_from_view(actor, visible, own),
            bids,
            starter=starter,
            actor_position=actor_position,
        )
    ]
    # An external player can produce an off-policy history. Retain the uniform
    # legal prior instead of crashing or inventing a hidden card.
    if not compatible:
        compatible = candidates

    errors = [0, 0]
    for own in compatible:
        if own == ACE_OF_DENARI:
            continue
        deal = _deal_from_view(actor, visible, own)
        winner = _winner_for_candidate(
            deal,
            bids,
            starter=starter,
            actor_position=actor_position,
        )
        for action in (0, 1):
            errors[action] += int(action != int(winner == actor))
    return min((0, 1), key=lambda action: (errors[action], action))


class ClosedBlindRoundPolicy:
    """R1 policy minimizing expected personal errors with closed signals."""

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        if info.hand_size != 1:
            raise ValueError("the closed blind-round policy supports only R1")
        phase = Phase(info.phase)
        if phase == Phase.BID:
            action = _closed_bid(
                len(info.bids),
                info.starting_player,
                info.player,
                tuple(info.visible_cards),
                tuple(info.bids),
            )
        elif phase == Phase.ACE_CHOICE:
            action = ACE_HIGH if info.bids[info.player] == 1 else ACE_LOW
        else:
            raise ValueError("R1 has no normal card-play decisions")
        return tuple(1.0 if candidate == action else 0.0 for candidate in info.legal_actions)


class ClosedR1PolicyProvider:
    """Override only R1 while delegating R2--R5 to an existing provider."""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.closed_r1 = ClosedBlindRoundPolicy()

    def for_hand_size(self, hand_size: int) -> object:
        if hand_size == 1:
            return self.closed_r1
        provider = self.delegate.for_hand_size
        return provider(hand_size)


class ClosedUnlessAceVisibleR1Policy:
    """Use the closed rule normally, but let a trained R1 head handle visible 30s."""

    def __init__(self, neural: object) -> None:
        self.neural = neural
        self.closed = ClosedBlindRoundPolicy()

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        if ACE_OF_DENARI in info.visible_cards:
            return self.neural.probabilities(info)  # type: ignore[attr-defined]
        return self.closed.probabilities(info)


def clear_closed_blind_round_cache() -> None:
    _closed_bid.cache_clear()
    _direct_bid.cache_clear()
    _no_visible_ace_bid.cache_clear()
