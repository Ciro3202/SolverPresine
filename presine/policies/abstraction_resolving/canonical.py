from __future__ import annotations

"""Lossless symmetries and bounded features for heads-up policies.

Cards deliberately keep their identities: Presine ranks every card and the
Ace of Denari has a special rule, so there is no safe suit permutation.  The
two *seats*, on the other hand, use identical rules.  A policy acting as
player zero and the same position viewed by player one are therefore the same
information set after a relative-seat relabelling.
"""

from presine.game.observation import InformationState


def relative_seat(player: int, observer: int, players: int = 2) -> int:
    """Return ``player`` in the acting observer's coordinate system."""
    if players != 2:
        raise ValueError("relative-seat canonicalization currently supports heads-up only")
    return (player - observer) % players


def canonical_information_key(info: InformationState) -> tuple[object, ...]:
    """Losslessly canonicalize a heads-up observation across seat exchange.

    The entire public history remains in the key; only absolute player labels
    are replaced by their relative role.  Unlike card bucketing, this reduction
    preserves all information available to the acting player.
    """
    if len(info.bids) != 2 or len(info.catches) != 2:
        raise ValueError("canonical_information_key requires a heads-up observation")
    observer = info.player
    order = tuple((observer + offset) % 2 for offset in range(2))
    return (
        info.phase,
        info.hand_size,
        relative_seat(info.starting_player, observer),
        relative_seat(info.leader, observer),
        info.trick_index,
        tuple(info.bids[index] for index in order),
        tuple(info.catches[index] for index in order),
        tuple(sorted(info.visible_cards)),
        tuple(sorted(info.initial_private_observation)),
        tuple((relative_seat(actor, observer), card) for actor, card in info.current_trick),
        tuple(
            (int(event.kind), relative_seat(event.actor, observer), event.value)
            for event in info.public_history
        ),
        tuple(sorted(info.legal_actions)),
    )
