from __future__ import annotations

"""Exact-card training strata for rare or strategically awkward decisions.

These labels never change the information given to a policy and never merge
cards.  They are used only to spend more replay/search budget on difficult
positions and to report where a policy is still weak.
"""

from presine.game.cards import ACE_OF_DENARI
from presine.game.observation import InformationState
from presine.game.state import Phase

MIDDLE_CUPS = frozenset(range(24, 27))
HIGH_CUPS = frozenset(range(27, 30))


def decision_categories(info: InformationState) -> tuple[str, ...]:
    """Return stable, human-readable categories visible to the acting player."""

    phase = Phase(info.phase)
    labels = ["all", phase.name.casefold()]
    private_cards = set(info.initial_private_observation)
    remaining_cards = set(info.visible_cards)

    if private_cards & HIGH_CUPS:
        labels.append("high_cups")
    if private_cards & MIDDLE_CUPS:
        labels.append("middle_cups")
    if private_cards & (HIGH_CUPS | MIDDLE_CUPS):
        labels.append("difficult_cups")
    if ACE_OF_DENARI in private_cards:
        labels.append("ace_dealt")
    if phase == Phase.ACE_CHOICE:
        labels.append("ace_choice")
    if phase == Phase.PLAY and ACE_OF_DENARI in remaining_cards:
        labels.append("ace_still_in_hand")
    if phase == Phase.PLAY and info.trick_index == info.hand_size - 1:
        labels.append("last_trick")
    if len(info.legal_actions) > 1:
        labels.append("real_choice")
    else:
        labels.append("forced_action")
    return tuple(labels)


def focus_multiplier(info: InformationState, configured: float) -> float:
    """Prioritise awkward natural positions without changing deal sampling."""

    categories = decision_categories(info)
    return configured if "difficult_cups" in categories else 1.0
