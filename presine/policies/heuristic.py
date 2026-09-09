from __future__ import annotations

from presine.game.cards import ACE_HIGH, ACE_LOW, ACE_OF_DENARI
from presine.game.observation import InformationState
from presine.game.state import Phase
from presine.policies.blind_round import ClosedBlindRoundPolicy

_CLOSED_R1 = ClosedBlindRoundPolicy()


class HeuristicPolicy:
    """Deterministic transparent baseline, not a training component."""

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        action = self._action(info)
        return tuple(1.0 if candidate == action else 0.0 for candidate in info.legal_actions)

    def _action(self, info: InformationState) -> int:
        phase = Phase(info.phase)
        if phase == Phase.BID:
            if info.hand_size == 1:
                probabilities = _CLOSED_R1.probabilities(info)
                return info.legal_actions[probabilities.index(1.0)]
            else:
                estimate = sum(card >= 20 or card == ACE_OF_DENARI for card in info.visible_cards)
            legal = info.legal_actions
            return min(legal, key=lambda action: (abs(action - estimate), action))
        if phase == Phase.ACE_CHOICE:
            target = info.bids[info.player]
            return ACE_HIGH if info.catches[info.player] < target else ACE_LOW
        cards = sorted(info.legal_actions)
        target = info.bids[info.player]
        need_trick = info.catches[info.player] < target
        if not info.current_trick:
            return cards[-1] if need_trick else cards[0]
        opposing_card = info.current_trick[0][1]
        winning = [card for card in cards if card > opposing_card or card == ACE_OF_DENARI]
        if need_trick and winning:
            return min(winning)
        return cards[0]
