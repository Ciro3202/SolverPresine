from __future__ import annotations

import unittest

from presine.game import GameConfig, RoundState
from presine.game.cards import ACE_HIGH, ACE_OF_DENARI
from presine.game.state import Phase
from presine.policies.blind_round import (
    ClosedBlindRoundPolicy,
    clear_closed_blind_round_cache,
)


def selected_action(policy: ClosedBlindRoundPolicy, state: RoundState) -> int:
    info = state.information_state()
    probabilities = policy.probabilities(info)
    return info.legal_actions[probabilities.index(1.0)]


class ClosedBlindRoundPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_closed_blind_round_cache()
        self.policy = ClosedBlindRoundPolicy()

    def test_first_bid_uses_exact_personal_error_count(self) -> None:
        state = RoundState(GameConfig(players=3, hand_size=1, starting_player=0))
        state.apply_chance(((25,), (14,), (19,)))
        self.assertEqual(selected_action(self.policy, state), 1)

    def test_previous_no_take_signal_supports_take_over_25(self) -> None:
        state = RoundState(GameConfig(players=3, hand_size=1, starting_player=0))
        state.apply_chance(((25,), (35,), (5,)))
        self.assertEqual(selected_action(self.policy, state), 0)
        state.apply_action(0)
        self.assertEqual(state.current_player, 1)
        self.assertEqual(selected_action(self.policy, state), 1)

    def test_policy_does_not_read_the_actual_hidden_own_card(self) -> None:
        actions = []
        for hidden in (26, 35):
            state = RoundState(GameConfig(players=3, hand_size=1, starting_player=0))
            state.apply_chance(((25,), (hidden,), (5,)))
            state.apply_action(selected_action(self.policy, state))
            actions.append(selected_action(self.policy, state))
        self.assertEqual(actions, [1, 1])

    def test_ace_choice_makes_the_personal_bid_correct(self) -> None:
        state = RoundState(GameConfig(players=3, hand_size=1, starting_player=0))
        state.apply_chance(((ACE_OF_DENARI,), (10,), (20,)))
        while state.phase == Phase.BID:
            state.apply_action(selected_action(self.policy, state))
        self.assertEqual(state.phase, Phase.ACE_CHOICE)
        expected = ACE_HIGH if state.bids[0] == 1 else 0
        self.assertEqual(selected_action(self.policy, state), expected)

    def test_complete_round_supported_for_two_to_six_players(self) -> None:
        for players in range(2, 7):
            state = RoundState(GameConfig(players=players, hand_size=1))
            deal = tuple((card,) for card in range(players))
            state.apply_chance(deal)
            while not state.is_terminal:
                state.apply_action(selected_action(self.policy, state))
            self.assertEqual(sum(state.result.catches), 1)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
