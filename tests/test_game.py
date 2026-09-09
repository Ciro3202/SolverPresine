from __future__ import annotations

import random
import unittest

from presine.evaluation.full_match import RepeatedPolicy
from presine.evaluation.multiplayer import evaluate_multiplayer_full_match
from presine.game.cards import ACE_HIGH, ACE_LOW, ACE_OF_DENARI
from presine.game.config import GameConfig
from presine.game.state import Phase, RoundState
from presine.policies.random import RandomPolicy


def finish_random(config: GameConfig, seed: int) -> RoundState:
    rng = random.Random(seed)
    state = RoundState(config, debug=True)
    state.apply_chance(state.sample_chance(rng))
    while not state.is_terminal:
        state.apply_action(rng.choice(state.legal_actions()))
    return state


class GameTests(unittest.TestCase):
    def test_exact_deal_count(self) -> None:
        self.assertEqual(
            RoundState(GameConfig(hand_size=5)).chance_outcome_count(),
            213_610_453_056,
        )

    def test_bid_restriction(self) -> None:
        state = RoundState(GameConfig(hand_size=5))
        state.apply_chance(((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)))
        state.apply_action(2)
        self.assertNotIn(3, state.legal_actions())

    def test_blind_round_observes_opponent_card(self) -> None:
        state = RoundState(GameConfig(hand_size=1))
        state.apply_chance(((3,), (20,)))
        self.assertEqual(state.information_state().visible_cards, (20,))
        state.apply_action(1)
        self.assertEqual(state.information_state().visible_cards, (3,))

    def test_ace_choice_is_explicit_and_exact(self) -> None:
        high = RoundState(GameConfig(hand_size=2))
        high.apply_chance(((ACE_OF_DENARI, 2), (39, 3)))
        high.apply_action(1)
        high.apply_action(0)
        high.apply_action(ACE_OF_DENARI)
        self.assertEqual(high.phase, Phase.ACE_CHOICE)
        self.assertEqual(high.legal_actions(), (ACE_LOW, ACE_HIGH))
        high.apply_action(ACE_HIGH)
        high.apply_action(39)
        self.assertEqual(high.catches, [1, 0])

    def test_low_ace_keeps_lead_with_lower_numbered_card(self) -> None:
        state = RoundState(GameConfig(players=2, hand_size=2, starting_player=0))
        state.apply_chance(((0, 38), (ACE_OF_DENARI, 24)))
        state.apply_action(1)
        state.apply_action(0)
        state.apply_action(0)
        state.apply_action(ACE_OF_DENARI)
        self.assertEqual(state.phase, Phase.ACE_CHOICE)
        state.apply_action(0)
        self.assertEqual(state.catches, [1, 0])
        self.assertEqual(state.current_player, 0)

    def test_apply_and_undo_restore_exact_state(self) -> None:
        rng = random.Random(9)
        state = RoundState(GameConfig(hand_size=3), debug=True)
        deal = state.sample_chance(rng)
        before = state.state_key()
        token = state.apply_chance(deal)
        state.undo(token)
        self.assertEqual(state.state_key(), before)
        state.apply_chance(deal)
        while not state.is_terminal:
            before = state.state_key()
            action = rng.choice(state.legal_actions())
            token = state.apply_action(action)
            after = state.state_key()
            state.undo(token)
            self.assertEqual(state.state_key(), before)
            state.apply_action(action)
            self.assertEqual(state.state_key(), after)

    def test_fast_undo_matches_regular_undo_in_round_three(self) -> None:
        rng = random.Random(29)
        state = RoundState(GameConfig(hand_size=3), debug=True)
        state.apply_chance(((ACE_OF_DENARI, 2, 3), (4, 5, 39)))
        while not state.is_terminal:
            before = state.state_key()
            action = rng.choice(state.legal_actions())
            regular = state.apply_action(action)
            after = state.state_key()
            state.undo(regular)

            fast = state.apply_action_fast(action)
            self.assertEqual(state.state_key(), after)
            state.undo_fast(fast)
            self.assertEqual(state.state_key(), before)
            state.apply_action(action)

    def test_unchecked_solver_action_matches_regular_action(self) -> None:
        regular = RoundState(GameConfig(hand_size=3))
        regular.apply_chance(((0, 1, 2), (3, 4, 5)))
        fast = RoundState(GameConfig(hand_size=3))
        fast.apply_chance(((0, 1, 2), (3, 4, 5)))
        rng = random.Random(41)
        while not regular.is_terminal:
            action = rng.choice(regular.legal_actions())
            regular.apply_action(action)
            fast.apply_action_unchecked(action)
            self.assertEqual(fast.state_key(), regular.state_key())

    def test_terminal_utility_is_zero_sum(self) -> None:
        state = finish_random(GameConfig(hand_size=5), 12)
        first, second = state.utilities()
        self.assertAlmostEqual(first, -second)
        self.assertLessEqual(abs(first), 1.0)

    def test_supported_player_counts(self) -> None:
        self.assertEqual(GameConfig(players=3, hand_size=5).players, 3)
        self.assertEqual(GameConfig(players=6, hand_size=5).players, 6)

    def test_invalid_player_count_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "2..6"):
            GameConfig(players=1)
        with self.assertRaisesRegex(ValueError, "2..6"):
            GameConfig(players=7)

    def test_four_player_round_and_utility(self) -> None:
        state = finish_random(GameConfig(players=4, hand_size=3), 73)
        self.assertEqual(len(state.result.errors), 4)  # type: ignore[union-attr]
        utilities = state.utilities()
        self.assertEqual(len(utilities), 4)
        self.assertAlmostEqual(sum(utilities), 0.0)

    def test_four_player_last_bid_restriction(self) -> None:
        state = RoundState(GameConfig(players=4, hand_size=5))
        state.apply_chance(
            (
                (0, 1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (10, 11, 12, 13, 14),
                (15, 16, 17, 18, 19),
            )
        )
        state.apply_action(1)
        state.apply_action(1)
        self.assertIn(3, state.legal_actions())
        state.apply_action(1)
        self.assertNotIn(2, state.legal_actions())

    def test_four_player_blind_information_and_turn_order(self) -> None:
        state = RoundState(GameConfig(players=4, hand_size=1, starting_player=2))
        state.apply_chance(((3,), (7,), (20,), (31,)))
        self.assertEqual(state.current_player, 2)
        self.assertEqual(state.information_state().visible_cards, (31, 3, 7))
        for expected_player in (2, 3, 0, 1):
            self.assertEqual(state.current_player, expected_player)
            state.apply_action(state.legal_actions()[0])
        self.assertTrue(state.is_terminal)

    def test_four_player_full_match_uses_split_tie_credit(self) -> None:
        random_policy = RepeatedPolicy(RandomPolicy())
        result = evaluate_multiplayer_full_match(
            random_policy,
            random_policy,
            matches=8,
            seed=91,
        )
        self.assertEqual(
            result.outright_wins + result.tied_firsts + result.losses,
            8,
        )
        self.assertTrue(0.0 <= result.mean_win_credit <= 1.0)
        self.assertEqual(len(result.by_seat_win_credit), 4)

    def test_six_player_full_match_rotates_all_seats(self) -> None:
        random_policy = RepeatedPolicy(RandomPolicy())
        result = evaluate_multiplayer_full_match(
            random_policy,
            random_policy,
            matches=6,
            seed=92,
            players=6,
        )
        self.assertEqual(len(result.by_seat_win_credit), 6)


if __name__ == "__main__":
    unittest.main()
