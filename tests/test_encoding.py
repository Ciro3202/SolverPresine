from __future__ import annotations

import unittest

import numpy as np

from presine.game.config import GameConfig
from presine.game.observation import EventKind, InformationState, PublicEvent
from presine.game.state import RoundState
from presine.learning.deep_cfr.encoding import (
    ACTION_DIM,
    CARD_OFFSET,
    STATE_DIM,
    encode_information_state,
    legal_action_mask,
)
from presine.learning.multiplayer.multiplayer_encoding import (
    encode_information_state as encode_multiplayer_information_state,
)
from presine.learning.multiplayer.multiplayer_encoding import (
    state_dim as multiplayer_state_dim,
)


class EncodingTests(unittest.TestCase):
    def test_multiplayer_encoding_is_invariant_to_clockwise_seat_rotation(self) -> None:
        first = InformationState(
            player=2,
            phase=2,
            hand_size=3,
            starting_player=1,
            leader=3,
            trick_index=1,
            bids=(0, 1, 2, 3),
            catches=(0, 1, 0, 0),
            visible_cards=(4, 8),
            initial_private_observation=(4, 8, 12),
            current_trick=((3, 20), (0, 21)),
            public_history=(
                PublicEvent(EventKind.BID, 1, 1),
                PublicEvent(EventKind.BID, 2, 2),
                PublicEvent(EventKind.TRICK_WINNER, 3, 3),
            ),
            legal_actions=(4, 8),
        )
        # Rotate every absolute seat by +1 while preserving the clockwise
        # relative position and all card/action information.
        second = InformationState(
            player=3,
            phase=2,
            hand_size=3,
            starting_player=2,
            leader=0,
            trick_index=1,
            bids=(3, 0, 1, 2),
            catches=(0, 0, 1, 0),
            visible_cards=(4, 8),
            initial_private_observation=(4, 8, 12),
            current_trick=((0, 20), (1, 21)),
            public_history=(
                PublicEvent(EventKind.BID, 2, 1),
                PublicEvent(EventKind.BID, 3, 2),
                PublicEvent(EventKind.TRICK_WINNER, 0, 0),
            ),
            legal_actions=(4, 8),
        )
        self.assertTrue(
            np.array_equal(
                encode_multiplayer_information_state(first),
                encode_multiplayer_information_state(second),
            )
        )

    def test_multiplayer_blind_encoder_retains_opponent_ownership(self) -> None:
        first = RoundState(GameConfig(players=4, hand_size=1))
        first.apply_chance(((1,), (2,), (3,), (4,)))
        second = RoundState(GameConfig(players=4, hand_size=1))
        second.apply_chance(((1,), (3,), (2,), (4,)))
        self.assertFalse(
            np.array_equal(
                encode_multiplayer_information_state(first.information_state()),
                encode_multiplayer_information_state(second.information_state()),
            )
        )

    def test_multiplayer_encoder_has_a_bounded_dimension_for_three_to_six_players(self) -> None:
        for players in (3, 5, 6):
            state = RoundState(GameConfig(players=players, hand_size=1))
            state.apply_chance(tuple((card,) for card in range(players)))
            encoded = encode_multiplayer_information_state(state.information_state())
            self.assertEqual(encoded.shape, (multiplayer_state_dim(players),))

    def test_exact_cards_that_old_abstraction_merged_stay_distinct(self) -> None:
        first = RoundState(GameConfig(hand_size=2))
        second = RoundState(GameConfig(hand_size=2))
        first.apply_chance(((0, 1), (10, 20)))
        second.apply_chance(((0, 2), (10, 20)))
        encoded_first = encode_information_state(first.information_state())
        encoded_second = encode_information_state(second.information_state())
        self.assertEqual(encoded_first.shape, (STATE_DIM,))
        self.assertFalse(np.array_equal(encoded_first, encoded_second))

    def test_every_exact_card_has_its_own_action(self) -> None:
        state = RoundState(GameConfig(hand_size=2))
        state.apply_chance(((0, 1), (10, 20)))
        state.apply_action(1)
        state.apply_action(0)
        info = state.information_state()
        mask = legal_action_mask(info)
        self.assertEqual(mask.shape, (ACTION_DIM,))
        self.assertTrue(mask[CARD_OFFSET + 0])
        self.assertTrue(mask[CARD_OFFSET + 1])
        self.assertEqual(int(mask.sum()), 2)

    def test_hidden_opponent_hand_does_not_leak(self) -> None:
        first = RoundState(GameConfig(hand_size=3))
        second = RoundState(GameConfig(hand_size=3))
        first.apply_chance(((0, 1, 2), (10, 11, 12)))
        second.apply_chance(((0, 1, 2), (30, 31, 32)))
        self.assertTrue(
            np.array_equal(
                encode_information_state(first.information_state()),
                encode_information_state(second.information_state()),
            )
        )

    def test_ordered_public_history_is_preserved(self) -> None:
        deal = ((0, 4), (10, 14))
        first = RoundState(GameConfig(hand_size=2))
        second = RoundState(GameConfig(hand_size=2))
        for state in (first, second):
            state.apply_chance(deal)
            state.apply_action(0)
            state.apply_action(0)
        first.apply_action(0)
        first.apply_action(10)
        second.apply_action(4)
        second.apply_action(14)
        self.assertFalse(
            np.array_equal(
                encode_information_state(first.information_state()),
                encode_information_state(second.information_state()),
            )
        )


if __name__ == "__main__":
    unittest.main()
