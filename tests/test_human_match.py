from __future__ import annotations

import re
import unittest

from presine.game.cards import ACE_HIGH, ACE_OF_DENARI, card_name
from presine.game.config import GameConfig
from presine.game.state import Phase, RoundState
from presine.human_match import (
    HumanMatch,
    HumanMatchOptions,
    _ask_human_action,
    parse_round_set,
    parse_rounds,
)


class HumanMatchTests(unittest.TestCase):
    def test_round_selection_supports_interval_and_subset(self) -> None:
        self.assertEqual(parse_rounds(from_round=5, to_round=3), (5, 4, 3))
        self.assertEqual(parse_rounds("5,3"), (5, 3))
        self.assertEqual(parse_round_set("5,4"), {5, 4})

    def test_human_can_enter_a_card_by_name_after_an_invalid_value(self) -> None:
        state = RoundState(GameConfig(players=2, hand_size=2))
        state.apply_chance(((2, 7), (11, 20)))
        state.apply_action(0)
        state.apply_action(0)
        answers = iter(("39", card_name(2)))
        messages: list[str] = []
        action = _ask_human_action(state, lambda _: next(answers), messages.append)
        self.assertEqual(action, 2)
        self.assertTrue(any("non valida" in message for message in messages))

    def test_human_can_choose_ace_high(self) -> None:
        state = RoundState(GameConfig(players=2, hand_size=1))
        state.apply_chance(((ACE_OF_DENARI,), (3,)))
        state.apply_action(0)
        state.apply_action(0)
        self.assertEqual(state.phase, Phase.ACE_CHOICE)
        self.assertEqual(_ask_human_action(state, lambda _: "alto", lambda _: None), ACE_HIGH)

    def test_multiplayer_match_reports_cumulative_errors_for_every_seat(self) -> None:
        class UniformPolicy:
            def probabilities(self, info):
                return tuple(1.0 for _ in info.legal_actions)

        def choose_first(prompt: str) -> str:
            match = re.search(r"\[([^\]]+)\]", prompt)
            return match.group(1).split()[0] if match else "0"

        output: list[str] = []
        match = HumanMatch(
            {2: UniformPolicy()},
            HumanMatchOptions(rounds=(2,), players=3, first_starter="human"),
            input_fn=choose_first,
            output_fn=output.append,
        )
        report = match.play()
        match.close()

        self.assertEqual(report["players"], 3)
        self.assertEqual(len(report["total_errors"]), 3)
        self.assertTrue(any(line.startswith("Presa in corso:") for line in output))
        self.assertTrue(any("Tu (prese rimaste:" in line for line in output))
        self.assertTrue(any(" --> " in line for line in output))
        self.assertTrue(any(line.startswith("Errori questa partita:") for line in output))

    def test_blind_multiplayer_round_uses_single_order_line_before_human_bid(self) -> None:
        class UniformPolicy:
            def probabilities(self, info):
                return tuple(1.0 for _ in info.legal_actions)

        output: list[str] = []
        match = HumanMatch(
            {1: UniformPolicy()},
            HumanMatchOptions(rounds=(1,), players=3, first_starter="model"),
            input_fn=lambda _: "0",
            output_fn=output.append,
        )
        match.play()
        match.close()

        order_lines = [line for line in output if line.startswith("Presa:")]
        self.assertEqual(len(order_lines), 1)
        self.assertIn("Tu", order_lines[0])
        self.assertNotIn("Carte degli altri:", "\n".join(output))
        self.assertFalse(any(line.startswith("Avv") for line in output))


if __name__ == "__main__":
    unittest.main()
