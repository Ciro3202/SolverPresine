from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from presine.evaluation.matches import evaluate_round
from presine.game.config import GameConfig
from presine.learning.checkpoint import load_policy
from presine.learning.exact_evaluation import exact_nashconv
from presine.learning.tabular_config import TabularCFRConfig
from presine.learning.tabular_mccfr.tabular_trainer import TabularCFRTrainer
from presine.policies.random import RandomPolicy


def tabular_config(iterations: int = 2) -> TabularCFRConfig:
    return TabularCFRConfig(
        iterations=iterations,
        workers=1,
        checkpoint_every=1,
        log_every=1,
        seed=11,
    )


class TabularCFRTests(unittest.TestCase):
    def test_one_card_iteration_enumerates_every_deal(self) -> None:
        game = GameConfig(hand_size=1)
        trainer = TabularCFRTrainer(game, tabular_config(1))
        report = trainer.train_iteration()
        self.assertEqual(report.deals, 1560)
        self.assertGreater(report.nodes_visited, report.deals)
        self.assertGreater(report.information_states, 0)
        self.assertGreaterEqual(report.nashconv_upper_bound, 0.0)

    def test_average_policy_covers_every_reachable_information_state(self) -> None:
        game = GameConfig(hand_size=1)
        trainer = TabularCFRTrainer(game, tabular_config(2))
        trainer.train_iteration()
        policy = trainer.average_policy()
        result = evaluate_round(game, policy, RandomPolicy(), games=100, seed=19)
        self.assertEqual(result.games, 100)

    def test_checkpoint_resume_matches_uninterrupted_training(self) -> None:
        game = GameConfig(hand_size=1)
        continuous = TabularCFRTrainer(game, tabular_config(2))
        continuous.train_iteration()
        continuous.train_iteration()

        split = TabularCFRTrainer(game, tabular_config(2))
        split.train_iteration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split.save(root)
            resumed = TabularCFRTrainer.restore(root / "training.pt")
            resumed.train_iteration()
            checkpoint_game, policy = load_policy(root / "policy.pt")
            self.assertEqual(checkpoint_game, game)
            self.assertGreater(len(policy.entries), 0)

        self.assertEqual(continuous.iteration, resumed.iteration)
        self.assertEqual(continuous.nodes.keys(), resumed.nodes.keys())
        for key in continuous.nodes:
            self.assertEqual(continuous.nodes[key].regrets, resumed.nodes[key].regrets)
            self.assertEqual(
                continuous.nodes[key].strategy_sum,
                resumed.nodes[key].strategy_sum,
            )

    def test_hand_blocks_cover_each_player_zero_hand_once(self) -> None:
        trainer = TabularCFRTrainer(
            GameConfig(hand_size=2),
            TabularCFRConfig(iterations=1, workers=8),
        )
        flattened = [hand for block in trainer._hand0_blocks() for hand in block]
        self.assertEqual(len(flattened), 780)
        self.assertEqual(len(set(flattened)), 780)

    def test_exact_nashconv_is_finite_for_one_card_profile(self) -> None:
        trainer = TabularCFRTrainer(GameConfig(hand_size=1), tabular_config(1))
        trainer.train_iteration()
        value = exact_nashconv(trainer.game_config, trainer.policy_entries(), max_nodes=1_000_000)
        self.assertGreaterEqual(value, 0.0)
        self.assertLess(value, 10.0)


if __name__ == "__main__":
    unittest.main()
