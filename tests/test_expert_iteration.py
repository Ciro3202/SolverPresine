from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.learning.expert_iteration import ExpertIterationConfig, ExpertIterationTrainer
from presine.policies.abstraction_resolving.abstract import load_blueprint
from presine.policies.linear import (
    LinearBlueprintPolicy,
    LinearFeatureConfig,
    load_linear_blueprint,
    save_linear_blueprint,
)
from presine.policies.multiplayer_bundle import load_multiplayer_bundle, save_multiplayer_bundle
from presine.search.config import BeliefConfig, ISMCTSConfig, SearchConfig


class ExpertIterationTests(unittest.TestCase):
    @staticmethod
    def _search() -> SearchConfig:
        return SearchConfig(
            workers=1,
            belief=BeliefConfig(particles=4, mcmc_steps=0, mcmc_burn_in=0),
            ismcts=ISMCTSConfig(simulations=6, rollout_depth=32),
        )

    def test_linear_policy_round_trips_without_storing_information_states(self) -> None:
        game = GameConfig(hand_size=3)
        policy = LinearBlueprintPolicy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.json"
            save_linear_blueprint(game, policy, path)
            loaded_game, loaded = load_linear_blueprint(path)
            generic_game, generic = load_blueprint(path)
        self.assertEqual(loaded_game, game)
        self.assertEqual(generic_game, game)
        self.assertEqual(loaded.weights.size, policy.weights.size)
        self.assertEqual(generic.stats()["stored_state_rows"], 0)

    def test_zero_residual_uses_the_safe_compact_base_instead_of_uniform(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((1, 12, 30), (4, 5, 6)))
        probabilities = LinearBlueprintPolicy().probabilities(state.information_state())
        self.assertNotEqual(len(set(probabilities)), 1)

    def test_factorized_tiles_round_trip_without_storing_states(self) -> None:
        game = GameConfig(hand_size=3)
        policy = LinearBlueprintPolicy(LinearFeatureConfig(tile_buckets=256))
        state = RoundState(game)
        state.apply_chance(((1, 12, 30), (4, 5, 6)))
        indices = policy.tile_indices(state.information_state())
        self.assertEqual(indices.shape[1], 5)
        self.assertEqual(policy.stats()["stored_state_rows"], 0)
        self.assertEqual(policy.stats()["tile_parameters"], 256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiles.json"
            save_linear_blueprint(game, policy, path)
            loaded_game, loaded = load_linear_blueprint(path)
            generic_game, generic = load_blueprint(path)
        self.assertEqual(loaded_game, game)
        self.assertEqual(generic_game, game)
        self.assertEqual(loaded.weights.size, policy.weights.size)
        self.assertEqual(generic.stats()["tile_parameters"], 256)
        self.assertEqual(loaded.tile_indices(state.information_state()).tolist(), indices.tolist())

    def test_tiny_expert_iteration_collects_bounded_examples_and_validates_pool(self) -> None:
        trainer = ExpertIterationTrainer(
            GameConfig(hand_size=3),
            self._search(),
            ExpertIterationConfig(
                iterations=1,
                games_per_iteration=2,
                epochs_per_iteration=1,
                memory_examples=32,
                seed=17,
            ),
        )
        report = trainer.train()
        validation = trainer.validation(games=2, seed=19)
        self.assertGreater(report["iterations"][0]["memory_examples"], 0)
        self.assertLessEqual(report["iterations"][0]["memory_examples"], 32)
        self.assertEqual(set(validation), {"standalone", "resolved", "resolved_stats"})
        self.assertEqual(set(validation["standalone"]), {"heuristic", "random", "snapshot"})
        self.assertGreater(validation["resolved_stats"]["local_resolutions"], 0)
        self.assertEqual(trainer.policy.stats()["stored_state_rows"], 0)
        self.assertEqual(trainer.policy.stats()["tile_parameters"], 32_768)
        self.assertGreater(
            np.count_nonzero(trainer.policy.weights[trainer.policy.base_parameter_count :]),
            0,
        )

    def test_multiplayer_expert_iteration_uses_relative_seats(self) -> None:
        trainer = ExpertIterationTrainer(
            GameConfig(players=3, hand_size=2),
            self._search(),
            ExpertIterationConfig(
                iterations=1,
                games_per_iteration=1,
                epochs_per_iteration=1,
                memory_examples=16,
                tile_buckets=64,
                seed=29,
            ),
        )
        report = trainer.train()
        self.assertGreater(report["training_statistics"]["information_states_visited_total"], 0)
        self.assertFalse(trainer.policy.config.compact_prior)
        validation = trainer.validation(games=1, seed=31)
        self.assertIn("mean_error_margin", validation["standalone"]["heuristic"])

    def test_multiplayer_bundle_round_trips_five_heads(self) -> None:
        game = GameConfig(players=3, hand_size=5)
        heads = {
            size: LinearBlueprintPolicy(
                LinearFeatureConfig(players=3, compact_prior=False, tile_buckets=64)
            )
            for size in range(1, 6)
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            save_multiplayer_bundle(game, heads, path)
            loaded_game, bundle = load_multiplayer_bundle(path)
        self.assertEqual(loaded_game.hand_size, 5)
        self.assertEqual(sorted(bundle.heads), [1, 2, 3, 4, 5])
        self.assertEqual(bundle.for_hand_size(1).config.players, 3)


if __name__ == "__main__":
    unittest.main()
