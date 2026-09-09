from __future__ import annotations

import math
import pickle
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.learning.checkpoint_deployment.sampled_checkpoint import (
    _load_pickle,
    load_sampled_policy,
    load_sampled_training_payload,
    load_sampled_training_policy,
)
from presine.learning.mccfr_config import MCCFRConfig
from presine.learning.tabular_config import TabularCFRConfig
from presine.learning.tabular_mccfr.mccfr_parallel import (
    _shard_seed,
    create_mccfr_trainer,
    restore_mccfr_trainer,
)
from presine.learning.tabular_mccfr.mccfr_trainer import MCCFRNode, MCCFRTrainer
from presine.learning.tabular_mccfr.tabular_trainer import TabularCFRTrainer
from presine.policies.tabular import TabularPolicy


class MCCFRTests(unittest.TestCase):
    def config(self, **kwargs) -> MCCFRConfig:
        values = dict(
            iterations=1,
            traversals_per_player=1,
            checkpoint_every=1,
            log_every=1,
            seed=5,
        )
        values.update(kwargs)
        return MCCFRConfig(**values)

    @staticmethod
    def known_info():
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (3, 4, 5)))
        return state.information_state()

    def test_hand_size_guard_and_sparse_training(self) -> None:
        with self.assertRaises(ValueError):
            MCCFRTrainer(GameConfig(players=3, hand_size=2), self.config())
        round_two = MCCFRTrainer(GameConfig(hand_size=2), self.config())
        self.assertGreater(round_two.train_iteration().nodes_visited, 0)
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        report = trainer.train_iteration()
        self.assertEqual(report.traversals, 4)
        self.assertEqual(report.traversals_by_player, (2, 2))
        self.assertGreater(report.nodes_visited, 0)
        self.assertGreater(len(trainer.nodes), 0)
        self.assertNotIn("nashconv_upper_bound", report.to_dict())
        self.assertIn("not a NashConv", report.to_dict()["warning"])

    def test_full_tree_tabular_rejects_unsupported_multiplayer(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two players"):
            TabularCFRTrainer(
                GameConfig(players=3, hand_size=1),
                TabularCFRConfig(iterations=1),
            )

    def test_mmap_cfr_plus_checkpoint_restores_without_numeric_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(
                algorithm="external_sampling_mccfr_plus",
                storage_backend="mmap",
                numeric_dtype="float32",
            )
            trainer = MCCFRTrainer(
                GameConfig(hand_size=2),
                config,
                storage_directory=root / ".internal" / "mmap-table",
            )
            trainer.train_iteration()
            expected = trainer.policy_entries()
            trainer.save_training(root)
            payload = load_sampled_training_payload(root / "training.pt")
            self.assertIn("persistent_store", payload)
            restored = MCCFRTrainer.restore(root / "training.pt")
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(len(restored.nodes), len(trainer.nodes))
            trainer.close()
            restored.close()
            policy_game, policy = load_sampled_training_policy(root / "training.pt")
            self.assertEqual(policy_game, GameConfig(hand_size=2))
            self.assertEqual(len(policy.entries), len(expected))
            for key, entry in list(expected.items())[:20]:
                self.assertEqual(policy.entries.get(key), entry)
            policy.entries.close()

    def test_ensemble_training_checkpoint_has_zero_copy_policy_view(self) -> None:
        game = GameConfig(hand_size=2)
        config = self.config(
            workers=2,
            replicas=2,
            export_policy_on_finish=False,
        )
        trainer = create_mccfr_trainer(game, config)
        trainer.train_iteration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.save_training(root)
            trainer.close()
            policy_game, policy = load_sampled_training_policy(root / "training.pt")
            self.assertEqual(policy_game, game)
            self.assertGreater(len(policy.entries), 0)
            key = next(iter(policy.entries))
            actions, probabilities = policy.entries[key]
            self.assertEqual(tuple(key[-1]), actions)
            self.assertAlmostEqual(sum(probabilities), 1.0)
            policy.entries.close()

    def test_policy_probabilities_are_legal_and_normalized(self) -> None:
        info = self.known_info()
        probabilities = (0.1, 0.2, 0.3, 0.4)
        policy = TabularPolicy({info.key(): (info.legal_actions, probabilities)})
        result = policy.probabilities(info)
        self.assertAlmostEqual(sum(result), 1.0)
        by_action = policy.probability_by_action(info)
        self.assertEqual(set(by_action), set(info.legal_actions))
        self.assertEqual(by_action.get(39, 0.0), 0.0)

    def test_solver_information_is_exactly_the_public_information_key(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (3, 4, 5)))
        info = state.information_state()
        player, key, actions = state.solver_information()
        self.assertEqual(player, info.player)
        self.assertEqual(key, info.key())
        self.assertEqual(actions, info.legal_actions)

    def test_solver_undo_restores_exact_state_through_a_complete_round(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (3, 4, 5)))
        while not state.is_terminal:
            actions = state.legal_actions()
            before = state.state_key()
            for action in actions:
                token = state.apply_action_solver(action)
                state.undo_solver(token)
                self.assertEqual(state.state_key(), before)
            state.apply_action_solver(actions[0])

    def test_unseen_information_state_is_exactly_uniform(self) -> None:
        info = self.known_info()
        policy = TabularPolicy({})
        expected = (1.0 / len(info.legal_actions),) * len(info.legal_actions)
        self.assertEqual(policy.probabilities(info), expected)
        stats = policy.stats()
        self.assertEqual(stats["uniform_fallbacks"], 1)
        self.assertEqual(stats["fallback_rate"], 1.0)
        self.assertIn(f"seat_{info.player}/phase_{info.phase}", stats["by_seat_phase"])

    def test_simple_average_updates_sampled_opponent_nodes_only(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (3, 4, 5)))
        trainer._cfr(state, traverser=0, iteration=1)
        player0 = [node for key, node in trainer.nodes.items() if int(key[0]) == 0]
        player1 = [node for key, node in trainer.nodes.items() if int(key[0]) == 1]
        self.assertTrue(player0 and player1)
        self.assertTrue(all(sum(node.strategy_sum) == 0.0 for node in player0))
        self.assertTrue(all(math.isclose(sum(node.strategy_sum), 1.0) for node in player1))

    def test_average_visit_is_not_multiplied_by_either_reach_again(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (3, 4, 5)))
        trainer._cfr(state, traverser=0, iteration=1)
        deep_opponent_nodes = [
            node for key, node in trainer.nodes.items() if int(key[0]) == 1 and len(key[11]) >= 2
        ]
        self.assertTrue(deep_opponent_nodes)
        # At these nodes traverser reach is below one and sampled opponent own
        # reach is generally below one.  Simple on-policy averaging nevertheless
        # contributes one full strategy vector per actual visit.
        self.assertTrue(
            all(math.isclose(sum(node.strategy_sum), 1.0) for node in deep_opponent_nodes)
        )

    def test_lazy_dcfr_matches_dense_reference_across_missed_visits(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        node = MCCFRNode((0, 1), [4.0, -6.0], [0.0, 0.0], 1)
        expected = list(node.regrets)
        for iteration in range(2, 8):
            positive = iteration**1.5 / (iteration**1.5 + 1.0)
            negative = iteration**0.0 / (iteration**0.0 + 1.0)
            expected = [value * (positive if value >= 0.0 else negative) for value in expected]
        trainer._discount_node(node, 7)
        for actual, dense in zip(node.regrets, expected):
            self.assertAlmostEqual(actual, dense, places=14)
        self.assertEqual(node.last_discount_iteration, 7)

    def test_checkpoint_materializes_all_lazy_discount_timestamps(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        info = self.known_info()
        trainer.nodes[info.key()] = MCCFRNode(
            info.legal_actions,
            [3.0, -2.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
            1,
        )
        trainer.iteration = 9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.save_training(root)
            self.assertEqual((root / "training.pt").read_bytes()[:2], b"\x1f\x8b")
            payload = load_sampled_training_payload(root / "training.pt")
            self.assertEqual(payload["codec"], "gzip")
        stored = payload["nodes"][info.key()]
        self.assertEqual(stored[3], 9)
        self.assertEqual(trainer.nodes[info.key()].last_discount_iteration, 9)

    def test_vanilla_and_dcfr_are_distinct_and_neither_clips_negative_regret(self) -> None:
        vanilla = MCCFRTrainer(
            GameConfig(hand_size=3),
            self.config(algorithm="external_sampling_mccfr"),
        )
        dcfr = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        vanilla_node = MCCFRNode((0, 1), [2.0, -4.0], [0.0, 0.0], 1)
        dcfr_node = MCCFRNode((0, 1), [2.0, -4.0], [0.0, 0.0], 1)
        vanilla._discount_node(vanilla_node, 4)
        dcfr._discount_node(dcfr_node, 4)
        self.assertEqual(vanilla_node.regrets, [2.0, -4.0])
        self.assertLess(dcfr_node.regrets[0], 2.0)
        self.assertLess(dcfr_node.regrets[1], 0.0)
        self.assertNotEqual(dcfr_node.regrets, vanilla_node.regrets)

    def test_chance_is_sampled_fresh_without_a_fixed_pool(self) -> None:
        trainer = MCCFRTrainer(
            GameConfig(hand_size=3),
            self.config(paired_starting_player=False),
        )
        original = RoundState.sample_chance
        with patch.object(
            RoundState,
            "sample_chance",
            autospec=True,
            side_effect=lambda state, rng: original(state, rng),
        ) as sampled:
            report = trainer.train_iteration()
        self.assertEqual(sampled.call_count, 4)
        self.assertEqual(report.chance_samples, 4)
        self.assertFalse(hasattr(trainer, "chance_pool"))

    def test_starting_player_and_traverser_counts_are_exactly_balanced(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config(traversals_per_player=3))
        report = trainer.train_iteration()
        self.assertEqual(report.traversals_by_player, (6, 6))
        self.assertEqual(report.traversals_by_starting_player, (6, 6))

    def test_diagnostic_cadence_does_not_change_training_state(self) -> None:
        game = GameConfig(hand_size=3)
        frequent = MCCFRTrainer(
            game,
            self.config(iterations=20, log_every=1, checkpoint_every=100),
        )
        sparse = MCCFRTrainer(
            game,
            self.config(iterations=20, log_every=100, checkpoint_every=100),
        )
        for _ in range(20):
            frequent.train_iteration()
            sparse.train_iteration()
        self.assertEqual(frequent.rng.getstate(), sparse.rng.getstate())
        self.assertEqual(frequent.nodes.keys(), sparse.nodes.keys())
        for key in frequent.nodes:
            self.assertEqual(frequent.nodes[key], sparse.nodes[key])
        self.assertEqual(frequent.diagnostics, sparse.diagnostics)

    def test_checkpoint_resume_matches_continuous_run(self) -> None:
        game = GameConfig(hand_size=3)
        config = self.config(iterations=2, traversals_per_player=2)
        continuous = MCCFRTrainer(game, config)
        continuous.train_iteration()
        continuous.train_iteration()

        split = MCCFRTrainer(game, config)
        split.train_iteration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split.save_training(root)
            resumed = MCCFRTrainer.restore(root / "training.pt")
            resumed.train_iteration()

        self.assertEqual(continuous.iteration, resumed.iteration)
        self.assertEqual(continuous.nodes_visited, resumed.nodes_visited)
        self.assertEqual(continuous.rng.getstate(), resumed.rng.getstate())
        self.assertEqual(continuous.diagnostics, resumed.diagnostics)
        self.assertEqual(continuous.nodes.keys(), resumed.nodes.keys())
        for key in continuous.nodes:
            self.assertEqual(continuous.nodes[key], resumed.nodes[key])

    def test_uncompressed_sampled_checkpoint_remains_loadable(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        trainer.train_iteration()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.save_training(root)
            payload = _load_pickle(root / "training.pt")
            payload.pop("codec", None)
            with (root / "training.pt").open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            restored = MCCFRTrainer.restore(root / "training.pt")
        self.assertEqual(restored.iteration, trainer.iteration)
        self.assertEqual(restored.nodes.keys(), trainer.nodes.keys())

    def test_legacy_sampled_nodes_migrate_to_compact_store(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        trainer.train_iteration()
        legacy_nodes = {
            key: (
                node.actions,
                tuple(node.regrets),
                tuple(node.strategy_sum),
                node.last_discount_iteration,
            )
            for key, node in trainer.nodes.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-sampled.pt"
            payload = {
                "format": 2,
                "kind": "external-sampling-tabular-training",
                "game": trainer.game_config.to_dict(),
                "mccfr": trainer.training_config.to_dict(),
                "iteration": trainer.iteration,
                "nodes_visited": trainer.nodes_visited,
                "rng_state": trainer.rng.getstate(),
                "diagnostics": trainer.diagnostics,
                "nodes": legacy_nodes,
            }
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            restored = MCCFRTrainer.restore(path)
        self.assertEqual(len(restored.nodes), len(trainer.nodes))
        self.assertEqual(set(restored.nodes.keys()), set(trainer.nodes.keys()))

    def test_final_checkpoint_is_not_serialized_twice(self) -> None:
        trainer = MCCFRTrainer(
            GameConfig(hand_size=3),
            self.config(iterations=2, checkpoint_every=1, log_every=1),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(trainer, "save_training", wraps=trainer.save_training) as save_training,
        ):
            root = Path(directory)
            trainer.train(root)
            self.assertEqual(save_training.call_count, 2)
            checkpoint_metrics = (
                (root / ".internal" / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
            )
        self.assertEqual(len(checkpoint_metrics), 2)

    def test_two_starting_player_workers_match_sequential_shards(self) -> None:
        game = GameConfig(hand_size=3)
        config = self.config(
            iterations=1,
            workers=2,
            checkpoint_every=10,
            log_every=10,
        )
        coordinator = random.Random(config.seed)
        sampler = RoundState(game)
        tape = tuple((sampler.sample_chance(coordinator),) for _ in (0, 1))
        references = []
        for starting_player in (0, 1):
            shard_config = replace(
                config,
                workers=1,
                alternate_starting_player=False,
                paired_starting_player=False,
            )
            shard = MCCFRTrainer(replace(game, starting_player=starting_player), shard_config)
            shard.rng.seed(_shard_seed(config.seed, starting_player))
            shard.train_shard_iteration(
                1,
                tape,
                collect_light_diagnostics=True,
                collect_regret_diagnostics=True,
            )
            references.append(shard)

        parallel = create_mccfr_trainer(game, config)
        report = parallel.train_iteration()
        self.assertEqual(report.traversals_by_player, (2, 2))
        self.assertEqual(report.traversals_by_starting_player, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parallel.save_training(root)
            manifest = load_sampled_training_payload(root / "training.pt")
            shards = [
                load_sampled_training_payload(root / relative) for relative in manifest["shards"]
            ]
        parallel.close()

        for starting_player, (actual, expected) in enumerate(zip(shards, references)):
            self.assertTrue(all(int(key[3]) == starting_player for key in actual["nodes"]))
            self.assertEqual(actual["rng_state"], expected.rng.getstate())
            self.assertEqual(actual["nodes"].keys(), expected.nodes.keys())
            for key, values in actual["nodes"].items():
                node = expected.nodes[key]
                self.assertEqual(
                    values,
                    (
                        node.actions,
                        tuple(node.regrets),
                        tuple(node.strategy_sum),
                        node.last_discount_iteration,
                    ),
                )

    def test_two_worker_checkpoint_resume_and_policy_round_trip(self) -> None:
        game = GameConfig(hand_size=3)
        config = self.config(
            iterations=2,
            workers=2,
            checkpoint_every=2,
            log_every=2,
        )
        trainer = create_mccfr_trainer(game, config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.train(root)
            manifest = load_sampled_training_payload(root / "training.pt")
            self.assertEqual(manifest["iteration"], 2)
            self.assertEqual(len(manifest["shards"]), 2)
            restored = restore_mccfr_trainer(root / "training.pt")
            restored.training_config = replace(config, iterations=3)
            resumed_report = restored.train_iteration()
            self.assertEqual(resumed_report.iteration, 3)
            restored.close()
            policy_game, policy = load_sampled_policy(root / "policy.pt")
        self.assertEqual(policy_game, game)
        self.assertGreater(len(policy.entries), 0)
        self.assertEqual({int(key[3]) for key in policy.entries}, {0, 1})

    def test_two_worker_resume_matches_continuous_shards_exactly(self) -> None:
        game = GameConfig(hand_size=3)
        config = self.config(
            iterations=3,
            workers=2,
            checkpoint_every=2,
            log_every=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous_root = root / "continuous"
            split_root = root / "split"

            continuous = create_mccfr_trainer(game, config)
            for _ in range(3):
                continuous.train_iteration()
            continuous.save_training(continuous_root)
            continuous.close()

            split = create_mccfr_trainer(game, config)
            split.train_iteration()
            split.train_iteration()
            split.save_training(split_root)
            split.close()
            resumed = restore_mccfr_trainer(split_root / "training.pt")
            resumed.train_iteration()
            resumed.save_training(split_root)
            resumed.close()

            continuous_manifest = load_sampled_training_payload(continuous_root / "training.pt")
            resumed_manifest = load_sampled_training_payload(split_root / "training.pt")
            self.assertEqual(continuous_manifest["rng_state"], resumed_manifest["rng_state"])
            self.assertEqual(continuous_manifest["diagnostics"], resumed_manifest["diagnostics"])
            for left_relative, right_relative in zip(
                continuous_manifest["shards"], resumed_manifest["shards"]
            ):
                left = load_sampled_training_payload(continuous_root / left_relative)
                right = load_sampled_training_payload(split_root / right_relative)
                self.assertEqual(left["rng_state"], right["rng_state"])
                self.assertEqual(left["nodes"], right["nodes"])

    def test_legacy_checkpoint_is_rejected_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            with path.open("wb") as handle:
                pickle.dump({"format": 1, "kind": "mccfr-plus-training"}, handle)
            with self.assertRaisesRegex(ValueError, "incompatible legacy MCCFR"):
                load_sampled_training_payload(path)

    def test_policy_checkpoint_is_pure_tabular_and_round_trips(self) -> None:
        trainer = MCCFRTrainer(GameConfig(hand_size=3), self.config())
        self.assertFalse(hasattr(trainer, "network"))
        self.assertFalse(hasattr(trainer, "networks"))
        info = self.known_info()
        actions = info.legal_actions
        trainer.iteration = 1
        trainer.nodes[info.key()] = MCCFRNode(
            actions,
            [2.0, -1.0, 0.0, 0.5],
            [1.0, 2.0, 3.0, 4.0],
            1,
        )
        expected = trainer.average_policy().probabilities(info)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            trainer.save_policy(path.parent)
            payload = _load_pickle(path)
            game, restored = load_sampled_policy(path)
        self.assertEqual(game, trainer.game_config)
        self.assertEqual(restored.probabilities(info), expected)
        self.assertNotIn("network", payload)
        self.assertNotIn("networks", payload)
        self.assertNotIn("neural", repr(payload).lower())

    def test_more_samples_reduce_distance_to_full_tree_reference_across_seeds(self) -> None:
        game = GameConfig(hand_size=1)
        reference = TabularCFRTrainer(
            game,
            TabularCFRConfig(iterations=30, workers=1, checkpoint_every=30, log_every=30),
        )
        for _ in range(30):
            reference.train_iteration()
        reference_entries = reference.policy_entries()

        def distance(trainer: MCCFRTrainer) -> float:
            sampled = trainer.policy_entries()
            total = 0.0
            for key, (actions, target) in reference_entries.items():
                uniform = (1.0 / len(actions),) * len(actions)
                actual = sampled.get(key, (actions, uniform))[1]
                total += sum(abs(left - right) for left, right in zip(target, actual))
            return total / len(reference_entries)

        low_distances = []
        high_distances = []
        for seed in (11, 22, 33):
            config = self.config(algorithm="external_sampling_mccfr", seed=seed)
            low = MCCFRTrainer(game, config, _testing_only_allow_small_game=True)
            high = MCCFRTrainer(game, config, _testing_only_allow_small_game=True)
            for _ in range(5):
                low.train_iteration()
            for _ in range(300):
                high.train_iteration()
            low_distances.append(distance(low))
            high_distances.append(distance(high))

        self.assertLess(
            sum(high_distances) / len(high_distances),
            sum(low_distances) / len(low_distances) - 0.05,
        )
        self.assertTrue(all(high < low for high, low in zip(high_distances, low_distances)))


if __name__ == "__main__":
    unittest.main()
