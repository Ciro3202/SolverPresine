from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for neural blueprint tests")
class NeuralMultiplayerTests(unittest.TestCase):
    def test_shared_bundle_round_trip_and_all_round_heads(self) -> None:
        from presine.game.config import GameConfig
        from presine.game.state import RoundState
        from presine.learning.multiplayer.neural_blueprint import (
            NeuralBlueprintArchitecture,
            load_neural_multiplayer_blueprint,
            make_blueprint,
            save_neural_multiplayer_blueprint,
        )

        policy = make_blueprint(3, NeuralBlueprintArchitecture(hidden_sizes=(16, 8)), seed=7)
        self.assertEqual(policy.stats()["stored_state_rows"], 0)
        self.assertLess(policy.stats()["fp32_parameter_bytes"], 1_000_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            save_neural_multiplayer_blueprint(policy, path)
            loaded = load_neural_multiplayer_blueprint(path, players=3)
            for hand_size in range(1, 6):
                state = RoundState(GameConfig(players=3, hand_size=hand_size))
                state.apply_chance(state.sample_chance(__import__("random").Random(hand_size)))
                probabilities = loaded.probabilities(state.information_state())
                self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_scheduler_reallocates_satisfied_round_capacity(self) -> None:
        from presine.learning.multiplayer.neural_expert_iteration import DynamicRoundScheduler

        scheduler = DynamicRoundScheduler((1, 1, 2, 2, 2), 0.05)
        before = scheduler.allocation(80, {}, 100)
        scheduler.satisfied[5] = True
        after = scheduler.allocation(80, {}, 100)
        self.assertGreater(before.count(5), after.count(5))
        self.assertGreater(after.count(5), 0)

    def test_ace_conditioned_training_includes_r1(self) -> None:
        from presine.learning.multiplayer.neural_expert_iteration import (
            DynamicRoundScheduler,
            NeuralExpertIterationConfig,
        )

        config = NeuralExpertIterationConfig()
        self.assertEqual(config.round_weights[0], 1.0)
        scheduler = DynamicRoundScheduler(config.round_weights, 0.05)
        allocation = scheduler.allocation(128, {}, 20_000)
        self.assertIn(1, allocation)

    def test_round_repair_trains_only_the_selected_heads(self) -> None:
        import torch

        from presine.learning.multiplayer.neural_blueprint import (
            NeuralBlueprintArchitecture,
        )
        from presine.learning.multiplayer.neural_expert_iteration import (
            NeuralExpertIterationConfig,
            NeuralMultiplayerExpertTrainer,
        )
        from presine.search.config import SearchConfig

        trainer = NeuralMultiplayerExpertTrainer(
            3,
            SearchConfig(),
            NeuralExpertIterationConfig(
                iterations=2,
                games_per_iteration=1,
                cpu_workers=1,
                train_steps_per_iteration=1,
                batch_size=1,
                memory_examples=8,
                validation_examples=8,
                snapshot_every=1,
                max_snapshots=1,
                report_history=2,
                repair_round=1,
                architecture=NeuralBlueprintArchitecture(hidden_sizes=(16, 8)),
            ),
        )
        trainer.prepare_round_repair(1)
        trainable = {
            name
            for name, parameter in trainer.network.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("policy_heads.1.") or name.startswith("value_heads.1.")
                for name in trainable
            )
        )

        protected = {
            name: value.detach().clone()
            for name, value in trainer.network.state_dict().items()
            if not name.startswith("policy_heads.1.") and not name.startswith("value_heads.1.")
        }
        states = torch.randn(4, trainer.network.input_dim)
        logits, values = trainer.network(states, 1)
        loss = logits.square().mean() + values.square().mean()
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.optimizer.step()

        current = trainer.network.state_dict()
        for name, before in protected.items():
            self.assertTrue(torch.equal(before, current[name]), name)

    def test_internal_runtime_limit_saves_before_scheduler_shutdown(self) -> None:
        from presine.learning.multiplayer.neural_blueprint import NeuralBlueprintArchitecture
        from presine.learning.multiplayer.neural_expert_iteration import (
            NeuralExpertIterationConfig,
            NeuralMultiplayerExpertTrainer,
            SatisfactionConfig,
        )
        from presine.search.config import SearchConfig

        trainer = NeuralMultiplayerExpertTrainer(
            3,
            SearchConfig(),
            NeuralExpertIterationConfig(
                iterations=10,
                games_per_iteration=1,
                cpu_workers=1,
                train_steps_per_iteration=1,
                batch_size=1,
                memory_examples=8,
                validation_examples=8,
                round_weights=(0, 0, 1, 0, 0),
                snapshot_every=10,
                max_snapshots=1,
                checkpoint_minutes=60,
                max_runtime_minutes=1e-9,
                report_history=2,
                architecture=NeuralBlueprintArchitecture(hidden_sizes=(16, 8)),
                satisfaction=SatisfactionConfig(validation_every=10),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pkl.gz"
            with (
                patch.object(trainer, "_collect", return_value=(0, {3: 1}, {})),
                patch.object(
                    trainer,
                    "_fit",
                    return_value={"total": 0.0, "policy": 0.0, "value": 0.0},
                ),
            ):
                report = trainer.train(checkpoint_path=checkpoint)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(report["status"], "runtime_limit")
            self.assertTrue(report["requires_continuation"])
            self.assertEqual(report["iteration"], 1)


if __name__ == "__main__":
    unittest.main()
