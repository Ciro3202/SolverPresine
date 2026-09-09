from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.policies.abstraction_resolving.canonical import canonical_information_key
from presine.policies.abstraction_resolving.resolving import ResolvingPolicy
from presine.policies.compact import (
    CompactBlueprintConfig,
    CompactBlueprintPolicy,
    audit_abstraction,
    load_compact_blueprint,
    save_compact_blueprint,
)
from presine.search.config import BeliefConfig, ISMCTSConfig, SearchConfig


class CompactBlueprintTests(unittest.TestCase):
    @staticmethod
    def _state() -> RoundState:
        state = RoundState(GameConfig(players=2, hand_size=4))
        state.apply_chance(state.sample_chance(random.Random(31)))
        return state

    def test_compact_policy_needs_no_state_table_and_returns_legal_distribution(self) -> None:
        state = self._state()
        policy = CompactBlueprintPolicy()
        info = state.information_state()
        probabilities = policy.probabilities(info)
        self.assertEqual(len(probabilities), len(info.legal_actions))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(policy.stats()["stored_state_rows"], 0)

    def test_strength_buckets_change_the_compact_action_abstraction(self) -> None:
        state = self._state()
        info = state.information_state()
        coarse = CompactBlueprintPolicy(CompactBlueprintConfig(strength_buckets=2))
        fine = CompactBlueprintPolicy(CompactBlueprintConfig(strength_buckets=16))
        self.assertNotEqual(coarse.probabilities(info), fine.probabilities(info))

    def test_seat_canonicalization_removes_only_absolute_player_labels(self) -> None:
        state = self._state()
        info = state.information_state()
        twin = type(info)(
            player=1 - info.player,
            phase=info.phase,
            hand_size=info.hand_size,
            starting_player=1 - info.starting_player,
            leader=1 - info.leader,
            trick_index=info.trick_index,
            bids=tuple(reversed(info.bids)),
            catches=tuple(reversed(info.catches)),
            visible_cards=info.visible_cards,
            initial_private_observation=info.initial_private_observation,
            current_trick=tuple((1 - player, card) for player, card in info.current_trick),
            public_history=tuple(
                type(event)(event.kind, 1 - event.actor, event.value)
                for event in info.public_history
            ),
            legal_actions=info.legal_actions,
        )
        self.assertEqual(canonical_information_key(info), canonical_information_key(twin))

    def test_compact_blueprint_round_trips_and_resolver_uses_it_without_exact_table(self) -> None:
        game = GameConfig(players=2, hand_size=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.json"
            save_compact_blueprint(game, path)
            loaded_game, blueprint = load_compact_blueprint(path)
            self.assertEqual(loaded_game, game)
            resolver = ResolvingPolicy(None, blueprint, None)
            probabilities = resolver.probabilities_state(self._state())
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertEqual(resolver.stats()["blueprint_fallbacks"], 1)

    def test_resolver_locally_improves_the_table_free_prior(self) -> None:
        state = self._state()
        search = SearchConfig(
            workers=1,
            belief=BeliefConfig(particles=8, mcmc_steps=0, mcmc_burn_in=0),
            ismcts=ISMCTSConfig(simulations=24, rollout_depth=32),
        )
        with ResolvingPolicy(None, CompactBlueprintPolicy(), search) as resolver:
            probabilities = resolver.probabilities_state(state)
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertEqual(resolver.stats()["local_resolutions"], 1)

    def test_abstraction_audit_reports_lossless_and_approximate_layers(self) -> None:
        report = audit_abstraction(GameConfig(players=2, hand_size=4), samples=40, seed=9)
        self.assertGreater(report["unique_exact"], 0)
        self.assertLessEqual(report["unique_seat_canonical"], report["unique_exact"])
        self.assertLessEqual(report["unique_feature_buckets"], report["unique_seat_canonical"])
        self.assertEqual(report["theoretical_seat_canonical_factor"], 2)


if __name__ == "__main__":
    unittest.main()
