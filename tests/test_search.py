from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.search.config import BeliefConfig, ISMCTSConfig, SearchConfig
from presine.search.ismcts import ISMCTS
from presine.search.policy import PresineSearchPolicy
from presine.search.presine_adapter import (
    HiddenHands,
    PresineAdapter,
    PresineBelief,
    PresineHistoryModel,
)


@dataclass
class ToyState:
    target: tuple[int, int, int]
    choices: list[int]


class ToyAdapter:
    def player_count(self, state: ToyState) -> int:
        return 3

    def current_player(self, state: ToyState) -> int:
        return len(state.choices)

    def is_terminal(self, state: ToyState) -> bool:
        return len(state.choices) == 3

    def legal_actions(self, state: ToyState) -> tuple[int, ...]:
        return () if self.is_terminal(state) else (0, 1)

    def clone(self, state: ToyState) -> ToyState:
        return ToyState(state.target, list(state.choices))

    def apply_action(self, state: ToyState, action: int) -> None:
        state.choices.append(action)

    def utilities(self, state: ToyState) -> tuple[float, ...]:
        return tuple(
            1.0 if choice == target else -1.0 for choice, target in zip(state.choices, state.target)
        )

    def information_key(self, state: ToyState, observer: int):
        return observer, tuple(state.choices)

    def perfect_information_key(self, state: ToyState):
        return state.target, tuple(state.choices)

    def rollout_action(self, state, legal, rng):
        return rng.choice(legal)


class ToyBelief:
    def sample_state(self, rng: random.Random) -> ToyState:
        return ToyState(tuple(rng.randrange(2) for _ in range(3)), [])


class SearchTests(unittest.TestCase):
    def test_ismcts_core_accepts_three_player_utilities(self) -> None:
        search = ISMCTS(
            ToyAdapter(),
            ToyBelief(),
            observer=0,
            config=ISMCTSConfig(simulations=200, rollout_depth=4),
            seed=7,
        )
        result = search.search()
        self.assertEqual(result.actions, (0, 1))
        self.assertAlmostEqual(sum(result.probabilities), 1.0)
        self.assertEqual(sum(result.visits), 200)
        self.assertGreater(result.tree_nodes, 1)

    def test_presine_belief_does_not_read_the_true_opponent_hand(self) -> None:
        config = GameConfig(hand_size=3)
        first = RoundState(config)
        first.apply_chance(((0, 1, 2), (3, 4, 5)))
        second = RoundState(config)
        second.apply_chance(((0, 1, 2), (30, 31, 32)))
        belief_config = BeliefConfig(particles=32, mcmc_steps=0, mcmc_burn_in=0)
        belief_a = PresineBelief(first, 0, belief_config, seed=101)
        belief_b = PresineBelief(second, 0, belief_config, seed=101)
        sample_a = belief_a.sample_state(random.Random(19))
        sample_b = belief_b.sample_state(random.Random(19))
        self.assertEqual(sample_a.hands, sample_b.hands)
        self.assertEqual(sample_a.hands[0], [0, 1, 2])
        self.assertEqual(len(sample_a.hands[1]), 3)

    def test_blind_round_belief_samples_only_the_hidden_own_card(self) -> None:
        state = RoundState(GameConfig(players=3, hand_size=1, starting_player=0))
        state.apply_chance(((3,), (7,), (20,)))
        belief = PresineBelief(
            state,
            observer=0,
            config=BeliefConfig(
                particles=32,
                mcmc_steps=4,
                mcmc_burn_in=4,
                history_model="uniform",
            ),
            seed=103,
        )
        rng = random.Random(107)
        own_cards: set[int] = set()
        for _ in range(24):
            sample = belief.sample_state(rng)
            self.assertEqual(sample.hands[1], [7])
            self.assertEqual(sample.hands[2], [20])
            self.assertEqual(sample.initial_observations[0], (7, 20))
            self.assertEqual(
                sample.initial_observations[1],
                (20, sample.hands[0][0]),
            )
            own_cards.add(sample.hands[0][0])
        self.assertGreater(len(own_cards), 1)
        self.assertNotIn(7, own_cards)
        self.assertNotIn(20, own_cards)

    def test_blind_round_rollout_does_not_read_the_hidden_own_card(self) -> None:
        first = RoundState(GameConfig(players=3, hand_size=1))
        first.apply_chance(((3,), (7,), (20,)))
        second = RoundState(GameConfig(players=3, hand_size=1))
        second.apply_chance(((31,), (7,), (20,)))
        adapter = PresineAdapter()
        self.assertEqual(
            adapter.rollout_action(first, first.legal_actions(), random.Random(109)),
            adapter.rollout_action(second, second.legal_actions(), random.Random(109)),
        )

    def test_history_model_prefers_hands_consistent_with_a_high_bid(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (30, 31, 39)))
        state.apply_action(1)
        state.apply_action(3)
        strong = HiddenHands(((0, 1, 2), (30, 31, 39)), ())
        weak = HiddenHands(((0, 1, 2), (3, 4, 5)), ())
        model = PresineHistoryModel()
        self.assertGreater(
            model.log_likelihood(state, strong, 0),
            model.log_likelihood(state, weak, 0),
        )

    def test_uniform_history_model_removes_preference_between_valid_hands(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (30, 31, 39)))
        state.apply_action(1)
        state.apply_action(3)
        strong = HiddenHands(((0, 1, 2), (30, 31, 39)), ())
        weak = HiddenHands(((0, 1, 2), (3, 4, 5)), ())
        model = PresineHistoryModel(mode="uniform")
        self.assertEqual(
            model.log_likelihood(state, strong, 0),
            model.log_likelihood(state, weak, 0),
        )

    def test_multiplayer_tree_keys_opponent_nodes_by_opponent_information(self) -> None:
        config = GameConfig(players=3, hand_size=2, starting_player=0)
        weak = RoundState(config)
        weak.apply_chance(((0, 1), (2, 3), (4, 5)))
        weak.apply_action(0)
        strong = RoundState(config)
        strong.apply_chance(((0, 1), (30, 31), (4, 5)))
        strong.apply_action(0)

        adapter = PresineAdapter()
        self.assertEqual(weak.current_player, 1)
        self.assertNotEqual(
            adapter.information_key(weak, observer=0),
            adapter.information_key(strong, observer=0),
        )

    def test_multiplayer_tree_key_does_not_include_other_hidden_hands(self) -> None:
        config = GameConfig(players=3, hand_size=2, starting_player=0)
        first = RoundState(config)
        first.apply_chance(((0, 1), (20, 21), (2, 3)))
        first.apply_action(0)
        second = RoundState(config)
        second.apply_chance(((0, 1), (20, 21), (30, 31)))
        second.apply_action(0)

        adapter = PresineAdapter()
        self.assertEqual(
            adapter.information_key(first, observer=0),
            adapter.information_key(second, observer=0),
        )

    def test_uniform_rollout_is_a_valid_legal_action(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (30, 31, 39)))
        adapter = PresineAdapter(rollout_policy="uniform")
        legal = state.legal_actions()
        action = adapter.rollout_action(state, legal, random.Random(7))
        self.assertIn(action, legal)

    def test_presine_particles_preserve_card_constraints_after_mcmc(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 2), (3, 4, 5)))
        state.apply_action(1)
        state.apply_action(1)
        state.apply_action(0)
        belief = PresineBelief(
            state,
            observer=state.current_player,
            config=BeliefConfig(particles=32, mcmc_steps=4, mcmc_burn_in=4),
            seed=31,
        )
        sample = belief.sample_state(random.Random(37))
        active = [card for hand in sample.hands for card in hand]
        self.assertEqual(len(active), len(set(active)))
        self.assertEqual([len(hand) for hand in sample.hands], [2, 3])
        self.assertGreater(belief.diagnostics()["effective_sample_size"], 0)

    def test_presine_ismcts_smoke_reaches_terminal_rounds(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 30), (3, 4, 39)))
        belief = PresineBelief(
            state,
            observer=0,
            config=BeliefConfig(particles=16, mcmc_steps=1, mcmc_burn_in=1),
            seed=43,
        )
        result = ISMCTS(
            PresineAdapter(),
            belief,
            observer=0,
            config=ISMCTSConfig(simulations=100, rollout_depth=32),
            seed=47,
        ).search()
        self.assertEqual(result.actions, (0, 1, 2, 3))
        self.assertEqual(sum(result.visits), 100)
        self.assertAlmostEqual(sum(result.probabilities), 1.0)

    def test_root_parallel_policy_merges_worker_results(self) -> None:
        state = RoundState(GameConfig(hand_size=3))
        state.apply_chance(((0, 1, 30), (3, 4, 39)))
        config = SearchConfig(
            workers=2,
            belief=BeliefConfig(particles=8, mcmc_steps=0, mcmc_burn_in=0),
            ismcts=ISMCTSConfig(simulations=40, rollout_depth=32),
        )
        with PresineSearchPolicy(config) as policy:
            probabilities = policy.probabilities_state(state)
            stats = policy.stats()
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(stats["simulations"], 40)
        self.assertEqual(stats["workers"], 2)


if __name__ == "__main__":
    unittest.main()
