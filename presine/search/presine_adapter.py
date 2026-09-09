from __future__ import annotations

import math
import random
from dataclasses import dataclass

from presine.game.cards import ACE_HIGH, ACE_LOW, ACE_OF_DENARI, NUM_CARDS
from presine.game.observation import EventKind
from presine.game.state import Phase, RoundState

from .belief import ParticleBelief
from .config import BeliefConfig

TREE_INFORMATION_SEMANTICS = "acting-player-information-sets-v1"


@dataclass(frozen=True, slots=True)
class HiddenHands:
    """Remaining hands plus the undealt stock for one determinization."""

    hands: tuple[tuple[int, ...], ...]
    stock: tuple[int, ...]


def _played_by(state: RoundState) -> tuple[tuple[int, ...], ...]:
    played: list[list[int]] = [[] for _ in range(state.config.players)]
    for event in state.public_history:
        if event.kind == EventKind.PLAY:
            played[event.actor].append(event.value)
    return tuple(tuple(cards) for cards in played)


class PresineHistoryModel:
    """Cheap probabilistic model of bids and card choices.

    It deliberately mixes a weak rational model with a uniform floor.  The
    filter can exploit informative history without deleting surprising but
    valid hands, which is important when the opponent is not the built-in
    heuristic.
    """

    def __init__(self, rational_weight: float = 0.35, mode: str = "heuristic") -> None:
        if not 0 <= rational_weight < 1:
            raise ValueError("rational_weight must be in [0, 1)")
        if mode not in {"heuristic", "uniform"}:
            raise ValueError("history model mode must be heuristic or uniform")
        self.rational_weight = rational_weight
        self.mode = mode

    @staticmethod
    def _mixture_probability(
        scores: dict[int, float], observed: int, rational_weight: float
    ) -> float:
        if observed not in scores:
            return 0.0
        maximum = max(scores.values())
        exp_scores = {key: math.exp(value - maximum) for key, value in scores.items()}
        total = sum(exp_scores.values())
        soft = exp_scores[observed] / total
        uniform = 1.0 / len(scores)
        return (1.0 - rational_weight) * uniform + rational_weight * soft

    def log_likelihood(
        self, observed_state: RoundState, hidden: HiddenHands, observer: int
    ) -> float:
        del observer
        players = observed_state.config.players
        hand_size = observed_state.config.hand_size
        played = _played_by(observed_state)
        initial = [sorted((*hidden.hands[player], *played[player])) for player in range(players)]
        hands = [list(hand) for hand in initial]
        bids = [-1] * players
        catches = [0] * players
        trick: list[tuple[int, int]] = []
        logp = 0.0
        for event in observed_state.public_history:
            if event.kind == EventKind.BID:
                legal = list(range(hand_size + 1))
                previous = [bid for bid in bids if bid >= 0]
                if hand_size > 1 and len(previous) == players - 1:
                    forbidden = hand_size - sum(previous)
                    if forbidden in legal:
                        legal.remove(forbidden)
                if hand_size == 1:
                    # In the blind round a player sees every *other* card and
                    # never their own.  Keep the cheap history prior aligned
                    # with that legal observation instead of scoring the bid
                    # from the determinized hidden card.
                    visible = [
                        card
                        for offset in range(1, players)
                        for card in initial[(event.actor + offset) % players]
                    ]
                    strength = int(bool(visible) and visible[0] < 20)
                else:
                    strength = sum(
                        card >= 20 or card == ACE_OF_DENARI for card in initial[event.actor]
                    )
                scores = {bid: -1.25 * abs(bid - strength) for bid in legal}
                probability = self._mixture_probability(scores, event.value, self.rational_weight)
                if probability <= 0:
                    return -math.inf
                logp += math.log(probability)
                bids[event.actor] = event.value
            elif event.kind == EventKind.PLAY:
                actor = event.actor
                if event.value not in hands[actor]:
                    return -math.inf
                need = bids[actor] < 0 or catches[actor] < bids[actor]
                scores: dict[int, float] = {}
                for card in hands[actor]:
                    rank = 40 if card == ACE_OF_DENARI else card
                    scores[card] = (rank / 20.0) * (1.0 if need else -1.0)
                probability = self._mixture_probability(scores, event.value, self.rational_weight)
                logp += math.log(max(probability, 1e-300))
                hands[actor].remove(event.value)
                trick.append((actor, event.value))
            elif event.kind == EventKind.TRICK_WINNER:
                catches[event.value] += 1
                trick.clear()
            elif event.kind == EventKind.ACE_CHOICE:
                need = bids[event.actor] < 0 or catches[event.actor] < bids[event.actor]
                preferred = ACE_HIGH if need else ACE_LOW
                probability = 0.75 if event.value == preferred else 0.25
                logp += math.log(
                    (1.0 - self.rational_weight) * 0.5 + self.rational_weight * probability
                )
        if self.mode == "uniform":
            # The uniform ablation retains hard consistency (a hidden deal may
            # not explain a card that was publicly played) but contributes no
            # preference for high cards, bids, or “sensible” choices.
            return 0.0
        return logp


class PresineSwapProposal:
    """Symmetric swap proposal that preserves hand sizes and unique cards."""

    def __init__(self, observer: int, *, blind_round: bool = False) -> None:
        self.observer = observer
        self.blind_round = blind_round

    def propose(self, hidden: HiddenHands, rng: random.Random) -> HiddenHands:
        groups = [list(hand) for hand in hidden.hands]
        slots = [
            player
            for player, hand in enumerate(groups)
            if hand and (player == self.observer if self.blind_round else player != self.observer)
        ]
        stock = list(hidden.stock)
        if stock:
            slots.append(-1)
        if len(slots) < 2:
            return hidden
        left, right = rng.sample(slots, 2)
        left_group = stock if left == -1 else groups[left]
        right_group = stock if right == -1 else groups[right]
        left_index = rng.randrange(len(left_group))
        right_index = rng.randrange(len(right_group))
        left_group[left_index], right_group[right_index] = (
            right_group[right_index],
            left_group[left_index],
        )
        return HiddenHands(tuple(tuple(sorted(hand)) for hand in groups), tuple(sorted(stock)))


class PresineBelief:
    """History-filtered hidden-hand belief and determinization sampler."""

    def __init__(
        self,
        observed_state: RoundState,
        observer: int,
        config: BeliefConfig,
        *,
        seed: int,
    ) -> None:
        if observed_state.is_terminal or observed_state.phase == Phase.CHANCE:
            raise ValueError("beliefs require a non-terminal decision state")
        if not 0 <= observer < observed_state.config.players:
            raise ValueError("observer is out of range")
        self.root = PresineAdapter().clone(observed_state)
        self.observer = observer
        self.config = config
        self.subgames = PresineSubgameBuilder(self.root, observer)
        self.played = _played_by(self.root)
        rng = random.Random(seed)
        particles = [self._uniform_hidden(rng) for _ in range(config.particles)]
        self._belief = ParticleBelief(
            self.root,
            observer,
            particles,
            PresineHistoryModel(mode=config.history_model),
            PresineSwapProposal(
                observer,
                blind_round=observed_state.config.hand_size == 1,
            ),
            config,
            rng,
        )

    def _uniform_hidden(self, rng: random.Random) -> HiddenHands:
        played = {card for cards in self.played for card in cards}
        if self.root.config.hand_size == 1:
            # R1 reverses the normal information structure: the observer sees
            # every opponent card while their own card is hidden.  Preserve
            # the visible opponent cards exactly and sample only the missing
            # own card (or none once it has become public).
            visible = tuple(self.root.initial_observations[self.observer])
            if len(visible) != self.root.config.players - 1:
                raise ValueError("blind-round observation has the wrong size")
            known = set(visible) | played
            unknown = [card for card in range(NUM_CARDS) if card not in known]
            rng.shuffle(unknown)
            hands: list[tuple[int, ...]] = [() for _ in range(self.root.config.players)]
            for offset, card in enumerate(visible, start=1):
                player = (self.observer + offset) % self.root.config.players
                if len(self.played[player]) < self.root.config.hand_size:
                    hands[player] = (card,)
            own_count = self.root.config.hand_size - len(self.played[self.observer])
            hands[self.observer] = tuple(sorted(unknown[:own_count]))
            return HiddenHands(
                tuple(hands),
                tuple(sorted(unknown[own_count:])),
            )

        known = set(self.root.hands[self.observer]) | played
        unknown = [card for card in range(NUM_CARDS) if card not in known]
        rng.shuffle(unknown)
        hands: list[tuple[int, ...]] = [() for _ in range(self.root.config.players)]
        hands[self.observer] = tuple(sorted(self.root.hands[self.observer]))
        cursor = 0
        for player in range(self.root.config.players):
            if player == self.observer:
                continue
            count = self.root.config.hand_size - len(self.played[player])
            hands[player] = tuple(sorted(unknown[cursor : cursor + count]))
            cursor += count
        return HiddenHands(tuple(hands), tuple(sorted(unknown[cursor:])))

    def sample_state(self, rng: random.Random) -> RoundState:
        hidden = self._belief.sample_hidden(rng)
        return self.subgames.build(hidden)

    def diagnostics(self) -> dict[str, float | int]:
        return self._belief.diagnostics()


class PresineAdapter:
    """Efficient adapter from RoundState to the generic multiplayer API."""

    def __init__(
        self,
        tablebase_tricks: int = 0,
        rollout_policy: str = "heuristic",
        neural_model=None,
        neural_guidance_fraction: float = 0.0,
        neural_value_fraction: float | None = None,
    ) -> None:
        from .tablebase import EndgameTablebase

        if rollout_policy not in {"heuristic", "uniform", "neural"}:
            raise ValueError("rollout_policy must be heuristic, uniform, or neural")
        if rollout_policy == "neural" and neural_model is None:
            raise ValueError("neural rollout policy requires a neural model")
        if not 0.0 <= neural_guidance_fraction <= 1.0:
            raise ValueError("neural_guidance_fraction must be in [0, 1]")
        self.rollout_policy = rollout_policy
        self.neural_model = neural_model
        self.neural_guidance_fraction = neural_guidance_fraction
        self.neural_value_fraction = (
            neural_guidance_fraction if neural_value_fraction is None else neural_value_fraction
        )
        if not 0.0 <= self.neural_value_fraction <= 1.0:
            raise ValueError("neural_value_fraction must be in [0, 1]")

        self.tablebase = EndgameTablebase(tablebase_tricks) if tablebase_tricks else None

    def player_count(self, state: RoundState) -> int:
        return state.config.players

    def current_player(self, state: RoundState) -> int:
        return state.current_player

    def is_terminal(self, state: RoundState) -> bool:
        return state.is_terminal

    def legal_actions(self, state: RoundState) -> tuple[int, ...]:
        return state.legal_actions()

    def clone(self, state: RoundState) -> RoundState:
        clone = RoundState(state.config, debug=False)
        clone.phase = state.phase
        clone.current_player = state.current_player
        clone.leader = state.leader
        clone.trick_index = state.trick_index
        clone.bids = list(state.bids)
        clone.catches = list(state.catches)
        clone.hands = [list(hand) for hand in state.hands]
        clone.initial_observations = list(state.initial_observations)
        clone.current_trick = list(state.current_trick)
        clone.public_history = list(state.public_history)
        clone.ace_high = state.ace_high
        clone.result = state.result
        return clone

    def apply_action(self, state: RoundState, action: int) -> None:
        state.apply_action_unchecked(action)

    def utilities(self, state: RoundState) -> tuple[float, ...]:
        return tuple(state.utilities())

    def leaf_utilities(self, state: RoundState) -> tuple[float, ...] | None:
        if self.tablebase is None or not self.tablebase.eligible(state):
            return None
        return self.tablebase.solve(state)

    def cutoff_utilities(self, state: RoundState) -> tuple[float, ...] | None:
        """Estimate a depth-limited leaf with the progressively trusted value head."""
        if self.neural_model is None or self.neural_value_fraction <= 0.0:
            return None
        info = state.information_state()
        relative = self.neural_model.relative_values(info)
        actor = state.current_player
        absolute = [0.0] * state.config.players
        for offset, value in enumerate(relative):
            absolute[(actor + offset) % state.config.players] = (
                float(value) * self.neural_value_fraction
            )
        return tuple(absolute)

    def information_key(self, state: RoundState, observer: int):
        # The root belongs to ``observer``.  Deeper decisions must instead be
        # grouped by what the player who is actually moving can know.  Using
        # the root observer here prevents an opponent's bid from depending on
        # that opponent's own cards and creates a systematically weak teacher.
        # The determinization supplies one possible private hand for the actor;
        # its information state never contains another player's hidden cards.
        del observer
        actor = state.current_player
        return actor, state.information_state(actor).key()

    def perfect_information_key(self, state: RoundState):
        return state.state_key()

    def rollout_action(self, state: RoundState, legal: tuple[int, ...], rng: random.Random) -> int:
        if self.rollout_policy == "uniform":
            return rng.choice(legal)
        if self.rollout_policy == "neural" and rng.random() < self.neural_guidance_fraction:
            probabilities = self.neural_model.probabilities(state.information_state())
            return rng.choices(legal, weights=probabilities, k=1)[0]
        # A small domain prior is cheaper and substantially less noisy than a
        # fully random rollout while leaving exploration in the tree intact.
        if state.phase == Phase.BID:
            if state.config.hand_size == 1:
                # The generic rollout has access to a determinization, but a
                # blind-round player does not know the card in their own hand.
                # Base the cheap choice only on their legal observation.
                visible = state.information_state().visible_cards
                estimate = int(bool(visible) and visible[0] < 20)
                return min(
                    legal,
                    key=lambda action: (abs(action - estimate), action),
                )
            hand = state.hands[state.current_player]
            estimate = sum(card >= 20 or card == ACE_OF_DENARI for card in hand)
            return min(legal, key=lambda action: (abs(action - estimate), action))
        if state.phase == Phase.ACE_CHOICE:
            actor = state.current_player
            return ACE_HIGH if state.catches[actor] < state.bids[actor] else ACE_LOW
        if rng.random() < 0.15:
            return rng.choice(legal)
        actor = state.current_player
        need = state.catches[actor] < state.bids[actor]
        return max(legal) if need else min(legal)


class PresineSubgameBuilder:
    """Build only the determinized root; descendants are created on demand.

    A public root template is kept once.  No full game tree and no collection
    of cloned subgames is retained, so memory grows with visited ISMCTS nodes,
    not with the number of possible deals.
    """

    def __init__(self, root: RoundState, observer: int) -> None:
        self.adapter = PresineAdapter()
        self.root = self.adapter.clone(root)
        self.observer = observer
        self.played = _played_by(root)

    def build(self, hidden: HiddenHands) -> RoundState:
        state = self.adapter.clone(self.root)
        state.hands = [list(hand) for hand in hidden.hands]
        initial_hands = [
            tuple(sorted((*hidden.hands[player], *self.played[player])))
            for player in range(state.config.players)
        ]
        if state.config.hand_size == 1:
            state.initial_observations = [
                tuple(
                    card
                    for offset in range(1, state.config.players)
                    for card in initial_hands[(player + offset) % state.config.players]
                )
                for player in range(state.config.players)
            ]
        else:
            state.initial_observations = initial_hands
            state.initial_observations[self.observer] = self.root.initial_observations[
                self.observer
            ]
        return state
