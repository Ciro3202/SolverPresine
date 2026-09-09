from __future__ import annotations

import random
from collections.abc import Hashable
from typing import Protocol, TypeVar

StateT = TypeVar("StateT")
HiddenT = TypeVar("HiddenT")
Action = int
Key = Hashable


class GameAdapter(Protocol[StateT]):
    """Minimal game API required by both search engines.

    Utilities are a vector and selection maximises the component belonging to
    the player who acts at a node, so the core is not restricted to zero-sum
    or to two players.
    """

    def player_count(self, state: StateT) -> int: ...

    def current_player(self, state: StateT) -> int: ...

    def is_terminal(self, state: StateT) -> bool: ...

    def legal_actions(self, state: StateT) -> tuple[Action, ...]: ...

    def clone(self, state: StateT) -> StateT: ...

    def apply_action(self, state: StateT, action: Action) -> None: ...

    def utilities(self, state: StateT) -> tuple[float, ...]: ...

    def information_key(self, state: StateT, observer: int) -> Key: ...

    def perfect_information_key(self, state: StateT) -> Key: ...

    def rollout_action(
        self, state: StateT, legal: tuple[Action, ...], rng: random.Random
    ) -> Action: ...


class BeliefSampler(Protocol[StateT]):
    """Samples a root state consistent with one player's observations."""

    def sample_state(self, rng: random.Random) -> StateT: ...


class LogLikelihoodModel(Protocol[StateT, HiddenT]):
    """Pluggable behavioural model used by history filtering."""

    def log_likelihood(self, observed_state: StateT, hidden: HiddenT, observer: int) -> float: ...


class MCMCProposal(Protocol[HiddenT]):
    def propose(self, hidden: HiddenT, rng: random.Random) -> HiddenT: ...
