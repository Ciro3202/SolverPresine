from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Generic, TypeVar

from .config import BeliefConfig
from .protocols import LogLikelihoodModel, MCMCProposal

HiddenT = TypeVar("HiddenT")
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class WeightedParticle(Generic[HiddenT]):
    hidden: HiddenT
    weight: float


class ParticleBelief(Generic[StateT, HiddenT]):
    """A bounded, history-weighted posterior with MCMC rejuvenation.

    Callers provide domain-specific uniform samples and valid proposals.  This
    keeps the inference machinery reusable for games with any player count.
    """

    def __init__(
        self,
        observed_state: StateT,
        observer: int,
        particles: list[HiddenT],
        likelihood: LogLikelihoodModel[StateT, HiddenT],
        proposal: MCMCProposal[HiddenT],
        config: BeliefConfig,
        rng: random.Random,
    ) -> None:
        if not particles:
            raise ValueError("a particle belief needs at least one particle")
        self.observed_state = observed_state
        self.observer = observer
        self.likelihood = likelihood
        self.proposal = proposal
        self.config = config
        self._particles = particles
        self._log_scores = [self._score(particle) for particle in particles]
        self._weights = self._normalise(self._log_scores)
        if self.effective_sample_size < len(self._particles) * config.resample_ess_ratio:
            self._systematic_resample(rng)
        for index in range(len(self._particles)):
            for _ in range(config.mcmc_burn_in):
                self._mcmc_step(index, rng)

    def _score(self, hidden: HiddenT) -> float:
        return (
            self.likelihood.log_likelihood(self.observed_state, hidden, self.observer)
            / self.config.history_temperature
        )

    @staticmethod
    def _normalise(log_scores: list[float]) -> list[float]:
        finite = [value for value in log_scores if math.isfinite(value)]
        if not finite:
            return [1.0 / len(log_scores)] * len(log_scores)
        offset = max(finite)
        raw = [math.exp(value - offset) if math.isfinite(value) else 0.0 for value in log_scores]
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(log_scores)] * len(log_scores)
        return [value / total for value in raw]

    @property
    def effective_sample_size(self) -> float:
        return 1.0 / sum(weight * weight for weight in self._weights)

    @property
    def particles(self) -> tuple[WeightedParticle[HiddenT], ...]:
        return tuple(
            WeightedParticle(hidden, weight)
            for hidden, weight in zip(self._particles, self._weights)
        )

    def sample_hidden(self, rng: random.Random) -> HiddenT:
        cumulative: list[float] = []
        total = 0.0
        for weight in self._weights:
            total += weight
            cumulative.append(total)
        index = min(
            bisect.bisect_left(cumulative, rng.random() * total),
            len(self._particles) - 1,
        )
        for _ in range(self.config.mcmc_steps):
            self._mcmc_step(index, rng)
        return self._particles[index]

    def _systematic_resample(self, rng: random.Random) -> None:
        size = len(self._particles)
        cumulative: list[float] = []
        total = 0.0
        for weight in self._weights:
            total += weight
            cumulative.append(total)
        offset = rng.random() / size
        selected: list[HiddenT] = []
        cursor = 0
        for index in range(size):
            target = offset + index / size
            while cursor + 1 < size and cumulative[cursor] < target:
                cursor += 1
            selected.append(self._particles[cursor])
        self._particles = selected
        self._log_scores = [self._score(particle) for particle in selected]
        self._weights = [1.0 / size] * size

    def _mcmc_step(self, index: int, rng: random.Random) -> None:
        if not self._particles:
            return
        current = self._particles[index]
        candidate = self.proposal.propose(current, rng)
        candidate_score = self._score(candidate)
        current_score = self._log_scores[index]
        if math.log(max(rng.random(), 1e-300)) <= candidate_score - current_score:
            self._particles[index] = candidate
            self._log_scores[index] = candidate_score
            # This is a Metropolis transition whose invariant distribution is
            # the posterior. Importance weights remain attached to their
            # chains; re-weighting by the likelihood here would count the same
            # evidence twice.

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "particles": len(self._particles),
            "effective_sample_size": self.effective_sample_size,
            "max_weight": max(self._weights),
            "min_weight": min(self._weights),
        }
