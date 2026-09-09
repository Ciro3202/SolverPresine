from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from presine.game.observation import InformationState
from presine.game.state import RoundState

from .config import SearchConfig
from .ismcts import ISMCTS, SearchResult, visit_probabilities
from .presine_adapter import PresineAdapter, PresineBelief


def _run_presine_search(
    state: RoundState,
    observer: int,
    config: SearchConfig,
    seed: int,
    guidance=None,
    guidance_fraction: float = 0.0,
    value_guidance_fraction: float | None = None,
) -> tuple[SearchResult, dict[str, float | int]]:
    belief = PresineBelief(
        state,
        observer,
        config.belief,
        seed=seed ^ 0x5DEECE66D,
    )
    adapter = PresineAdapter(
        tablebase_tricks=config.tablebase_tricks,
        rollout_policy="neural" if guidance is not None else config.ismcts.rollout_policy,
        neural_model=guidance,
        neural_guidance_fraction=guidance_fraction,
        neural_value_fraction=value_guidance_fraction,
    )
    result = ISMCTS(adapter, belief, observer, config.ismcts, seed=seed).search()
    return result, belief.diagnostics()


class PresineSearchPolicy:
    """Online policy backed by information-set MCTS."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._executor: ProcessPoolExecutor | None = None
        self.decisions = 0
        self.simulations = 0
        self.tree_nodes = 0
        self.ess_total = 0.0
        self.elapsed_seconds = 0.0

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        del info
        raise RuntimeError("PresineSearchPolicy needs the observed state; use probabilities_state")

    def probabilities_state(self, state: RoundState) -> tuple[float, ...]:
        started = time.perf_counter()
        results = self._search_many(state, state.current_player)
        actions = results[0][0].actions
        visits = [0] * len(actions)
        ess = 0.0
        for result, diagnostics in results:
            if result.actions != actions:
                raise RuntimeError("parallel searches returned different root actions")
            for index, value in enumerate(result.visits):
                visits[index] += value
            self.simulations += result.simulations
            self.tree_nodes += result.tree_nodes
            ess += float(diagnostics["effective_sample_size"])
        self.decisions += 1
        self.ess_total += ess / len(results)
        self.elapsed_seconds += time.perf_counter() - started
        return visit_probabilities(visits, self.config.ismcts.temperature)

    def _search_many(
        self, state: RoundState, observer: int
    ) -> list[tuple[SearchResult, dict[str, float | int]]]:
        workers = min(self.config.workers, self.config.ismcts.simulations)
        if workers == 1:
            seed = self._rng.randrange(2**63)
            return [_run_presine_search(state, observer, self.config, seed)]

        base, remainder = divmod(self.config.ismcts.simulations, workers)
        configs = [
            replace(
                self.config,
                workers=1,
                ismcts=replace(
                    self.config.ismcts,
                    simulations=base + (1 if index < remainder else 0),
                ),
            )
            for index in range(workers)
        ]
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.config.workers)
        futures = [
            self._executor.submit(
                _run_presine_search,
                state,
                observer,
                worker_config,
                self._rng.randrange(2**63),
            )
            for worker_config in configs
        ]
        return [future.result() for future in futures]

    def stats(self) -> dict[str, float | int | str]:
        return {
            "mode": "ismcts",
            "workers": self.config.workers,
            "decisions": self.decisions,
            "simulations": self.simulations,
            "tree_nodes": self.tree_nodes,
            "elapsed_seconds": self.elapsed_seconds,
            "simulations_per_second": (
                self.simulations / self.elapsed_seconds if self.elapsed_seconds else 0.0
            ),
            "mean_effective_particles": (
                self.ess_total / self.decisions if self.decisions else 0.0
            ),
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> PresineSearchPolicy:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
