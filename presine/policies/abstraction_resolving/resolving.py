from __future__ import annotations

"""State-aware policy: exact lookup, local search, then abstract fallback."""

from presine.game.observation import InformationState
from presine.game.state import RoundState
from presine.search.config import SearchConfig
from presine.search.policy import PresineSearchPolicy

from ..tabular import TabularPolicy


class ResolvingPolicy:
    """Resolve exact-policy misses online with bounded belief search.

    The abstract blueprint is mixed into the local solution as a conservative
    prior.  It is also the failure/disabled-search fallback, so a missing exact
    row never silently becomes uniform unless neither learned layer covers it.
    """

    def __init__(
        self,
        exact: TabularPolicy | None,
        blueprint: object | None,
        search: SearchConfig | None,
        *,
        blueprint_weight: float = 0.20,
    ) -> None:
        if not 0.0 <= blueprint_weight <= 1.0:
            raise ValueError("blueprint_weight must be in [0, 1]")
        self.exact = exact
        self.blueprint = blueprint
        self.search = PresineSearchPolicy(search) if search is not None else None
        self.blueprint_weight = blueprint_weight
        self.exact_hits = 0
        self.resolutions = 0
        self.blueprint_fallbacks = 0
        self.uniform_fallbacks = 0

    @staticmethod
    def _validated_entry(info: InformationState, entry: object) -> tuple[float, ...] | None:
        if entry is None:
            return None
        actions, probabilities = entry  # type: ignore[misc]
        if tuple(actions) != info.legal_actions or len(probabilities) != len(actions):
            return None
        if any(value < 0 for value in probabilities):
            return None
        total = sum(probabilities)
        if total <= 0:
            return None
        return tuple(value / total for value in probabilities)

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        del info
        raise RuntimeError("ResolvingPolicy needs the full state; use state-aware evaluation")

    def probabilities_state(self, state: RoundState) -> tuple[float, ...]:
        info = state.information_state()
        if self.exact is not None:
            result = self._validated_entry(info, self.exact.entries.get(info.key()))
            if result is not None:
                self.exact_hits += 1
                return result

        lookup = getattr(self.blueprint, "lookup", None)
        probability_method = getattr(self.blueprint, "probabilities", None)
        prior = (
            lookup(info)
            if callable(lookup)
            else probability_method(info)
            if callable(probability_method)
            else None
        )
        if self.search is not None:
            resolved = self.search.probabilities_state(state)
            self.resolutions += 1
            if prior is not None and self.blueprint_weight:
                weight = self.blueprint_weight
                mixed = tuple(
                    (1.0 - weight) * local + weight * base for local, base in zip(resolved, prior)
                )
                total = sum(mixed)
                return tuple(value / total for value in mixed)
            return resolved
        if prior is not None:
            self.blueprint_fallbacks += 1
            return prior
        self.uniform_fallbacks += 1
        return (1.0 / len(info.legal_actions),) * len(info.legal_actions)

    def stats(self) -> dict[str, object]:
        return {
            "exact_hits": self.exact_hits,
            "local_resolutions": self.resolutions,
            "blueprint_fallbacks": self.blueprint_fallbacks,
            "uniform_fallbacks": self.uniform_fallbacks,
            "blueprint_weight": self.blueprint_weight,
            "search": self.search.stats() if self.search is not None else None,
            "blueprint": (
                self.blueprint.stats()
                if self.blueprint is not None and callable(getattr(self.blueprint, "stats", None))
                else None
            ),
        }

    def close(self) -> None:
        if self.search is not None:
            self.search.close()
        if self.exact is not None:
            close = getattr(self.exact.entries, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> ResolvingPolicy:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
