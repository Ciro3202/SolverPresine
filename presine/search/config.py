from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class BeliefConfig:
    """Bounded particle belief over hidden hands."""

    particles: int = 256
    history_temperature: float = 1.0
    resample_ess_ratio: float = 0.50
    mcmc_steps: int = 2
    mcmc_burn_in: int = 8
    history_model: str = "heuristic"

    def __post_init__(self) -> None:
        if self.particles <= 0:
            raise ValueError("belief.particles must be positive")
        if self.history_temperature <= 0:
            raise ValueError("belief.history_temperature must be positive")
        if not 0 < self.resample_ess_ratio <= 1:
            raise ValueError("belief.resample_ess_ratio must be in (0, 1]")
        if self.mcmc_steps < 0 or self.mcmc_burn_in < 0:
            raise ValueError("belief MCMC counts cannot be negative")
        if self.history_model not in {"heuristic", "uniform"}:
            raise ValueError("belief.history_model must be heuristic or uniform")


@dataclass(frozen=True, slots=True)
class ISMCTSConfig:
    simulations: int = 4_000
    exploration: float = 1.4
    rollout_depth: int = 64
    max_tree_nodes: int = 200_000
    temperature: float = 0.35
    rollout_policy: str = "heuristic"

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("ismcts.simulations must be positive")
        if self.exploration < 0:
            raise ValueError("ismcts.exploration cannot be negative")
        if self.rollout_depth <= 0 or self.max_tree_nodes <= 0:
            raise ValueError("ISMCTS depth and tree limits must be positive")
        if self.temperature < 0:
            raise ValueError("ismcts.temperature cannot be negative")
        if self.rollout_policy not in {"heuristic", "uniform"}:
            raise ValueError("ismcts.rollout_policy must be heuristic or uniform")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    seed: int = 20260804
    mode: str = "ismcts"
    workers: int = 1
    tablebase_tricks: int = 2
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    ismcts: ISMCTSConfig = field(default_factory=ISMCTSConfig)

    def __post_init__(self) -> None:
        if self.mode != "ismcts":
            raise ValueError("search.mode must be ismcts")
        if self.workers <= 0:
            raise ValueError("search.workers must be positive")
        if self.tablebase_tricks not in (0, 1, 2):
            raise ValueError("search.tablebase_tricks must be 0, 1, or 2")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> SearchConfig:
        raw = dict(values)
        belief = BeliefConfig(**dict(raw.pop("belief", {})))
        ismcts = ISMCTSConfig(**dict(raw.pop("ismcts", {})))
        return cls(belief=belief, ismcts=ismcts, **raw)
