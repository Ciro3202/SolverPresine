from __future__ import annotations

from dataclasses import asdict, dataclass

ALGORITHMS = (
    "external_sampling_mccfr",
    "external_sampling_mccfr_plus",
    "external_sampling_dcfr",
)
STORAGE_BACKENDS = ("memory", "mmap")
NUMERIC_DTYPES = ("float32", "float64")


@dataclass(frozen=True, slots=True)
class MCCFRConfig:
    """Configuration for the sparse heads-up round 2--5 solver.

    ``iterations`` is one-based in the DCFR equations.  Each iteration performs
    ``traversals_per_player`` external-sampling updates for each traverser and,
    when starting-player alternation is enabled, for both starting players.
    """

    algorithm: str = "external_sampling_dcfr"
    iterations: int = 1_000
    traversals_per_player: int = 100
    alpha: float = 1.5
    beta: float = 0.0
    gamma: float = 2.0
    alternate_starting_player: bool = True
    paired_starting_player: bool = True
    workers: int = 1
    replicas: int = 1
    export_policy_on_finish: bool = True
    seed: int = 12345
    checkpoint_every: int = 5_000
    log_every: int = 100
    # A wall-clock budget is deliberately part of the experiment definition:
    # sampled CFR has no reliable fixed iteration-to-time conversion.
    max_wall_seconds: int | None = None
    storage_backend: str = "memory"
    numeric_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"mccfr.algorithm must be one of {ALGORITHMS}")
        counts = (
            self.iterations,
            self.traversals_per_player,
            self.workers,
            self.replicas,
            self.checkpoint_every,
            self.log_every,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("sampled-tabular counts must be positive")
        if self.workers not in (1, 2):
            raise ValueError(
                "sampled-tabular MCCFR supports workers=1 or the two-shard "
                "starting-player implementation with workers=2"
            )
        if self.workers == 2 and not (
            self.alternate_starting_player and self.paired_starting_player
        ):
            raise ValueError(
                "workers=2 requires alternate_starting_player=true and paired_starting_player=true"
            )
        if self.replicas > 1 and self.workers != 2:
            raise ValueError(
                "independent sampled-tabular replicas require workers=2 "
                "so every replica owns both starting-player shards"
            )
        if not all(
            value == value and abs(value) != float("inf")
            for value in (
                self.alpha,
                self.beta,
                self.gamma,
            )
        ):
            raise ValueError("DCFR parameters must be finite")
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            raise ValueError("mccfr.max_wall_seconds must be positive when set")
        if self.storage_backend not in STORAGE_BACKENDS:
            raise ValueError(f"mccfr.storage_backend must be one of {STORAGE_BACKENDS}")
        if self.numeric_dtype not in NUMERIC_DTYPES:
            raise ValueError(f"mccfr.numeric_dtype must be one of {NUMERIC_DTYPES}")

    @property
    def uses_dcfr(self) -> bool:
        return self.algorithm == "external_sampling_dcfr"

    @property
    def uses_cfr_plus(self) -> bool:
        return self.algorithm == "external_sampling_mccfr_plus"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> MCCFRConfig:
        values = dict(values)
        legacy_neural = {
            "chance_pool_size",
            "chance_exploration_rate",
            "neural_fit_every",
            "neural_steps",
            "batch_size",
            "learning_rate",
            "device",
            "use_neural_fallback",
            "network",
        }
        present_legacy = sorted(set(values) & legacy_neural)
        if present_legacy:
            raise ValueError(
                "legacy MCCFR+/neural fields are not supported by the pure "
                f"sampled-tabular solver: {present_legacy}"
            )
        unknown = set(values) - {
            "algorithm",
            "iterations",
            "traversals_per_player",
            "alpha",
            "beta",
            "gamma",
            "alternate_starting_player",
            "paired_starting_player",
            "workers",
            "replicas",
            "export_policy_on_finish",
            "seed",
            "checkpoint_every",
            "log_every",
            "max_wall_seconds",
            "storage_backend",
            "numeric_dtype",
        }
        if unknown:
            raise ValueError(f"unknown sampled-tabular fields: {sorted(unknown)}")
        return cls(**values)
