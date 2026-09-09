from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TabularCFRConfig:
    """Configuration for exact-chance, full-tree tabular CFR."""

    iterations: int = 100
    workers: int = 1
    checkpoint_every: int = 10
    log_every: int = 1
    seed: int = 12345
    alternate_starting_player: bool = True
    # Disabled by default so legacy invocations retain their fixed iteration
    # count. A convergence check is intentionally opt-in.
    convergence_check_every: int = 0
    min_iterations: int = 0
    stable_checks: int = 3
    target_nashconv_upper_bound: float | None = None
    target_nashconv_exact: float | None = None
    target_policy_change: float | None = None
    exact_evaluation_max_nodes: int = 1_000_000
    policy_change_max_states: int = 1_000_000

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.convergence_check_every < 0:
            raise ValueError("convergence_check_every must be non-negative")
        if self.min_iterations < 0:
            raise ValueError("min_iterations must be non-negative")
        if self.stable_checks <= 0:
            raise ValueError("stable_checks must be positive")
        if self.exact_evaluation_max_nodes <= 0:
            raise ValueError("exact_evaluation_max_nodes must be positive")
        if self.policy_change_max_states <= 0:
            raise ValueError("policy_change_max_states must be positive")
        for name, value in (
            ("target_nashconv_upper_bound", self.target_nashconv_upper_bound),
            ("target_nashconv_exact", self.target_nashconv_exact),
            ("target_policy_change", self.target_policy_change),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TabularCFRConfig:
        unknown = set(values) - {
            "iterations",
            "workers",
            "checkpoint_every",
            "log_every",
            "seed",
            "alternate_starting_player",
            "convergence_check_every",
            "min_iterations",
            "stable_checks",
            "target_nashconv_upper_bound",
            "target_nashconv_exact",
            "target_policy_change",
            "exact_evaluation_max_nodes",
            "policy_change_max_states",
        }
        if unknown:
            raise ValueError(f"unknown tabular training fields: {sorted(unknown)}")
        return cls(**values)
