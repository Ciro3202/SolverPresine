from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> NetworkConfig:
        values = dict(values)
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(values["hidden_sizes"])  # type: ignore[arg-type]
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    iterations: int = 1000
    traversals_per_player: int = 100
    inference_batch_size: int = 1
    train_every: int = 1
    advantage_steps: int = 200
    policy_steps: int = 500
    policy_train_every: int = 25
    batch_size: int = 1024
    validation_samples: int = 4096
    advantage_memory: int = 200_000
    policy_memory: int = 200_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    best_response_reset_each_fit: bool = False
    best_response_policy_fallback: str = "greedy"
    seed: int = 12345
    device: str = "auto"
    deterministic: bool = True
    alternate_starting_player: bool = True
    checkpoint_every: int = 25
    log_every: int = 1
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.traversals_per_player,
            self.inference_batch_size,
            self.train_every,
            self.advantage_steps,
            self.policy_steps,
            self.batch_size,
            self.validation_samples,
            self.advantage_memory,
            self.policy_memory,
            self.checkpoint_every,
            self.log_every,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training counts and capacities must be positive")
        if self.policy_train_every < 0:
            raise ValueError("policy_train_every must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if not isinstance(self.best_response_reset_each_fit, bool):
            raise ValueError("best_response_reset_each_fit must be boolean")
        if self.best_response_policy_fallback not in {"greedy", "uniform"}:
            raise ValueError("best_response_policy_fallback must be 'greedy' or 'uniform'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TrainingConfig:
        values = dict(values)
        if "network" in values:
            values["network"] = NetworkConfig.from_dict(values["network"])  # type: ignore[arg-type]
        return cls(**values)
