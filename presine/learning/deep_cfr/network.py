from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_DIM, STATE_DIM


class StrategyNetwork(nn.Module):
    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 256, 128),
        *,
        dropout: float = 0.0,
        input_dim: int = STATE_DIM,
        output_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers: list[nn.Module] = []
        incoming = input_dim
        for size in hidden_sizes:
            layers.extend((nn.Linear(incoming, size), nn.LayerNorm(size), nn.GELU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            incoming = size
        layers.append(nn.Linear(incoming, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
