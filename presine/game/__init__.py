"""Exact, reversible Presine rules for two to six players.

Start with :class:`GameConfig`, then :class:`RoundState`. This package has no
dependency on PyTorch, CFR, files, or SLURM and is the stable core of the
project.
"""

from .config import GameConfig
from .state import Phase, RoundState

__all__ = ["GameConfig", "Phase", "RoundState"]
