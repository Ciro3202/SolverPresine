"""Presine solver public entry point.

Read the project in this order:

``game`` defines rules and information; ``policies`` turns information into
action distributions; ``learning`` builds policies; ``search`` optionally
improves one decision; ``evaluation`` measures a candidate on paired deals.
The command-line adapter deliberately lives last in :mod:`presine.cli`.
"""

from .game.config import GameConfig
from .game.state import RoundState

__all__ = ["GameConfig", "RoundState"]
__version__ = "0.1.0"
