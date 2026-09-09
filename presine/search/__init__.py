"""Online imperfect-information search, independent from the CFR trainers.

The core interfaces are multiplayer. The exact Presine game state supports
multiple player counts; individual belief and resolving methods enforce their
own heads-up restrictions where needed.
"""

from .config import BeliefConfig, ISMCTSConfig, SearchConfig
from .ismcts import ISMCTS, SearchResult

__all__ = [
    "ISMCTS",
    "BeliefConfig",
    "ISMCTSConfig",
    "SearchConfig",
    "SearchResult",
]
