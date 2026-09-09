"""Typed configuration loading shared by the CLI and SLURM wrappers.

The top-level keys deliberately mirror the three training families:
``game`` plus ``training`` (Deep CFR), ``mccfr`` (tabular CFR), or ``search``
(blueprint/resolver). Overrides are applied before dataclass validation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from presine.game.config import GameConfig
from presine.learning.config import TrainingConfig
from presine.learning.mccfr_config import MCCFRConfig
from presine.learning.tabular_config import TabularCFRConfig
from presine.search.config import SearchConfig


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return values


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = deepcopy(values)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value: {override}")
        dotted, rendered = override.split("=", 1)
        keys = dotted.split(".")
        cursor = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"override crosses a scalar at {key}")
            cursor = child
        cursor[keys[-1]] = yaml.safe_load(rendered)
    return result


def load_training_config(
    path: Path, overrides: list[str]
) -> tuple[GameConfig, TrainingConfig, dict[str, Any]]:
    raw = apply_overrides(load_yaml(path), overrides)
    unknown = set(raw) - {"game", "training"}
    if unknown:
        raise ValueError(f"unknown top-level training fields: {sorted(unknown)}")
    game = GameConfig.from_dict(raw.get("game", {}))
    training = TrainingConfig.from_dict(raw.get("training", {}))
    return game, training, raw


def load_tabular_config(
    path: Path, overrides: list[str]
) -> tuple[GameConfig, TabularCFRConfig, dict[str, Any]]:
    raw = apply_overrides(load_yaml(path), overrides)
    unknown = set(raw) - {"game", "tabular"}
    if unknown:
        raise ValueError(f"unknown top-level tabular fields: {sorted(unknown)}")
    game = GameConfig.from_dict(raw.get("game", {}))
    if game.hand_size not in (1, 2):
        raise ValueError("tabular CFR is intentionally limited to hand sizes 1 and 2")
    training = TabularCFRConfig.from_dict(raw.get("tabular", {}))
    return game, training, raw


def load_mccfr_config(
    path: Path, overrides: list[str]
) -> tuple[GameConfig, MCCFRConfig, dict[str, Any]]:
    raw = apply_overrides(load_yaml(path), overrides)
    unknown = set(raw) - {"game", "mccfr"}
    if unknown:
        raise ValueError(f"unknown top-level MCCFR fields: {sorted(unknown)}")
    game = GameConfig.from_dict(raw.get("game", {}))
    if game.players != 2 or game.hand_size not in (2, 3, 4, 5):
        raise ValueError("sampled tabular MCCFR/DCFR is implemented for heads-up hand_size=2..5")
    training = MCCFRConfig.from_dict(raw.get("mccfr", {}))
    return game, training, raw


def load_search_config(
    path: Path, overrides: list[str]
) -> tuple[GameConfig, SearchConfig, dict[str, Any]]:
    raw = apply_overrides(load_yaml(path), overrides)
    unknown = set(raw) - {"game", "search"}
    if unknown:
        raise ValueError(f"unknown top-level search fields: {sorted(unknown)}")
    game = GameConfig.from_dict(raw.get("game", {}))
    # The blind one-card round is now supported by the same belief/search
    # machinery.  It still has a small, bounded action space and benefits from
    # the exact public-card constraints in ISMCTS.
    search = SearchConfig.from_dict(raw.get("search", {}))
    return game, search, raw


def dump_yaml(values: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
