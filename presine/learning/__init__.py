"""Training methods, intentionally separated from policy deployment.

* ``trainer`` / ``network`` / ``encoding``: Deep CFR with neural networks.
* ``mccfr_trainer`` / ``tabular_trainer``: sampled and exact tabular CFR.
* ``expert_iteration``: bounded factorized blueprint training for R4/R5.
* ``checkpoint`` / ``sampled_checkpoint``: serialization only.

Importing this package does not import PyTorch until a neural class is used.
"""

from typing import Any

__all__ = ["DeepCFRTrainer", "MCCFRTrainer"]


def __getattr__(name: str) -> Any:
    if name == "DeepCFRTrainer":
        from .deep_cfr.trainer import DeepCFRTrainer

        return DeepCFRTrainer
    if name == "MCCFRTrainer":
        from .tabular_mccfr.mccfr_trainer import MCCFRTrainer

        return MCCFRTrainer
    raise AttributeError(name)
