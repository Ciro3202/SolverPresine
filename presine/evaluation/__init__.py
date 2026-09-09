"""Measurement only: no trainer or policy is modified by evaluation.

``matches`` is the paired single-round comparison; ``full_match`` composes
round policies; ``multiplayer`` rotates the candidate across seats; and
``best_response`` supplies an explicitly approximate exploitability probe.
"""

from .matches import EvaluationResult, evaluate_round

__all__ = ["EvaluationResult", "evaluate_round"]
