from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from presine.game.config import GameConfig
from presine.policies.base import Policy

from .matches import play_round


class RoundPolicyProvider(Protocol):
    def for_hand_size(self, hand_size: int) -> Policy: ...


class RepeatedPolicy:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def for_hand_size(self, hand_size: int) -> Policy:
        return self.policy


@dataclass(frozen=True, slots=True)
class MatchEvaluation:
    matches: int
    wins: int
    ties: int
    losses: int
    mean_score: float
    score_ci95: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "win_rate": self.wins / self.matches,
            "tie_rate": self.ties / self.matches,
            "loss_rate": self.losses / self.matches,
            "mean_score": self.mean_score,
            "score_ci95": self.score_ci95,
        }


def evaluate_full_match(
    candidate: RoundPolicyProvider,
    opponent: RoundPolicyProvider,
    *,
    matches: int,
    seed: int,
    rotate_seats: bool = True,
) -> MatchEvaluation:
    if matches <= 0:
        raise ValueError("matches must be positive")
    scores: list[float] = []
    wins = ties = losses = 0
    for index in range(matches):
        pair_index = index // 2 if rotate_seats else index
        match_seed = seed + pair_index * 1_000_003
        candidate_seat = index % 2 if rotate_seats else 0
        total_errors = [0, 0]
        for round_index, hand_size in enumerate((5, 4, 3, 2, 1)):
            config = GameConfig(
                hand_size=hand_size,
                starting_player=round_index % 2,
            )
            candidate_policy = candidate.for_hand_size(hand_size)
            opponent_policy = opponent.for_hand_size(hand_size)
            policies = (
                (candidate_policy, opponent_policy)
                if candidate_seat == 0
                else (opponent_policy, candidate_policy)
            )
            round_seed = match_seed + hand_size * 8191
            margin, result = play_round(config, policies, seed=round_seed)
            del margin
            total_errors[0] += result.errors[0]
            total_errors[1] += result.errors[1]
        raw_score = total_errors[1] - total_errors[0]
        score = raw_score if candidate_seat == 0 else -raw_score
        scores.append(float(score))
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            ties += 1
    mean = sum(scores) / matches
    variance = sum((score - mean) ** 2 for score in scores) / (matches - 1) if matches > 1 else 0.0
    half_width = 1.96 * math.sqrt(variance / matches)
    return MatchEvaluation(
        matches=matches,
        wins=wins,
        ties=ties,
        losses=losses,
        mean_score=mean,
        score_ci95=(mean - half_width, mean + half_width),
    )
