from __future__ import annotations

import math
import random
from dataclasses import dataclass

from presine.game.config import GameConfig
from presine.game.state import RoundResult, RoundState
from presine.policies.base import Policy

from .full_match import RoundPolicyProvider
from .statistics import adaptive_sample_complete, mean_ci95


def play_multiplayer_round(
    config: GameConfig,
    policies: tuple[Policy, ...],
    *,
    seed: int,
) -> RoundResult:
    if config.players < 3 or len(policies) != config.players:
        raise ValueError("multiplayer evaluation requires one policy per player")
    rng = random.Random(seed)
    state = RoundState(config)
    state.apply_chance(state.sample_chance(rng))
    while not state.is_terminal:
        info = state.information_state()
        policy = policies[state.current_player]
        state_method = getattr(policy, "probabilities_state", None)
        probabilities = (
            state_method(state) if state_method is not None else policy.probabilities(info)
        )
        if len(probabilities) != len(info.legal_actions):
            raise ValueError("policy returned the wrong probability vector")
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("policy returned invalid probabilities")
        total = sum(probabilities)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("policy probabilities do not sum to one")
        action = rng.choices(info.legal_actions, weights=probabilities, k=1)[0]
        state.apply_action(action)
    assert state.result is not None
    return state.result


@dataclass(frozen=True, slots=True)
class MultiplayerMatchEvaluation:
    matches: int
    outright_wins: int
    tied_firsts: int
    losses: int
    mean_win_credit: float
    mean_candidate_errors: float
    mean_best_opponent_errors: float
    mean_error_margin: float
    error_margin_ci95: tuple[float, float]
    by_seat_win_credit: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "outright_wins": self.outright_wins,
            "tied_firsts": self.tied_firsts,
            "losses": self.losses,
            "outright_win_rate": self.outright_wins / self.matches,
            "tied_first_rate": self.tied_firsts / self.matches,
            "loss_rate": self.losses / self.matches,
            "mean_win_credit": self.mean_win_credit,
            "mean_candidate_errors": self.mean_candidate_errors,
            "mean_best_opponent_errors": self.mean_best_opponent_errors,
            "mean_error_margin": self.mean_error_margin,
            "error_margin_ci95": self.error_margin_ci95,
            "by_seat_win_credit": self.by_seat_win_credit,
            "tie_rule": "first-place credit is split equally among all minimum-error players",
        }


@dataclass(frozen=True, slots=True)
class MultiplayerRoundEvaluation:
    games: int
    mean_win_credit: float
    mean_focal_errors: float
    mean_best_other_errors: float
    mean_other_errors: float
    mean_error_margin: float
    error_margin_ci95: tuple[float, float]
    mean_pairwise_error_margin: float
    pairwise_error_margin_ci95: tuple[float, float]
    ci95_half_width: float
    precision_target_reached: bool
    by_seat_win_credit: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "mean_win_credit": self.mean_win_credit,
            "mean_focal_errors": self.mean_focal_errors,
            "mean_best_other_errors": self.mean_best_other_errors,
            "mean_other_errors": self.mean_other_errors,
            "mean_error_margin": self.mean_error_margin,
            "error_margin_ci95": self.error_margin_ci95,
            "mean_pairwise_error_margin": self.mean_pairwise_error_margin,
            "pairwise_error_margin_ci95": self.pairwise_error_margin_ci95,
            "ci95_half_width": self.ci95_half_width,
            "precision_target_reached": self.precision_target_reached,
            "by_seat_win_credit": self.by_seat_win_credit,
            "tie_rule": "first-place credit is split equally among minimum-error players",
        }


def evaluate_multiplayer_round_adaptive(
    focal: object,
    repeated_opponent: object,
    *,
    players: int,
    hand_size: int,
    minimum_games: int,
    maximum_games: int,
    target_half_width: float,
    seed: int,
) -> MultiplayerRoundEvaluation:
    """Cross-play one focal policy against a repeated opponent population."""

    if not 3 <= players <= 6 or not 1 <= hand_size <= 5:
        raise ValueError("invalid multiplayer round")
    if minimum_games <= 0 or maximum_games < minimum_games or target_half_width <= 0:
        raise ValueError("invalid adaptive evaluation settings")
    margins: list[float] = []
    credits: list[float] = []
    focal_errors: list[float] = []
    best_other_errors: list[float] = []
    pairwise_margins: list[float] = []
    other_errors: list[float] = []
    seat_credit = [0.0] * players
    seat_count = [0] * players
    for game_index in range(maximum_games):
        focal_seat = game_index % players
        policies = tuple(
            focal if seat == focal_seat else repeated_opponent for seat in range(players)
        )
        result = play_multiplayer_round(
            GameConfig(
                players=players,
                hand_size=hand_size,
                starting_player=(game_index // players) % players,
            ),
            policies,  # type: ignore[arg-type]
            seed=seed + game_index * 1_000_003,
        )
        own = result.errors[focal_seat]
        opponents = [error for seat, error in enumerate(result.errors) if seat != focal_seat]
        best_other = min(opponents)
        mean_other = sum(opponents) / len(opponents)
        winners = [seat for seat, error in enumerate(result.errors) if error == min(result.errors)]
        credit = 1.0 / len(winners) if focal_seat in winners else 0.0
        margins.append(float(best_other - own))
        credits.append(credit)
        focal_errors.append(float(own))
        best_other_errors.append(float(best_other))
        other_errors.append(float(mean_other))
        pairwise_margins.append(float(mean_other - own))
        seat_credit[focal_seat] += credit
        seat_count[focal_seat] += 1
        if (game_index + 1) % (players * players) == 0 and adaptive_sample_complete(
            margins,
            minimum=minimum_games,
            maximum=maximum_games,
            target_half_width=target_half_width,
        ):
            break
    mean, interval, half_width = mean_ci95(margins)
    pairwise_mean, pairwise_interval, _ = mean_ci95(pairwise_margins)
    return MultiplayerRoundEvaluation(
        games=len(margins),
        mean_win_credit=sum(credits) / len(credits),
        mean_focal_errors=sum(focal_errors) / len(focal_errors),
        mean_best_other_errors=sum(best_other_errors) / len(best_other_errors),
        mean_other_errors=sum(other_errors) / len(other_errors),
        mean_error_margin=mean,
        error_margin_ci95=interval,
        mean_pairwise_error_margin=pairwise_mean,
        pairwise_error_margin_ci95=pairwise_interval,
        ci95_half_width=half_width,
        precision_target_reached=half_width <= target_half_width,
        by_seat_win_credit=tuple(
            seat_credit[seat] / seat_count[seat] if seat_count[seat] else math.nan
            for seat in range(players)
        ),
    )


def evaluate_multiplayer_full_match(
    candidate: RoundPolicyProvider,
    opponent: RoundPolicyProvider,
    *,
    matches: int,
    seed: int,
    players: int = 4,
) -> MultiplayerMatchEvaluation:
    if matches <= 0:
        raise ValueError("matches must be positive")
    if not 3 <= players <= 6:
        raise ValueError("multiplayer evaluation supports players in 3..6")
    credits: list[float] = []
    margins: list[float] = []
    candidate_errors: list[float] = []
    best_opponent_errors: list[float] = []
    seat_credit = [0.0] * players
    seat_count = [0] * players
    outright = tied = losses = 0
    for match_index in range(matches):
        candidate_seat = match_index % players
        total_errors = [0] * players
        for round_index, hand_size in enumerate((5, 4, 3, 2, 1)):
            config = GameConfig(
                players=players,
                hand_size=hand_size,
                starting_player=(match_index + round_index) % players,
            )
            candidate_policy = candidate.for_hand_size(hand_size)
            opponent_policy = opponent.for_hand_size(hand_size)
            policies = tuple(
                candidate_policy if seat == candidate_seat else opponent_policy
                for seat in range(players)
            )
            result = play_multiplayer_round(
                config,
                policies,
                seed=seed + match_index * 1_000_003 + hand_size * 8191,
            )
            for seat in range(players):
                total_errors[seat] += result.errors[seat]
        minimum = min(total_errors)
        winners = [seat for seat, value in enumerate(total_errors) if value == minimum]
        credit = 1.0 / len(winners) if candidate_seat in winners else 0.0
        best_other = min(value for seat, value in enumerate(total_errors) if seat != candidate_seat)
        margin = best_other - total_errors[candidate_seat]
        credits.append(credit)
        margins.append(float(margin))
        candidate_errors.append(float(total_errors[candidate_seat]))
        best_opponent_errors.append(float(best_other))
        seat_credit[candidate_seat] += credit
        seat_count[candidate_seat] += 1
        if candidate_seat not in winners:
            losses += 1
        elif len(winners) == 1:
            outright += 1
        else:
            tied += 1
    mean_margin = sum(margins) / matches
    variance = (
        sum((value - mean_margin) ** 2 for value in margins) / (matches - 1) if matches > 1 else 0.0
    )
    half_width = 1.96 * math.sqrt(variance / matches)
    return MultiplayerMatchEvaluation(
        matches=matches,
        outright_wins=outright,
        tied_firsts=tied,
        losses=losses,
        mean_win_credit=sum(credits) / matches,
        mean_candidate_errors=sum(candidate_errors) / matches,
        mean_best_opponent_errors=sum(best_opponent_errors) / matches,
        mean_error_margin=mean_margin,
        error_margin_ci95=(mean_margin - half_width, mean_margin + half_width),
        by_seat_win_credit=tuple(
            seat_credit[seat] / seat_count[seat] if seat_count[seat] else math.nan
            for seat in range(players)
        ),
    )
