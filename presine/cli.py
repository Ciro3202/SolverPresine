from __future__ import annotations

import argparse
import json
from pathlib import Path

from presine.config import load_search_config
from presine.human_match import HumanMatch, HumanMatchOptions, parse_round_set, parse_rounds
from presine.policies.abstraction_resolving.abstract import load_blueprint
from presine.policies.abstraction_resolving.resolving import ResolvingPolicy
from presine.policies.blind_round import ClosedBlindRoundPolicy

ROOT = Path(__file__).resolve().parents[1]


def _heads_up_policies(
    rounds: tuple[int, ...],
    resolver_rounds: set[int],
    workers: int,
    blueprint_weight: float,
) -> dict[int, object]:
    policies: dict[int, object] = {}
    for round_size in rounds:
        if round_size == 1:
            policies[round_size] = ClosedBlindRoundPolicy()
            continue
        path = ROOT / "models" / "heads-up" / f"r{round_size}" / "blueprint.json"
        game, blueprint = load_blueprint(path)
        if game.players != 2 or game.hand_size != round_size:
            raise ValueError(f"modello HU R{round_size} incompatibile")
        if round_size not in resolver_rounds:
            policies[round_size] = blueprint
            continue
        config = ROOT / "configs" / "play" / "heads-up" / f"resolver-r{round_size}.yaml"
        _, search, _ = load_search_config(config, [f"search.workers={workers}"])
        prior = blueprint if blueprint_weight > 0 else None
        policies[round_size] = ResolvingPolicy(
            None, prior, search, blueprint_weight=blueprint_weight
        )
    return policies


def _multiplayer_policies(
    rounds: tuple[int, ...],
    resolver_rounds: set[int],
    workers: int,
    blueprint_weight: float,
    device: str,
) -> dict[int, object]:
    from presine.learning.multiplayer.neural_blueprint import (
        load_neural_multiplayer_blueprint,
    )

    bundle = ROOT / "models" / "multiplayer" / "3p" / "strategy-bundle.pt"
    neural = load_neural_multiplayer_blueprint(bundle, device=device, players=3)
    policies: dict[int, object] = {}
    for round_size in rounds:
        if round_size == 1:
            policies[round_size] = ClosedBlindRoundPolicy()
            continue
        if round_size not in resolver_rounds:
            policies[round_size] = neural
            continue
        config = ROOT / "configs" / "play" / "multiplayer" / "resolver-local.yaml"
        _, search, _ = load_search_config(
            config,
            [
                "game.players=3",
                f"game.hand_size={round_size}",
                f"search.workers={workers}",
            ],
        )
        prior = neural if blueprint_weight > 0 else None
        policies[round_size] = ResolvingPolicy(
            None, prior, search, blueprint_weight=blueprint_weight
        )
    return policies


def play(args: argparse.Namespace) -> None:
    rounds = parse_rounds(args.rounds)
    defaults = "5,4,3" if args.players == 2 else "5,4,3,2"
    resolver_rounds = set() if args.no_resolver else parse_round_set(
        args.resolver_rounds or defaults
    )
    resolver_rounds &= set(rounds)
    resolver_rounds.discard(1)
    policies = (
        _heads_up_policies(
            rounds, resolver_rounds, args.resolver_workers, args.blueprint_weight
        )
        if args.players == 2
        else _multiplayer_policies(
            rounds,
            resolver_rounds,
            args.resolver_workers,
            args.blueprint_weight,
            args.device,
        )
    )
    match = HumanMatch(
        policies,
        HumanMatchOptions(
            rounds=rounds,
            players=args.players,
            human_seat=args.human_seat,
            first_starter=args.first_starter,
            model_choice=args.model_choice,
            seed=args.seed,
            solver_learn=args.solver_learn,
        ),
    )
    try:
        report = match.play()
        report["resolver_rounds"] = sorted(resolver_rounds)
        if args.output is not None:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        match.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gioca a Presine contro il solver.")
    parser.add_argument("--players", type=int, choices=(2, 3), default=2)
    parser.add_argument("--rounds", default="5,4,3,2,1")
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--first-starter", choices=("random", "human", "model"), default="random")
    parser.add_argument("--model-choice", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--solver-learn", action="store_true")
    parser.add_argument("--no-resolver", action="store_true")
    parser.add_argument("--resolver-rounds")
    parser.add_argument("--resolver-workers", type=int, default=4)
    parser.add_argument("--blueprint-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "last-match.json")
    parser.set_defaults(function=play)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)
