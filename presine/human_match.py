from __future__ import annotations

"""Interactive human-versus-policy five-round match.

The game rules stay in :class:`RoundState`; this module only translates the
state into a small terminal interface and routes model turns to either a
normal policy or a state-aware resolver.  The same loop supports HU and one
human against two to five model opponents.
"""

import json
import os
import random
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from presine.config import load_search_config
from presine.game.cards import ACE_HIGH, ACE_LOW, card_name
from presine.game.config import GameConfig
from presine.game.state import Phase, RoundState
from presine.policies.abstraction_resolving.resolving import ResolvingPolicy
from presine.policies.linear import LinearBlueprintPolicy

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
PolicyLoader = Callable[[], object]


def parse_rounds(
    spec: str | None = None, *, from_round: int = 5, to_round: int = 1
) -> tuple[int, ...]:
    """Parse either ``5,4,3`` or an inclusive descending interval."""
    if spec:
        try:
            rounds = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError("rounds must be a comma-separated list such as 5,4,3") from exc
        if not rounds:
            raise ValueError("rounds cannot be empty")
    else:
        if not 1 <= from_round <= 5 or not 1 <= to_round <= 5:
            raise ValueError("round bounds must be between 1 and 5")
        step = -1 if from_round >= to_round else 1
        rounds = tuple(range(from_round, to_round + step, step))
    if any(round_size not in range(1, 6) for round_size in rounds):
        raise ValueError("every round must be between 1 and 5")
    if len(set(rounds)) != len(rounds):
        raise ValueError("a round may appear only once")
    return rounds


def parse_round_set(spec: str) -> set[int]:
    if not spec.strip():
        return set()
    return set(parse_rounds(spec))


@dataclass(frozen=True, slots=True)
class HumanMatchOptions:
    rounds: tuple[int, ...] = (5, 4, 3, 2, 1)
    players: int = 2
    human_seat: int = 0
    first_starter: str = "random"
    model_choice: str = "sample"
    seed: int | None = None
    solver_learn: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.players <= 6:
            raise ValueError("players must be in 2..6")
        if not 0 <= self.human_seat < self.players:
            raise ValueError(f"human_seat must be between 0 and {self.players - 1}")
        if self.first_starter not in ("random", "human", "model"):
            raise ValueError("first_starter must be random, human, or model")
        if self.model_choice not in ("sample", "greedy"):
            raise ValueError("model_choice must be sample or greedy")


def _action_text(phase: Phase, action: int) -> str:
    if phase == Phase.BID:
        return f"puntata {action}"
    if phase == Phase.PLAY:
        return f"carta {action} ({card_name(action)})"
    if phase == Phase.ACE_CHOICE:
        return "asso alto" if action == ACE_HIGH else "asso basso"
    return str(action)


_SUIT_COLORS = {
    "Bastoni": "\033[32m",  # verde
    "Spade": "\033[34m",  # blu
    "Coppe": "\033[31m",  # rosso
    "Denari": "\033[33m",  # giallo
}
_RESET = "\033[0m"


def _card_text(card: int, *, color: bool = False) -> str:
    text = card_name(card)
    if not color:
        return text
    suit = text.rsplit(" di ", 1)[-1]
    return f"{_SUIT_COLORS[suit]}{text}{_RESET}"


def _card_number_text(card: int, *, color: bool = False) -> str:
    """Render the unambiguous card id, optionally colored by suit."""
    number = str(card)
    if not color:
        return number
    suit = card_name(card).rsplit(" di ", 1)[-1]
    return f"{_SUIT_COLORS[suit]}{number}{_RESET}"


def _action_text_colored(phase: Phase, action: int, *, color: bool = False) -> str:
    if phase == Phase.PLAY:
        return _card_number_text(action, color=color)
    if phase == Phase.BID:
        return f"presa {action}"
    return "asso alto" if action == ACE_HIGH else "asso basso"


def _compact_numbers(values: tuple[int, ...] | list[int]) -> str:
    return "/".join("-" if value < 0 else str(value) for value in values)


def _player_label(player: int, human_seat: int) -> str:
    """Use stable human-facing names, ready for more than one opponent."""
    if player == human_seat:
        return "Tu"
    # Number opponents from one, independently of the human seat.
    opponent_number = player + (1 if player < human_seat else 0)
    return f"Avv{opponent_number}"


def _errors_text(errors: tuple[int, ...] | list[int], human_seat: int) -> str:
    return " | ".join(
        f"{_player_label(player, human_seat)}: {value}" for player, value in enumerate(errors)
    )


def _card_numbers(cards: tuple[int, ...] | list[int], *, color: bool = False) -> str:
    """Render card ids only; the id is what the CLI accepts as input."""
    return " ".join(_card_number_text(card, color=color) for card in cards)


def _distribution_text(
    state: RoundState, probabilities: tuple[float, ...], *, color: bool = False
) -> str:
    """Render the model distribution against the actions in this position."""
    actions = state.legal_actions()
    phase = Phase(state.phase)
    parts: list[str] = []
    for action, probability in zip(actions, probabilities):
        label = _card_number_text(action, color=color) if phase == Phase.PLAY else str(action)
        parts.append(f"{label}={probability * 100:.1f}%")
    return " ".join(parts)


def _compact_trick(state: RoundState, *, color: bool = False) -> str:
    if not state.current_trick:
        return "-"
    return " ".join(
        f"G{player}:{_card_text(card, color=color)}" for player, card in state.current_trick
    )


def _trick_order_reference(state: RoundState, human_seat: int, *, color: bool = False) -> str:
    """Render the current trick as a compact clockwise sequence."""
    players = state.config.players
    order = [(state.leader + offset) % players for offset in range(players)]
    played_cards = dict(state.current_trick)
    parts: list[str] = []
    for player in order:
        bid = state.bids[player]
        remaining = "?" if bid < 0 else str(max(0, bid - state.catches[player]))
        segment = f"{_player_label(player, human_seat)} (prese rimaste: {remaining})"
        if player in played_cards:
            segment += " " + _card_number_text(played_cards[player], color=color)
        elif player == state.current_player:
            # The graphic is only shown before the human's card decision, so
            # this is the seat that needs the visual emphasis.
            segment = f"\033[1m{segment}\033[0m" if color else segment
        parts.append(segment)
    return "Presa in corso: " + " --> ".join(parts)


def _blind_round_reference(state: RoundState, human_seat: int, *, color: bool = False) -> str:
    """Show the R1 clockwise order while bids are still being declared."""
    parts: list[str] = []
    order = [
        (state.leader + offset) % state.config.players for offset in range(state.config.players)
    ]
    for player in order:
        if player == human_seat:
            segment = "Tu"
        else:
            # In the blind round the human sees every opponent's card, but
            # not their own until the declaration has been made.
            card = state.hands[player][0]
            segment = f"{_player_label(player, human_seat)} {_card_number_text(card, color=color)}"
            bid = state.bids[player]
            if bid >= 0:
                segment += ' ("prendo")' if bid == 1 else ' ("non prendo")'
        if player == state.current_player and player == human_seat:
            segment = f"\033[1m{segment}\033[0m" if color else segment
        parts.append(segment)
    return "Presa: " + " --> ".join(parts)


def _probabilities(policy: object, state: RoundState) -> tuple[float, ...]:
    state_method = getattr(policy, "probabilities_state", None)
    if callable(state_method):
        values = tuple(float(value) for value in state_method(state))
    else:
        values = tuple(float(value) for value in policy.probabilities(state.information_state()))  # type: ignore[attr-defined]
    actions = state.legal_actions()
    if len(values) != len(actions) or any(value < 0 for value in values):
        raise ValueError("model returned an invalid action distribution")
    total = sum(values)
    if total <= 0:
        raise ValueError("model returned an empty action distribution")
    return tuple(value / total for value in values)


def _choose_action(
    state: RoundState, policy: object, rng: random.Random, mode: str
) -> tuple[int, tuple[float, ...]]:
    actions = state.legal_actions()
    probabilities = _probabilities(policy, state)
    if mode == "greedy":
        index = max(range(len(actions)), key=lambda item: probabilities[item])
    else:
        index = rng.choices(range(len(actions)), weights=probabilities, k=1)[0]
    return actions[index], probabilities


def _ask_human_action(
    state: RoundState,
    input_fn: InputFn,
    output: OutputFn,
    *,
    color: bool = False,
    round_size: int | None = None,
) -> int:
    info = state.information_state()
    phase = Phase(state.phase)
    if phase == Phase.PLAY:
        options = _card_numbers(info.legal_actions, color=color)
        prompt = f"Gioca [{options}] > "
    elif phase == Phase.BID:
        options = " ".join(str(action) for action in info.legal_actions)
        prompt = f"Prese [{options}] > "
    else:
        prompt = "Asso [0=basso 1=alto] > "
    while True:
        try:
            raw = input_fn(prompt).strip().casefold()
        except (EOFError, KeyboardInterrupt):
            output("\nPartita interrotta: input terminato.")
            raise KeyboardInterrupt
        if raw in {"q", "quit", "esci"}:
            output("\nPartita interrotta dall'utente.")
            raise KeyboardInterrupt
        try:
            if phase == Phase.ACE_CHOICE:
                if raw in {"0", "basso", "low"}:
                    action = ACE_LOW
                elif raw in {"1", "alto", "high"}:
                    action = ACE_HIGH
                else:
                    raise ValueError
            elif raw.lstrip("-").isdigit():
                action = int(raw)
            elif phase == Phase.PLAY:
                matches = [
                    action for action in info.legal_actions if card_name(action).casefold() == raw
                ]
                if len(matches) != 1:
                    raise ValueError
                action = matches[0]
            else:
                raise ValueError
            if action not in info.legal_actions:
                raise ValueError
            return action
        except ValueError:
            output("  Mossa non valida: scegli una delle opzioni mostrate.")


class HumanMatch:
    """Run one complete human-versus-model match in a terminal."""

    def __init__(
        self,
        policies: Mapping[int, object] | None,
        options: HumanMatchOptions,
        *,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
        policy_loaders: Mapping[int, PolicyLoader] | None = None,
    ) -> None:
        supplied = dict(policies or {})
        loaders = dict(policy_loaders or {})
        missing = [
            round_size
            for round_size in options.rounds
            if round_size not in supplied and round_size not in loaders
        ]
        if missing:
            raise ValueError(f"missing policy for round(s): {', '.join(map(str, missing))}")
        self.policies = supplied
        self.policy_loaders = loaders
        self.loaded_rounds: set[int] = set()
        self.options = options
        self.input = input_fn
        self.output = output_fn
        # None means a fresh, system-random seed for every match.  Keep the
        # resolved value so the JSON report can reproduce that exact match.
        self.seed = (
            options.seed if options.seed is not None else random.SystemRandom().randrange(0, 2**63)
        )
        self.rng = random.Random(self.seed)
        self.color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    @staticmethod
    def _close_policy(policy: object) -> None:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
            return
        entries = getattr(policy, "entries", None)
        close_entries = getattr(entries, "close", None)
        if callable(close_entries):
            close_entries()

    def close(self) -> None:
        """Release policies loaded for the current or interrupted round."""
        for policy in tuple(self.policies.values()):
            self._close_policy(policy)
        self.policies.clear()

    def _policy_for_round(self, round_size: int) -> object:
        policy = self.policies.get(round_size)
        if policy is not None:
            return policy
        loader = self.policy_loaders.get(round_size)
        if loader is None:
            raise ValueError(f"missing policy for round {round_size}")
        policy = loader()
        self.policies[round_size] = policy
        self.loaded_rounds.add(round_size)
        return policy

    def _initial_starter(self) -> int:
        if self.options.first_starter == "random":
            return self.rng.randrange(self.options.players)
        if self.options.first_starter == "human":
            return self.options.human_seat
        # Pick the first model seat clockwise from the human.  This keeps the
        # old HU behaviour (seat 1 starts) and gives a deterministic meaning
        # to "model" when there are several opponents.
        return (self.options.human_seat + 1) % self.options.players

    def play(self) -> dict[str, object]:
        first_starter = self._initial_starter()
        total_errors = [0] * self.options.players
        round_reports: list[dict[str, object]] = []
        self.output(f"\nPresine {self.options.players}p | q=esci")
        if self.options.solver_learn:
            self.output("Solver learn: ON")
        for index, round_size in enumerate(self.options.rounds):
            # In multiplayer the starter rotates clockwise; with two players
            # this is exactly the historical alternation.
            starter = (first_starter + index) % self.options.players
            config = GameConfig(
                players=self.options.players,
                hand_size=round_size,
                starting_player=starter,
            )
            state = RoundState(config)
            state.apply_chance(state.sample_chance(self.rng))
            policy = self._policy_for_round(round_size)
            self.output(f"\nROUND {round_size}")
            hand_shown = False
            while not state.is_terminal:
                if (
                    self.options.players > 2
                    and state.current_player == self.options.human_seat
                    and Phase(state.phase) == Phase.BID
                    and round_size == 1
                ):
                    self.output(
                        _blind_round_reference(
                            state,
                            self.options.human_seat,
                            color=self.color,
                        )
                    )
                elif (
                    self.options.players > 2
                    and state.current_player == self.options.human_seat
                    and Phase(state.phase) == Phase.PLAY
                ):
                    self.output(
                        _trick_order_reference(
                            state,
                            self.options.human_seat,
                            color=self.color,
                        )
                    )
                if state.current_player == self.options.human_seat:
                    # Mostra la mano una sola volta, prima della prima scelta
                    # umana del round (normalmente la dichiarazione delle prese).
                    if not hand_shown:
                        info = state.information_state()
                        if round_size != 1:
                            self.output(
                                f"Carte: {_card_numbers(info.visible_cards, color=self.color)}"
                            )
                        hand_shown = True
                    previous_catches = tuple(state.catches)
                    phase_before = Phase(state.phase)
                    hidden_own_card = (
                        state.hands[self.options.human_seat][0]
                        if round_size == 1
                        and phase_before == Phase.BID
                        and state.hands[self.options.human_seat]
                        else None
                    )
                    action = _ask_human_action(
                        state,
                        self.input,
                        self.output,
                        color=self.color,
                        round_size=round_size,
                    )
                    if self.options.solver_learn:
                        probabilities = _probabilities(policy, state)
                        self.output(
                            f"Solver: {_distribution_text(state, probabilities, color=self.color)}"
                        )
                    state.apply_action(action)
                    if hidden_own_card is not None:
                        self.output(
                            f"Tua carta: {_card_number_text(hidden_own_card, color=self.color)}"
                        )
                    if round_size != 1 and tuple(state.catches) != previous_catches:
                        winner = next(
                            player
                            for player, (before, after) in enumerate(
                                zip(previous_catches, state.catches)
                            )
                            if after > before
                        )
                        self.output(f"Presa: {_player_label(winner, self.options.human_seat)}")
                else:
                    previous_catches = tuple(state.catches)
                    action, _ = _choose_action(state, policy, self.rng, self.options.model_choice)
                    if round_size != 1:
                        self.output(
                            f"{_player_label(state.current_player, self.options.human_seat)}: "
                            f"{_action_text_colored(Phase(state.phase), action, color=self.color)}"
                        )
                    state.apply_action(action)
                    if round_size != 1 and tuple(state.catches) != previous_catches:
                        winner = next(
                            player
                            for player, (before, after) in enumerate(
                                zip(previous_catches, state.catches)
                            )
                            if after > before
                        )
                        self.output(f"Presa: {_player_label(winner, self.options.human_seat)}")
            assert state.result is not None
            for player, errors in enumerate(state.result.errors):
                total_errors[player] += errors
            round_report = {
                "hand_size": round_size,
                "starting_player": starter,
                "bids": list(state.result.bids),
                "catches": list(state.result.catches),
                "errors": list(state.result.errors),
            }
            stats = getattr(policy, "stats", None)
            if callable(stats):
                round_report["policy_stats"] = stats()
            round_reports.append(round_report)
            self.output(
                f"Fine ROUND {round_size} | "
                f"errori: {_errors_text(state.result.errors, self.options.human_seat)}"
            )
            self.output(
                f"Errori questa partita: {_errors_text(total_errors, self.options.human_seat)}"
            )
            if round_size in self.loaded_rounds:
                self._close_policy(policy)
                self.policies.pop(round_size, None)
                self.loaded_rounds.remove(round_size)
        best_errors = min(total_errors)
        winners = [player for player, value in enumerate(total_errors) if value == best_errors]
        winner = (
            _player_label(winners[0], self.options.human_seat) if len(winners) == 1 else "pareggio"
        )
        errors = ", ".join(
            f"{_player_label(player, self.options.human_seat)} {value}"
            for player, value in enumerate(total_errors)
        )
        self.output(f"\nFINE | winner: {winner} | errori partita: {errors}")
        return {
            # Keep the historical report kind for HU consumers; multiplayer
            # reports are explicitly identified as a separate shape.
            "kind": (
                "human_vs_model_match_v1"
                if self.options.players == 2
                else "human_vs_model_match_multiplayer_v1"
            ),
            "players": self.options.players,
            "rounds": list(self.options.rounds),
            "human_seat": self.options.human_seat,
            "first_starter": first_starter,
            "model_choice": self.options.model_choice,
            "solver_learn": self.options.solver_learn,
            "seed": self.seed,
            "rounds_report": round_reports,
            "total_errors": total_errors,
            "winner": winner,
        }


def load_policy_map(path: Path) -> dict[int, Path]:
    """Load ``{"5": "...", "4": "..."}``, resolving relative paths nearby."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy map must be a JSON object")
    result: dict[int, Path] = {}
    for key, value in payload.items():
        round_size = int(key)
        if not isinstance(value, str):
            raise ValueError(f"policy path for R{round_size} must be a string")
        candidate = Path(value)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = path.parent / candidate
        result[round_size] = candidate
    return result


def wrap_resolver_policies(
    policies: dict[int, object],
    paths: Mapping[int, Path],
    resolver_rounds: set[int],
    *,
    search_config_dir: Path,
    workers: int,
    blueprint_weight: float,
) -> list[object]:
    """Replace selected compact blueprints with state-aware resolving policies."""
    owned: list[object] = []
    for round_size in sorted(resolver_rounds):
        policy = policies.get(round_size)
        if policy is None:
            raise ValueError(f"cannot resolve R{round_size}: policy is missing")
        if not isinstance(policy, LinearBlueprintPolicy):
            raise ValueError(f"resolver mode for R{round_size} requires a blueprint JSON policy")
        config_path = search_config_dir / f"resolver-r{round_size}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"search config not found for R{round_size}: {config_path}")
        game, search, _ = load_search_config(config_path, [f"search.workers={workers}"])
        if game.hand_size != round_size:
            raise ValueError(f"search config {config_path} is not for R{round_size}")
        # With zero weight the production resolver must not retain the compact
        # policy even as an emergency prior: a failed search should fall back
        # uniformly rather than silently reintroducing the old heuristic.
        prior = policy if blueprint_weight > 0 else None
        resolving = ResolvingPolicy(None, prior, search, blueprint_weight=blueprint_weight)
        policies[round_size] = resolving
        owned.append(resolving)
    return owned
