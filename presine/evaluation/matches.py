from __future__ import annotations

import math
import random

# Questo modulo serve a giocare molte partite simulate e a trasformare i
# risultati in numeri utili per confrontare due policy.
# Non aggiorna le policy: misura soltanto quanto bene si comportano.
from dataclasses import dataclass

from presine.game.config import GameConfig
from presine.game.state import RoundState
from presine.policies.base import Policy


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    # Qui raccogliamo tutte le statistiche principali di un confronto.
    # Tenerle insieme rende più semplice riutilizzarle sia nel codice sia nella CLI.
    games: int
    mean_margin: float
    ci95: tuple[float, float]
    wins: int
    ties: int
    losses: int
    by_seat: tuple[float, float]
    ci_unit: str = "game"

    def to_dict(self) -> dict[str, object]:
        # Il dizionario è comodo per JSON, log e report senza dover conoscere
        # direttamente i dettagli della dataclass.
        return {
            # Numero totale di partite effettivamente giocate.
            "games": self.games,
            # Vantaggio medio: positivo per il candidato, negativo per l'avversario.
            "mean_margin": self.mean_margin,
            # Intervallo entro cui stimiamo il vero vantaggio medio con confidenza 95%.
            "ci95": self.ci95,
            # Specifica se l'intervallo è calcolato per partita o per coppia abbinata.
            "ci_unit": self.ci_unit,
            # Conteggi grezzi, utili per capire subito l'esito del confronto.
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            # Rapporti già pronti, così chi legge il risultato non deve ricalcolarli.
            "win_rate": self.wins / self.games,
            "tie_rate": self.ties / self.games,
            "loss_rate": self.losses / self.games,
            # Media ottenuta quando il candidato occupa ciascuno dei due posti.
            "by_seat": self.by_seat,
        }


def play_round(
    config: GameConfig,
    policies: tuple[Policy, Policy],
    *,
    seed: int,
    deal=None,
) -> tuple[int, object]:
    # Usiamo un generatore locale: lo stesso seed rende la singola partita
    # ripetibile, cosa utile per test, confronto e riproduzione degli errori.
    rng = random.Random(seed)

    # Lo stato parte vuoto e riceve prima la distribuzione delle carte.
    state = RoundState(config)
    state.apply_chance(state.sample_chance(rng) if deal is None else deal)

    # Cerchiamo una sola volta il metodo piu diretto di ciascuna policy.
    # Durante la partita il giocatore cambia, ma le policy restano le stesse.
    state_methods = tuple(getattr(policy, "probabilities_state", None) for policy in policies)

    # Facciamo avanzare la partita finché non sono state raccolte tutte le mosse.
    while not state.is_terminal:
        # La policy può decidere usando lo stato completo oppure la sola
        # informazione visibile al giocatore, se non offre il metodo più diretto.
        info = state.information_state()
        player = state.current_player
        policy = policies[player]
        state_method = state_methods[player]
        probabilities = (
            state_method(state) if state_method is not None else policy.probabilities(info)
        )

        # Una policy deve restituire una probabilità per ogni mossa possibile.
        # Controllarlo qui evita risultati silenziosamente sbagliati più avanti.
        if len(probabilities) != len(info.legal_actions):
            raise ValueError("policy returned the wrong probability vector")

        # Estraiamo una mossa rispettando le probabilità prodotte dalla policy.
        action = rng.choices(info.legal_actions, weights=probabilities, k=1)[0]

        # La mossa arriva direttamente dall'elenco appena creato sopra.
        # Possiamo quindi saltare il secondo controllo e la copia per l'undo:
        # questa partita viene giocata fino alla fine e poi lo stato viene scartato.
        state.apply_action_unchecked(action)

    # A questo punto il risultato deve esistere: l'assert segnala eventuali
    # incoerenze nello stato durante lo sviluppo.
    assert state.result is not None

    # Il margine è espresso dal punto di vista del giocatore 1.
    margin = state.result.errors[1] - state.result.errors[0]
    return margin, state.result


def evaluate_round(
    config: GameConfig,
    candidate: Policy,
    opponent: Policy,
    *,
    games: int,
    seed: int,
    rotate_seats: bool = True,
    rotate_starting_player: bool = True,
) -> EvaluationResult:
    # Servono almeno una partita per poter calcolare statistiche sensate.
    if games <= 0:
        raise ValueError("games must be positive")

    # Queste liste e contatori raccolgono sia il risultato complessivo sia
    # l'effetto del posto occupato dal candidato.
    margins: list[float] = []
    seat_sums = [0.0, 0.0]
    seat_counts = [0, 0]
    wins = ties = losses = 0

    # Con i posti ruotati, due partite consecutive fanno parte della stessa
    # coppia e condividono configurazione e affare. Li prepariamo una sola volta.
    prepared_pair_index = -1

    # Ogni giro simula una partita; quando richiesto, due partite consecutive
    # usano lo stesso affare ma scambiano i posti dei giocatori.
    for index in range(games):
        # Il numero della coppia evita di consumare due volte la stessa
        # sequenza di casualità quando i posti vengono ruotati.
        pair_index = index // 2 if rotate_seats else index

        if pair_index != prepared_pair_index:
            # Cambiare il giocatore iniziale riduce il rischio che il confronto
            # favorisca sempre lo stesso ordine di gioco.
            starting_player = (
                (config.starting_player + pair_index) % config.players
                if rotate_starting_player
                else config.starting_player
            )
            game_config = GameConfig(
                players=config.players,
                hand_size=config.hand_size,
                starting_player=starting_player,
                payoff=config.payoff,
            )

            # Prepariamo l'affare una sola volta. Nella seconda partita della
            # coppia lo riutilizziamo, cambiando soltanto i posti delle policy.
            game_seed = seed + pair_index * 104_729
            deal_state = RoundState(game_config)
            deal = deal_state.sample_chance(random.Random(game_seed))
            prepared_pair_index = pair_index

        # In una partita il candidato parte dal posto 0, in quella abbinata
        # dal posto 1: così il confronto è più equo rispetto al posto.
        candidate_seat = index % 2 if rotate_seats else 0
        policies = (candidate, opponent) if candidate_seat == 0 else (opponent, candidate)

        # play_round restituisce il margine dal punto di vista del posto 0.
        # Se il candidato è al posto 1, invertiamo il segno per mantenere
        # sempre la stessa lettura: positivo significa meglio per il candidato.
        raw_margin, _ = play_round(game_config, policies, seed=game_seed + 1, deal=deal)
        margin = raw_margin if candidate_seat == 0 else -raw_margin
        margins.append(float(margin))

        # Conserviamo anche la media separata per posto, utile per individuare
        # un eventuale vantaggio strutturale del primo o del secondo giocatore.
        seat_sums[candidate_seat] += margin
        seat_counts[candidate_seat] += 1

        # Classifichiamo il margine per ottenere vittorie, pareggi e sconfitte.
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1

    # Il margine medio è la misura sintetica principale del confronto.
    mean = sum(margins) / games

    # Quando due partite condividono lo stesso affare non sono osservazioni
    # completamente indipendenti: per l'incertezza le trattiamo come una
    # singola coppia, evitando di dare troppo peso a quei dati.
    paired_ci = rotate_seats and games % 2 == 0
    observations = (
        [sum(margins[index : index + 2]) / 2.0 for index in range(0, games, 2)]
        if paired_ci
        else margins
    )

    # Stimiamo la variabilità dei risultati tra partite (o tra coppie abbinate).
    variance = (
        sum((value - mean) ** 2 for value in observations) / (len(observations) - 1)
        if len(observations) > 1
        else 0.0
    )

    # 1.96 è il fattore usato per costruire un intervallo approssimativo al 95%.
    half_width = 1.96 * math.sqrt(variance / len(observations))

    # Se un posto non è mai stato usato, lasciamo un valore non numerico invece
    # di inventare una media: succede, ad esempio, con una valutazione fissa.
    by_seat = tuple(
        seat_sums[seat] / seat_counts[seat] if seat_counts[seat] else math.nan for seat in range(2)
    )

    # Restituiamo un oggetto unico e leggibile, pronto per log, JSON o CLI.
    return EvaluationResult(
        games=games,
        mean_margin=mean,
        ci95=(mean - half_width, mean + half_width),
        wins=wins,
        ties=ties,
        losses=losses,
        by_seat=by_seat,
        ci_unit="seat_swapped_deal_pair" if paired_ci else "game",
    )


def evaluate_fixed_seats(
    config: GameConfig,
    policies: tuple[Policy, Policy],
    *,
    games: int,
    seed: int,
    rotate_starting_player: bool = True,
) -> tuple[float, float]:
    # Questa variante mantiene i giocatori sempre negli stessi posti.
    # È utile quando serve misurare separatamente il rendimento dei due posti.
    totals = [0.0, 0.0]

    # Giochiamo più partite cambiando, se richiesto, il giocatore che comincia.
    for index in range(games):
        starting_player = (
            (config.starting_player + index) % config.players
            if rotate_starting_player
            else config.starting_player
        )
        game_config = GameConfig(
            players=config.players,
            hand_size=config.hand_size,
            starting_player=starting_player,
            payoff=config.payoff,
        )

        # Il seed cambia a ogni partita, ma resta riproducibile dato il seed base.
        margin, _ = play_round(game_config, policies, seed=seed + index * 104_729)

        # Normalizziamo gli errori rispetto alla dimensione della mano, così
        # il numero è confrontabile anche tra configurazioni diverse.
        totals[0] += margin / config.hand_size
        totals[1] -= margin / config.hand_size

    # Il primo valore riguarda il posto 0 e il secondo il posto 1.
    return totals[0] / games, totals[1] / games
