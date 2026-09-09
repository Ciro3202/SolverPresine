"""Raccoglie ciò che un giocatore può sapere durante una mano.

Questo modulo contiene i piccoli oggetti usati per descrivere la parte
pubblica della partita e la situazione vista da un singolo giocatore.
Servono soprattutto al training: il programma può confrontare situazioni
uguali e imparare quale azione scegliere senza dover conservare l'intera
partita in memoria.

Qui non vengono fatte mosse e non vengono calcolati risultati. Il modulo si
limita a mettere le informazioni in una forma ordinata, immutabile e facile
da usare come chiave di una tabella.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class EventKind(IntEnum):
    """Indica che tipo di informazione pubblica è stata registrata."""

    # Durante la dichiarazione si vede quanto ogni giocatore vuole prendere.
    BID = 0

    # Durante il gioco si vede quale carta è stata giocata.
    PLAY = 1

    # A fine presa si può sapere chi ha raccolto le carte.
    TRICK_WINNER = 2

    # In questo caso il giocatore ha dovuto scegliere quale asso tenere.
    ACE_CHOICE = 3


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """Una singola informazione che tutti i giocatori possono vedere.

    ``actor`` dice chi ha compiuto l'azione, mentre ``value`` contiene il
    numero o il codice collegato all'evento. Tenere gli eventi piccoli aiuta
    il training perché la cronologia viene copiata molto spesso.
    """

    # Il tipo permette di capire come leggere il valore dell'evento.
    kind: EventKind

    # È il giocatore che ha prodotto questa informazione.
    actor: int

    # È il dato associato all'evento: per esempio una carta o una puntata.
    value: int


@dataclass(frozen=True, slots=True)
class InformationState:
    """Fotografia della situazione vista dal giocatore indicato.

    Contiene solo ciò che il giocatore può usare per decidere. Le carte
    nascoste agli altri non vengono messe qui: in questo modo il modello non
    può imparare informazioni che durante una partita reale non avrebbe.

    L'oggetto è immutabile e leggero perché viene creato molte volte durante
    le simulazioni. Il metodo :meth:`key` lo trasforma nella forma usata per
    cercare o aggiornare la relativa strategia.
    """

    # Chi deve scegliere la prossima azione.
    player: int

    # In quale momento della mano ci troviamo: puntate, gioco o scelta asso.
    phase: int

    # Quante carte restano ancora in mano a ogni giocatore.
    hand_size: int

    # Chi ha iniziato la mano; serve a distinguere situazioni altrimenti simili.
    starting_player: int

    # Chi guida la presa in corso.
    leader: int

    # Il numero della presa che si sta giocando.
    trick_index: int

    # Le puntate già dichiarate, nell'ordine in cui sono arrivate.
    bids: tuple[int, ...]

    # Quante prese ha già raccolto ciascun giocatore.
    catches: tuple[int, ...]

    # Le carte che questo giocatore può vedere in questo momento.
    visible_cards: tuple[int, ...]

    # Le informazioni private ricevute all'inizio e conservate per tutta la mano.
    initial_private_observation: tuple[int, ...]

    # Le carte già calate nella presa che non è ancora finita.
    current_trick: tuple[tuple[int, int], ...]

    # Tutto ciò che è successo e che può essere visto da tutti.
    public_history: tuple[PublicEvent, ...]

    # Le mosse tra cui il giocatore può scegliere adesso.
    legal_actions: tuple[int, ...]

    # La chiave viene costruita solo alla prima richiesta e poi riutilizzata.
    # Non fa parte dell'osservazione: è soltanto un piccolo aiuto per il training.
    _key_cache: tuple[object, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def key(self) -> tuple[object, ...]:
        """Restituisce una chiave completa per la strategia del giocatore.

        Due fotografie producono la stessa chiave solo quando, dal punto di
        vista del giocatore, sono davvero indistinguibili. La cronologia viene
        convertita in numeri semplici così la chiave resta stabile e facile da
        confrontare durante il training.
        """

        # La situazione è immutabile: se la chiave esiste già, non ricostruiamo
        # inutilmente tutta la cronologia a ogni consultazione.
        cached = self._key_cache
        if cached is not None:
            return cached

        # Qui raccogliamo i dati che possono cambiare la scelta da fare.
        key = (
            self.player,
            self.phase,
            self.hand_size,
            self.starting_player,
            self.leader,
            self.trick_index,
            self.bids,
            self.catches,
            self.visible_cards,
            self.initial_private_observation,
            self.current_trick,
            # Gli eventi diventano tuple semplici, senza oggetti personalizzati.
            tuple((int(event.kind), event.actor, event.value) for event in self.public_history),
            # Anche l'insieme delle mosse disponibili fa parte della situazione.
            self.legal_actions,
        )

        # L'oggetto è frozen, quindi usiamo questa sola assegnazione interna per
        # ricordare il risultato senza renderlo modificabile dal chiamante.
        object.__setattr__(self, "_key_cache", key)
        return key
