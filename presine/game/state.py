"""Tiene sotto controllo tutto quello che succede in una mano di Presine.

Qui vengono distribuite le carte, raccolte le dichiarazioni, giocate le
prese, gestito l'asso di denari e calcolato il risultato finale. ``RoundState``
conserva sia ciò che tutti possono vedere sia ciò che appartiene al singolo
giocatore, così il training può usare esattamente le informazioni corrette.

La mano può anche tornare indietro: le operazioni normali conservano una
fotografia dello stato, mentre le varianti rapide usano copie più piccole o
nessuna copia quando il percorso viene usato una sola volta dal training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import IntEnum

from .cards import ACE_HIGH, ACE_LOW, ACE_OF_DENARI, NUM_CARDS, card_strength
from .config import GameConfig
from .observation import EventKind, InformationState, PublicEvent


class Phase(IntEnum):
    """I diversi momenti in cui può trovarsi una mano."""

    CHANCE = 0
    BID = 1
    PLAY = 2
    ACE_CHOICE = 3
    TERMINAL = 4


Deal = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Riassume come è finita la mano."""

    bids: tuple[int, ...]
    catches: tuple[int, ...]
    errors: tuple[int, ...]


@dataclass(slots=True)
class UndoToken:
    """Conserva abbastanza informazioni per tornare alla situazione di prima."""

    phase: Phase
    current_player: int
    leader: int
    trick_index: int
    bids: tuple[int, ...]
    catches: tuple[int, ...]
    hands: tuple[tuple[int, ...], ...]
    initial_observations: tuple[tuple[int, ...], ...]
    current_trick: tuple[tuple[int, int], ...]
    public_history: tuple[PublicEvent, ...]
    ace_high: bool | None
    result: RoundResult | None


@dataclass(slots=True)
class FastUndoToken:
    """Versione più piccola del ricordo usato durante il training.

    In una mano normale non cambiano né le carte iniziali né tutte le
    informazioni private. Conserviamo quindi solo ciò che può davvero
    cambiare, così il training fa meno copie.
    """

    phase: Phase
    current_player: int
    leader: int
    trick_index: int
    bids: tuple[int, ...]
    catches: tuple[int, ...]
    actor: int
    actor_hand: tuple[int, ...]
    current_trick: tuple[tuple[int, int], ...]
    public_history_length: int
    ace_high: bool | None
    result: RoundResult | None


class RoundState:
    """Rappresenta una mano di Presine da due a sei giocatori.

    Qui si trovano le carte, le dichiarazioni, le prese, il turno e la storia
    pubblica. Il training può provare molte possibilità e poi tornare indietro
    senza dover ricominciare ogni volta da capo.
    """

    # Questi sono tutti i pezzi che descrivono una mano. Tenerli fissi rende
    # ogni nuovo stato più piccolo, cosa utile quando il training ne crea molti.
    __slots__ = (
        "_bid_actions",
        "ace_high",
        "bids",
        "catches",
        "config",
        "current_player",
        "current_trick",
        "debug",
        "hands",
        "initial_observations",
        "leader",
        "phase",
        "public_history",
        "result",
        "trick_index",
    )

    def __init__(self, config: GameConfig, *, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        self.phase = Phase.CHANCE
        self.current_player = config.starting_player
        self.leader = config.starting_player
        self.trick_index = 0
        self.bids = [-1] * config.players
        self.catches = [0] * config.players
        self.hands: list[list[int]] = [[] for _ in range(config.players)]
        self.initial_observations: list[tuple[int, ...]] = [() for _ in range(config.players)]
        self.current_trick: list[tuple[int, int]] = []
        self.public_history: list[PublicEvent] = []
        self.ace_high: bool | None = None
        self.result: RoundResult | None = None
        # Le dichiarazioni possibili dipendono solo dal numero di carte: le
        # prepariamo una volta e poi le riutilizziamo durante tutta la mano.
        self._bid_actions = tuple(range(config.hand_size + 1))

    @property
    def is_terminal(self) -> bool:
        return self.phase == Phase.TERMINAL

    def sample_chance(self, rng: random.Random) -> Deal:
        # All'inizio non è ancora successo nulla: qui scegliamo e distribuiamo
        # le carte in modo casuale, lasciando ogni mano ordinata.
        if self.phase != Phase.CHANCE:
            raise ValueError("chance can only be sampled at the root")
        count = self.config.players * self.config.hand_size
        cards = rng.sample(range(NUM_CARDS), count)
        n = self.config.hand_size
        return tuple(
            tuple(sorted(cards[player * n : (player + 1) * n]))
            for player in range(self.config.players)
        )

    def chance_outcome_count(self) -> int:
        # Questo numero serve a sapere quante distribuzioni diverse sono
        # possibili, senza doverle costruire tutte una per una.
        remaining = NUM_CARDS
        count = 1
        for _ in range(self.config.players):
            count *= math.comb(remaining, self.config.hand_size)
            remaining -= self.config.hand_size
        return count

    def apply_chance(self, deal: Deal) -> UndoToken:
        # La distribuzione normale controlla le carte e conserva una copia,
        # perché chi usa questa funzione potrebbe voler tornare indietro.
        if self.phase != Phase.CHANCE:
            raise ValueError("deal can only be applied at chance")
        self._validate_deal(deal)
        token = self._capture()
        self._set_chance(deal)
        if self.debug:
            self.validate()
        return token

    def apply_chance_fast(self, deal: Deal) -> None:
        """Applica una distribuzione valida senza conservare una copia.

        Controlla comunque le carte, ma non può essere annullata. Va bene
        quando lo stato viene usato fino alla fine e poi buttato via.
        """
        if self.phase != Phase.CHANCE:
            raise ValueError("deal can only be applied at chance")
        self._validate_deal(deal)
        self._set_chance(deal)
        if self.debug:
            self.validate()

    def apply_chance_fast_unchecked(self, deal: Deal) -> None:
        """Applica una distribuzione già sicura senza fare controlli ripetuti.

        Usarla solo con carte generate dal programma stesso. Per dati ricevuti
        dall'esterno restano disponibili :meth:`apply_chance` e
        :meth:`apply_chance_fast`.
        """
        if self.phase != Phase.CHANCE:
            raise ValueError("deal can only be applied at chance")
        # Qui risparmiamo sia il controllo già fatto sia la copia che servirebbe
        # solo per tornare indietro in una mano che stiamo per scartare.
        self._set_chance(deal)
        if self.debug:
            self.validate()

    def legal_actions(self) -> tuple[int, ...]:
        """Dice quali mosse sono possibili in questo momento."""

        # Prima delle prese si dichiarano il numero di carte che si pensa di
        # vincere; nell'ultimo turno può esserci una dichiarazione vietata.
        if self.phase == Phase.BID:
            actions = self._bid_actions
            if (
                self.config.hand_size > 1
                and sum(bid >= 0 for bid in self.bids) == self.config.players - 1
            ):
                forbidden = self.config.hand_size - sum(bid for bid in self.bids if bid >= 0)
                if forbidden in actions:
                    # Il caso vincolato è raro rispetto alla restituzione del
                    # tuple completo, quindi filtriamo solo quando necessario.
                    return tuple(action for action in actions if action != forbidden)
            return actions
        if self.phase == Phase.PLAY:
            # Durante il gioco delle carte, le mosse sono semplicemente le
            # carte ancora presenti nella mano del giocatore di turno.
            return tuple(self.hands[self.current_player])
        if self.phase == Phase.ACE_CHOICE:
            # L'asso di denari lascia sempre le due scelte previste dal gioco.
            return (ACE_LOW, ACE_HIGH)
        # A mano finita, o prima della distribuzione, non si può giocare nulla.
        return ()

    def information_state(self, player: int | None = None) -> InformationState:
        """Prepara ciò che un giocatore può vedere della situazione corrente."""

        if self.phase not in (Phase.BID, Phase.PLAY, Phase.ACE_CHOICE):
            raise ValueError("information states exist only at decision nodes")
        actor = self.current_player if player is None else player
        if not 0 <= actor < self.config.players:
            raise ValueError("player is out of range")
        # Nella mano cieca si vede la propria osservazione iniziale; nelle altre
        # mani si guardano le carte che il giocatore ha ancora in mano.
        visible = (
            self.initial_observations[actor]
            if self.config.hand_size == 1
            else tuple(self.hands[actor])
        )
        return InformationState(
            player=actor,
            phase=int(self.phase),
            hand_size=self.config.hand_size,
            starting_player=self.config.starting_player,
            leader=self.leader,
            trick_index=self.trick_index,
            bids=tuple(self.bids),
            catches=tuple(self.catches),
            visible_cards=visible,
            initial_private_observation=self.initial_observations[actor],
            current_trick=tuple(self.current_trick),
            public_history=tuple(self.public_history),
            legal_actions=self.legal_actions() if actor == self.current_player else (),
        )

    def solver_information(self) -> tuple[int, tuple[object, ...], tuple[int, ...]]:
        """Restituisce le informazioni del giocatore senza creare un oggetto extra.

        Il training usa questi dati per riconoscere situazioni uguali. Evitiamo
        di costruire una struttura intermedia che verrebbe subito abbandonata.
        """
        if self.phase not in (Phase.BID, Phase.PLAY, Phase.ACE_CHOICE):
            raise ValueError("information states exist only at decision nodes")
        actor = self.current_player
        # Da qui in poi raccogliamo solo le informazioni necessarie per il
        # giocatore che deve muovere, senza mostrare carte nascoste altrui.
        actions = self.legal_actions()
        visible = (
            self.initial_observations[actor]
            if self.config.hand_size == 1
            else tuple(self.hands[actor])
        )
        key = (
            actor,
            int(self.phase),
            self.config.hand_size,
            self.config.starting_player,
            self.leader,
            self.trick_index,
            tuple(self.bids),
            tuple(self.catches),
            visible,
            self.initial_observations[actor],
            tuple(self.current_trick),
            tuple((int(event.kind), event.actor, event.value) for event in self.public_history),
            actions,
        )
        return actor, key, actions

    def apply_action(self, action: int) -> UndoToken:
        # Questa è la strada completa e sicura: controlliamo la mossa e poi
        # conserviamo tutto il necessario per poterla annullare.
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}; legal={self.legal_actions()}")
        token = self._capture()
        if self.phase == Phase.BID:
            self._apply_bid(action)
        elif self.phase == Phase.PLAY:
            self._apply_card(action)
        elif self.phase == Phase.ACE_CHOICE:
            self._apply_ace_choice(action)
        else:
            raise ValueError("not a decision node")
        if self.debug:
            self.validate()
        return token

    def apply_action_fast(self, action: int) -> FastUndoToken:
        """Applica una mossa normale conservando solo una copia piccola.

        È utile nel training: nella maggior parte delle mosse non cambiano le
        carte iniziali, quindi non serve conservare una fotografia completa.
        """
        if self.config.hand_size <= 1:
            raise ValueError("fast actions are not supported for the blind round")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}; legal={self.legal_actions()}")
        return self.apply_action_fast_unchecked(action)

    def apply_action_fast_unchecked(self, action: int) -> FastUndoToken:
        """Applica una mossa già controllata usando una copia piccola.

        Il chiamante ha appena ottenuto la mossa dall'elenco delle mosse
        possibili, quindi rifare lo stesso controllo a ogni passo sarebbe solo
        lavoro ripetuto.
        """
        if self.config.hand_size <= 1:
            raise ValueError("fast actions are not supported for the blind round")
        actor = self.current_player
        token = FastUndoToken(
            phase=self.phase,
            current_player=self.current_player,
            leader=self.leader,
            trick_index=self.trick_index,
            bids=tuple(self.bids),
            catches=tuple(self.catches),
            actor=actor,
            actor_hand=tuple(self.hands[actor]) if actor >= 0 else (),
            current_trick=tuple(self.current_trick),
            public_history_length=len(self.public_history),
            ace_high=self.ace_high,
            result=self.result,
        )
        if self.phase == Phase.BID:
            self._apply_bid(action)
        elif self.phase == Phase.PLAY:
            self._apply_card(action)
        elif self.phase == Phase.ACE_CHOICE:
            self._apply_ace_choice(action)
        else:
            raise ValueError("not a decision phase")
        if self.debug:
            self.validate()
        return token

    def apply_action_solver(self, action: int) -> tuple[object, ...]:
        """Applica una mossa già sicura con il ricordo più piccolo possibile."""
        # Questa variante è pensata per il training più intenso: conserva solo
        # i dati che servono davvero per ripristinare il passo appena fatto.
        if self.config.hand_size <= 1:
            raise ValueError("solver actions are not supported for the blind round")
        actor = self.current_player
        history_length = len(self.public_history)
        if self.phase == Phase.BID:
            token: tuple[object, ...] = (
                int(Phase.BID),
                actor,
                self.bids[actor],
                history_length,
            )
            self._apply_bid(action)
        elif self.phase == Phase.PLAY:
            token = (
                int(Phase.PLAY),
                self.current_player,
                self.leader,
                self.trick_index,
                tuple(self.catches),
                actor,
                tuple(self.hands[actor]),
                tuple(self.current_trick),
                history_length,
                self.ace_high,
                self.result,
            )
            self._apply_card(action)
        elif self.phase == Phase.ACE_CHOICE:
            token = (
                int(Phase.ACE_CHOICE),
                self.current_player,
                self.leader,
                self.trick_index,
                tuple(self.catches),
                tuple(self.current_trick),
                history_length,
                self.ace_high,
                self.result,
            )
            self._apply_ace_choice(action)
        else:
            raise ValueError("not a decision phase")
        if self.debug:
            self.validate()
        return token

    def undo_solver(self, token: tuple[object, ...]) -> None:
        """Riporta la mano alla situazione precedente alla mossa del solver."""
        phase = Phase(int(token[0]))
        if phase == Phase.BID:
            # Per una dichiarazione basta rimettere il vecchio valore e togliere
            # gli eventi pubblici aggiunti dopo quel momento.
            _, actor, old_bid, history_length = token
            actor = int(actor)
            self.phase = Phase.BID
            self.current_player = actor
            self.bids[actor] = int(old_bid)
            del self.public_history[int(history_length) :]
            return
        if phase == Phase.PLAY:
            # Dopo una carta ripristiniamo la mano del giocatore, la presa e i
            # contatori che possono essere cambiati durante la mossa.
            (
                _,
                current_player,
                leader,
                trick_index,
                catches,
                actor,
                actor_hand,
                current_trick,
                history_length,
                ace_high,
                result,
            ) = token
            self.phase = Phase.PLAY
            self.current_player = int(current_player)
            self.leader = int(leader)
            self.trick_index = int(trick_index)
            self.catches = list(catches)  # type: ignore[arg-type]
            self.hands[int(actor)] = list(actor_hand)  # type: ignore[arg-type]
            self.current_trick = list(current_trick)  # type: ignore[arg-type]
            del self.public_history[int(history_length) :]
            self.ace_high = ace_high  # type: ignore[assignment]
            self.result = result  # type: ignore[assignment]
            return
        (
            _,
            current_player,
            leader,
            trick_index,
            catches,
            current_trick,
            history_length,
            ace_high,
            result,
        ) = token
        self.phase = Phase.ACE_CHOICE
        self.current_player = int(current_player)
        self.leader = int(leader)
        self.trick_index = int(trick_index)
        self.catches = list(catches)  # type: ignore[arg-type]
        self.current_trick = list(current_trick)  # type: ignore[arg-type]
        del self.public_history[int(history_length) :]
        self.ace_high = ace_high  # type: ignore[assignment]
        self.result = result  # type: ignore[assignment]

    def apply_action_unchecked(self, action: int) -> None:
        """Applica una mossa sicura senza controllarla e senza creare copie.

        Va usata solo quando la mossa è stata appena letta dallo stesso stato.
        Nella ricerca si lavora su copie che verranno comunque buttate, quindi
        conservare un ricordo a ogni ramo sarebbe tempo sprecato.
        """
        if self.phase == Phase.BID:
            self._apply_bid(action)
        elif self.phase == Phase.PLAY:
            self._apply_card(action)
        elif self.phase == Phase.ACE_CHOICE:
            self._apply_ace_choice(action)
        else:
            raise ValueError("not a decision phase")
        if self.debug:
            self.validate()

    def undo_fast(self, token: FastUndoToken) -> None:
        """Ripristina la copia piccola salvata prima della mossa."""
        # Rimettiamo solo ciò che può essere cambiato in una mossa normale.
        self.phase = token.phase
        self.current_player = token.current_player
        self.leader = token.leader
        self.trick_index = token.trick_index
        self.bids = list(token.bids)
        self.catches = list(token.catches)
        if token.actor >= 0:
            self.hands[token.actor] = list(token.actor_hand)
        self.current_trick = list(token.current_trick)
        del self.public_history[token.public_history_length :]
        self.ace_high = token.ace_high
        self.result = token.result

    def undo(self, token: UndoToken) -> None:
        """Ripristina la copia completa restituita da :meth:`apply_action`."""
        # Questa strada è più pesante, ma è quella più generale e più facile da
        # usare quando non si può fare alcuna supposizione sul cambiamento.

        self.phase = token.phase
        self.current_player = token.current_player
        self.leader = token.leader
        self.trick_index = token.trick_index
        self.bids = list(token.bids)
        self.catches = list(token.catches)
        self.hands = [list(hand) for hand in token.hands]
        self.initial_observations = list(token.initial_observations)
        self.current_trick = list(token.current_trick)
        self.public_history = list(token.public_history)
        self.ace_high = token.ace_high
        self.result = token.result

    def utilities(self) -> tuple[float, ...]:
        """Restituisce il punteggio finale, riportato su una scala confrontabile."""

        if not self.is_terminal or self.result is None:
            raise ValueError("utilities exist only at terminal states")
        scale = float(self.config.hand_size)
        errors = self.result.errors
        if self.config.players == 2:
            # In due giocatori il risultato di uno è esattamente l'opposto
            # dell'altro: questa è la forma più semplice e più pulita.
            value = (errors[1] - errors[0]) / scale
            return value, -value
        # Con più giocatori usiamo un confronto medio contro gli altri. La
        # classifica completa delle cinque mani viene valutata altrove.
        # Constant-sum per-round learning surrogate: each player maximises the
        # error advantage over the other independent players.  The actual
        # five-round competition is evaluated separately by minimum total
        # error and split tie credit.
        return tuple(
            (
                sum(errors[other] for other in range(self.config.players) if other != player)
                / (self.config.players - 1)
                - errors[player]
            )
            / scale
            for player in range(self.config.players)
        )

    def state_key(self) -> tuple[object, ...]:
        # Questa chiave descrive proprio tutto lo stato e serve quando occorre
        # confrontare due posizioni senza correre il rischio di perdere dettagli.
        return (
            int(self.phase),
            self.current_player,
            self.leader,
            self.trick_index,
            tuple(self.bids),
            tuple(self.catches),
            tuple(tuple(hand) for hand in self.hands),
            tuple(self.initial_observations),
            tuple(self.current_trick),
            tuple(self.public_history),
            self.ace_high,
            self.result,
        )

    def validate(self) -> None:
        """Controlla che la situazione corrente abbia ancora senso.

        È un controllo utile mentre si sviluppa, ma nel training normale resta
        spento perché controllare tutto a ogni mossa costerebbe tempo.
        """

        # Prima raccogliamo le carte attive e quelle già giocate, così possiamo
        # scoprire duplicati o valori impossibili.
        cards = [card for hand in self.hands for card in hand]
        cards.extend(card for _, card in self.current_trick)
        played = [event.value for event in self.public_history if event.kind == EventKind.PLAY]
        if len(played) != len(set(played)):
            raise AssertionError("a card was played twice")
        if any(card < 0 or card >= NUM_CARDS for card in cards):
            raise AssertionError("card out of range")
        if len(cards) != len(set(cards)):
            raise AssertionError("duplicate active card")
        if self.phase == Phase.PLAY and self.config.hand_size == 1:
            raise AssertionError("the blind round has no play decisions")
        if sum(self.catches) != self.trick_index:
            raise AssertionError("catch count does not match completed tricks")

    def _apply_bid(self, bid: int) -> None:
        """Registra una dichiarazione e passa al gioco quando sono complete."""

        # La dichiarazione diventa subito pubblica e poi si passa al giocatore
        # successivo, finché tutti non hanno parlato.
        actor = self.current_player
        self.bids[actor] = bid
        self.public_history.append(PublicEvent(EventKind.BID, actor, bid))
        if any(value < 0 for value in self.bids):
            self.current_player = (actor + 1) % self.config.players
            return
        if self.config.hand_size == 1:
            # Nella mano cieca le carte vengono scoperte tutte insieme e la
            # presa viene risolta senza una normale fase di gioco.
            ordered = tuple(
                (
                    (self.leader + offset) % self.config.players,
                    self.hands[(self.leader + offset) % self.config.players][0],
                )
                for offset in range(self.config.players)
            )
            self.hands = [[] for _ in range(self.config.players)]
            self.current_trick = list(ordered)
            for player, card in ordered:
                self.public_history.append(PublicEvent(EventKind.PLAY, player, card))
            ace_owner = next((player for player, card in ordered if card == ACE_OF_DENARI), None)
            if ace_owner is None:
                self._resolve_trick()
            else:
                self.phase = Phase.ACE_CHOICE
                self.current_player = ace_owner
            return
        self.phase = Phase.PLAY
        self.current_player = self.leader

    def _apply_card(self, card: int) -> None:
        """Registra una carta giocata e gestisce l'eventuale asso di denari."""

        # Togliamo la carta dalla mano, la mettiamo nella presa e la rendiamo
        # visibile nella storia pubblica.
        actor = self.current_player
        self.hands[actor].remove(card)
        self.current_trick.append((actor, card))
        self.public_history.append(PublicEvent(EventKind.PLAY, actor, card))
        if card == ACE_OF_DENARI:
            # L'asso ferma momentaneamente la presa: prima bisogna scegliere
            # se in questa mano vale basso oppure alto.
            self.phase = Phase.ACE_CHOICE
            self.current_player = actor
            return
        self._continue_after_card(actor)

    def _apply_ace_choice(self, choice: int) -> None:
        """Registra se l'asso di denari vale basso o alto."""

        # La scelta dell'asso chiude la pausa e permette di continuare la presa.
        actor = self.current_player
        self.ace_high = choice == ACE_HIGH
        self.public_history.append(PublicEvent(EventKind.ACE_CHOICE, actor, choice))
        self.phase = Phase.PLAY
        self._continue_after_card(actor)

    def _continue_after_card(self, actor: int) -> None:
        # Se non tutti hanno giocato, il turno gira; altrimenti si decide chi
        # ha vinto la presa appena completata.
        if len(self.current_trick) < self.config.players:
            self.current_player = (actor + 1) % self.config.players
        else:
            self._resolve_trick()

    def _resolve_trick(self) -> None:
        """Determina il vincitore della presa e prepara quella successiva."""

        # Il vincitore è la carta più forte secondo il valore corrente dell'asso.
        winner = max(
            self.current_trick,
            key=lambda item: card_strength(item[1], self.ace_high),
        )[0]
        self.catches[winner] += 1
        self.public_history.append(PublicEvent(EventKind.TRICK_WINNER, winner, winner))
        self.current_trick = []
        self.ace_high = None
        self.trick_index += 1
        self.leader = winner
        self.current_player = winner
        if self.trick_index == self.config.hand_size:
            # Quando abbiamo completato tutte le prese, la mano può terminare.
            self._finish()
        else:
            # Altrimenti si ricomincia una nuova presa guidata dal vincitore.
            self.phase = Phase.PLAY

    def _finish(self) -> None:
        """Calcola gli errori finali e porta la mano allo stato terminale."""

        # L'errore è la distanza tra quante prese erano state dichiarate e
        # quante ne sono state effettivamente vinte.
        errors = tuple(
            abs(self.bids[player] - self.catches[player]) for player in range(self.config.players)
        )
        self.result = RoundResult(tuple(self.bids), tuple(self.catches), errors)
        self.phase = Phase.TERMINAL
        self.current_player = -1

    def _validate_deal(self, deal: Deal) -> None:
        # Prima di iniziare controlliamo che il deal abbia mani della misura
        # giusta e che nessuna carta compaia due volte.
        if len(deal) != self.config.players:
            raise ValueError("deal has the wrong number of hands")
        if any(len(hand) != self.config.hand_size for hand in deal):
            raise ValueError("deal has the wrong hand size")
        cards = [card for hand in deal for card in hand]
        if len(cards) != len(set(cards)):
            raise ValueError("deal contains duplicate cards")
        if any(type(card) is not int or not 0 <= card < NUM_CARDS for card in cards):
            raise ValueError("deal contains an invalid card")

    def _set_chance(self, deal: Deal) -> None:
        """Imposta le strutture iniziali dopo che il deal è stato accettato."""
        # Da qui parte davvero la mano: copiamo le carte e prepariamo ciò che
        # ogni giocatore potrà vedere dall'inizio.
        self.hands = [list(hand) for hand in deal]
        if self.config.hand_size == 1:
            # Nella mano cieca ogni giocatore conserva la vista delle carte
            # altrui, perché le carte proprie vengono subito scoperte.
            self.initial_observations = [
                tuple(
                    card
                    for offset in range(1, self.config.players)
                    for card in deal[(player + offset) % self.config.players]
                )
                for player in range(self.config.players)
            ]
        else:
            # Nelle mani normali la propria mano iniziale resta privata.
            self.initial_observations = [tuple(hand) for hand in deal]
        self.phase = Phase.BID
        self.current_player = self.config.starting_player

    def _capture(self) -> UndoToken:
        # Qui facciamo la fotografia completa richiesta dalla versione normale
        # dell'undo; le varianti rapide evitano di passare da questo punto.
        return UndoToken(
            phase=self.phase,
            current_player=self.current_player,
            leader=self.leader,
            trick_index=self.trick_index,
            bids=tuple(self.bids),
            catches=tuple(self.catches),
            hands=tuple(tuple(hand) for hand in self.hands),
            initial_observations=tuple(self.initial_observations),
            current_trick=tuple(self.current_trick),
            public_history=tuple(self.public_history),
            ace_high=self.ace_high,
            result=self.result,
        )
