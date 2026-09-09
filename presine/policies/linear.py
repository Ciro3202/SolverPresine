"""Policy compatta per HU r2-r5.

Questo modulo trasforma la situazione della partita e ogni azione possibile in
piccoli numeri descrittivi. Poi combina questi numeri con un vettore di pesi e
li trasforma in probabilità. In questo modo il modello ha sempre la stessa
dimensione, anche quando durante il training vengono incontrate moltissime
situazioni diverse.

La parte principale usa fasce di forza delle carte e informazioni relative al
giocatore. Se richiesto, può aggiungere anche alcune posizioni calcolate con
un hash: servono a dare più memoria al modello senza creare una riga per ogni
stato possibile.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from presine.game.cards import ACE_OF_DENARI, NUM_CARDS
from presine.game.config import GameConfig
from presine.game.observation import InformationState
from presine.game.state import Phase

from .abstraction_resolving.canonical import relative_seat
from .compact import CompactBlueprintPolicy

LINEAR_BLUEPRINT_FORMAT = "presine-linear-blueprint-v1"
TILE_BLUEPRINT_FORMAT = "presine-tile-blueprint-v1"

# Questi nomi vengono salvati nel file per capire con quale formato ricaricare
# i pesi e per evitare di scambiare un modello lineare con uno a tile.


@dataclass(frozen=True, slots=True)
class LinearFeatureConfig:
    """Regola quanto è grande e quanto è morbida la policy lineare."""

    # Più fasce distinguono meglio le carte, ma aumentano il numero di pesi.
    strength_buckets: int = 8

    # Una temperatura più alta distribuisce di più la scelta tra le azioni.
    temperature: float = 0.35

    # Zero disattiva le tile; altrimenti indica quante posizioni aggiuntive usare.
    tile_buckets: int = 0

    # The same compact policy can be trained for one fixed player count.  The
    # default keeps every historical heads-up blueprint byte-for-byte
    # compatible while multiplayer runs use relative seat features for 3..6.
    players: int = 2

    # The historical blueprint added a hand-crafted compact prior to every
    # logit.  Neutral refits can disable it while keeping exactly the same
    # feature width and file size.
    compact_prior: bool = True

    def __post_init__(self) -> None:
        """Controlla i pochi valori che possono rendere inutilizzabile il modello."""

        # Limitiamo la dimensione del modello per non trasformare una policy
        # leggera in una tabella nascosta troppo costosa.
        if not 2 <= self.strength_buckets <= 20:
            raise ValueError("strength_buckets must be in 2..20")

        # La temperatura entra nella divisione della softmax e non può essere zero.
        if self.temperature <= 0:
            raise ValueError("linear policy temperature must be positive")

        # Le tile usano una maschera binaria: per questo il numero deve essere
        # una potenza di due e abbastanza grande da ridurre le collisioni.
        if self.tile_buckets and (
            self.tile_buckets < 64 or self.tile_buckets & (self.tile_buckets - 1)
        ):
            raise ValueError("tile_buckets must be zero or a power of two of at least 64")
        if not 2 <= self.players <= 6:
            raise ValueError("players must be in 2..6")


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Trasforma i punteggi delle azioni in probabilità che sommano a uno."""

    # Togliere il punteggio massimo evita numeri enormi dentro l'esponenziale.
    shifted = (logits - float(np.max(logits))) / temperature

    # Il taglio protegge la stabilità numerica quando le preferenze sono molto forti.
    weights = np.exp(np.clip(shifted, -60.0, 60.0))

    # La divisione finale rende i valori direttamente utilizzabili come probabilità.
    return weights / float(np.sum(weights))


class LinearBlueprintPolicy:
    """Policy compatta che assegna un punteggio a ogni azione possibile.

    La policy non conserva una riga per ogni situazione incontrata. Usa invece
    un numero fisso di pesi NumPy, così la memoria resta prevedibile anche se
    il training continua a lungo.
    """

    def __init__(
        self,
        config: LinearFeatureConfig | None = None,
        *,
        weights: np.ndarray | None = None,
    ) -> None:
        # La configurazione decide subito quanti pesi servono al modello.
        self.config = config or LinearFeatureConfig()
        width = self.parameter_count_for(self.config)
        if weights is None:
            # All'inizio i pesi lineari sono neutri: la policy di base continua
            # comunque a proporre azioni sensate.
            self.weights = np.zeros(width, dtype=np.float64)
        else:
            # Quando ricarichiamo un modello, controlliamo che la sua dimensione
            # corrisponda esattamente alla configurazione scelta.
            array = np.asarray(weights, dtype=np.float64)
            if array.shape != (width,):
                raise ValueError(f"linear blueprint expected {width} weights, got {array.shape}")
            self.weights = array.copy()

        # La parte imparata è volutamente correttiva: con pesi tutti a zero la
        # policy di base resta già valida, invece di partire da scelte casuali.
        self.base = CompactBlueprintPolicy()

        # Serve solo per tenere d'occhio quante consultazioni riceve la policy.
        self.calls = 0

    @staticmethod
    def state_width_for(config: LinearFeatureConfig) -> int:
        """Restituisce quanti numeri descrivono la situazione del giocatore."""

        # Ci sono la fase, alcuni riassunti della mano e due piccoli istogrammi
        # per le carte visibili e per quelle ricevute all'inizio.
        return 3 + 5 + 2 * (config.strength_buckets + 1) + 3 * config.players

    @staticmethod
    def action_width_for(config: LinearFeatureConfig) -> int:
        """Restituisce quanti numeri descrivono una singola azione."""

        # Descriviamo la fase, la forza della carta, la posizione tra le mosse
        # possibili e alcuni valori utili per puntate, prese e scelta dell'asso.
        return 3 + (config.strength_buckets + 1) + 3 + 3

    @classmethod
    def base_parameter_count_for(cls, config: LinearFeatureConfig) -> int:
        """Calcola quanti pesi servono alla parte lineare senza tile."""

        state = cls.state_width_for(config)
        action = cls.action_width_for(config)

        # Ogni caratteristica dello stato può combinarsi con ogni caratteristica
        # dell'azione; poi aggiungiamo un termine per l'azione e uno di base.
        return state * action + action + 1

    @classmethod
    def parameter_count_for(cls, config: LinearFeatureConfig) -> int:
        """Calcola tutti i pesi, compresi quelli aggiuntivi delle tile."""

        return cls.base_parameter_count_for(config) + config.tile_buckets

    @property
    def base_parameter_count(self) -> int:
        """Numero dei pesi usati dalla sola parte lineare."""

        return self.base_parameter_count_for(self.config)

    def _bucket(self, card: int) -> int:
        """Raggruppa una carta in una fascia di forza semplice e stabile."""

        # L'asso di denari ha un trattamento speciale nel gioco e resta nella
        # fascia più alta, mentre le altre carte vengono distribuite per intervalli.
        return (
            self.config.strength_buckets
            if card == ACE_OF_DENARI
            else min(
                self.config.strength_buckets - 1,
                card * self.config.strength_buckets // NUM_CARDS,
            )
        )

    def state_features(self, info: InformationState) -> np.ndarray:
        """Crea il piccolo riassunto numerico della situazione corrente."""

        buckets = self.config.strength_buckets
        values = np.zeros(self.state_width_for(self.config), dtype=np.float64)
        phase = Phase(info.phase)

        # Una posizione a uno indica in quale fase della mano ci troviamo.
        values[int(phase) - int(Phase.BID)] = 1.0

        # Questi valori riassumono chi ha iniziato, chi guida e quanto è avanti
        # la presa, usando numeri confrontabili tra i due giocatori.
        offset = 3
        values[offset] = len(info.visible_cards) / max(1, info.hand_size)
        values[offset + 1] = info.trick_index / max(1, info.hand_size)
        values[offset + 2] = float(relative_seat(info.starting_player, info.player))
        values[offset + 3] = float(relative_seat(info.leader, info.player))
        values[offset + 4] = len(info.current_trick) / 2.0
        offset += 5

        # Invece di ricordare ogni carta con un indice enorme, contiamo quante
        # carte appartengono a ciascuna fascia di forza.
        for card in info.visible_cards:
            values[offset + self._bucket(card)] += 1.0 / max(1, info.hand_size)
        offset += buckets + 1

        # Conserviamo anche l'osservazione iniziale: aiuta a non dimenticare
        # informazioni private importanti quando la mano va avanti.
        for card in info.initial_private_observation:
            values[offset + self._bucket(card)] += 1.0 / max(1, info.hand_size)
        offset += buckets + 1

        # Le puntate e le prese vengono lette dal punto di vista del giocatore,
        # così la stessa situazione resta riconoscibile anche cambiando posto.
        # Per il vecchio heads-up manteniamo esattamente l'ordine storico; per
        # il multiplayer aggiungiamo una terna (puntata, prese, mancanti) per
        # ogni posto relativo in senso orario.
        players = len(info.bids)
        if players != self.config.players:
            raise ValueError(
                f"blueprint configured for {self.config.players} players, got {players}"
            )
        scale = max(1, info.hand_size)
        if players == 2:
            order = (info.player, (info.player + 1) % 2)
            own_bid, opponent_bid = (info.bids[index] for index in order)
            own_catches, opponent_catches = (info.catches[index] for index in order)
            values[offset : offset + 6] = (
                own_bid / scale,
                opponent_bid / scale,
                own_catches / scale,
                opponent_catches / scale,
                max(0, own_bid - own_catches) / scale,
                max(0, opponent_bid - opponent_catches) / scale,
            )
        else:
            for relative in range(players):
                seat = (info.player + relative) % players
                bid = info.bids[seat]
                catches = info.catches[seat]
                values[offset + relative * 3 : offset + relative * 3 + 3] = (
                    bid / scale,
                    catches / scale,
                    max(0, bid - catches) / scale,
                )
        return values

    def action_features(self, info: InformationState) -> np.ndarray:
        """Crea il riassunto numerico di ogni azione consentita."""

        actions = info.legal_actions
        width = self.action_width_for(self.config)
        result = np.zeros((len(actions), width), dtype=np.float64)
        phase = Phase(info.phase)

        # Questi spostamenti tengono separate le parti del vettore dedicate a
        # fascia della carta, posizione dell'azione e valori numerici.
        bucket_offset = 3
        position_offset = bucket_offset + self.config.strength_buckets + 1
        scalar_offset = position_offset + 3
        opponent_card = info.current_trick[0][1] if info.current_trick else None

        # Ogni riga descrive una mossa diversa, ma tutte usano lo stesso schema.
        for index, action in enumerate(actions):
            result[index, int(phase) - int(Phase.BID)] = 1.0
            if phase == Phase.PLAY:
                # Nel gioco delle carte contano sia la fascia della carta sia
                # la sua posizione rispetto alle altre mosse possibili.
                result[index, bucket_offset + self._bucket(action)] = 1.0
                position = index / max(1, len(actions) - 1)
                result[index, position_offset + min(2, int(position * 3.0))] = 1.0
                rank = 1.05 if action == ACE_OF_DENARI else action / max(1, NUM_CARDS - 1)
                result[index, scalar_offset] = rank
                if opponent_card is not None:
                    # Durante una presa ricordiamo anche se la carta supera
                    # quella già giocata dall'avversario.
                    opponent_rank = (
                        1.05
                        if opponent_card == ACE_OF_DENARI
                        else opponent_card / max(1, NUM_CARDS - 1)
                    )
                    result[index, scalar_offset + 1] = float(rank > opponent_rank)
            elif phase == Phase.BID:
                # Una puntata viene riportata su una scala compatta e confrontabile.
                result[index, scalar_offset] = action / max(1, info.hand_size)
            else:
                # Nella scelta dell'asso basta distinguere la scelta alta da quella bassa.
                result[index, scalar_offset + 2] = float(action == 1)
        return result

    def design_matrix(self, info: InformationState) -> np.ndarray:
        """Unisce caratteristiche della situazione e delle azioni."""

        state = self.state_features(info)
        actions = self.action_features(info)

        # Il prodotto esterno permette alla policy di imparare che la stessa
        # azione può essere buona o cattiva a seconda della situazione.
        joint = np.einsum("i,aj->aij", state, actions, optimize=True).reshape(len(actions), -1)

        # Aggiungiamo anche le caratteristiche dell'azione da sole e una colonna
        # costante, utile come piccolo spostamento di base del punteggio.
        return np.concatenate((joint, actions, np.ones((len(actions), 1))), axis=1)

    def linear_logits_from_features(
        self,
        state: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Calcola i punteggi lineari senza costruire la matrice completa.

        ``design_matrix`` serve ancora quando vogliamo ispezionare tutte le
        caratteristiche. Nel percorso caldo, però, ci servono solo i punteggi:
        possiamo quindi fare prima il prodotto tra lo stato e i pesi e poi
        confrontare il risultato con tutte le azioni.
        """

        state_width = self.state_width_for(self.config)
        action_width = self.action_width_for(self.config)
        joint_width = state_width * action_width

        # I primi pesi rappresentano le interazioni tra situazione e azione.
        # La forma della matrice segue l'ordine usato da np.einsum in
        # design_matrix, quindi il risultato resta identico.
        interaction_weights = self.weights[:joint_width].reshape(state_width, action_width)

        # Prima riassumiamo l'effetto dello stato sulle caratteristiche dell'azione.
        state_effect = np.asarray(state, dtype=np.float64) @ interaction_weights

        # Poi applichiamo lo stesso vettore a tutte le azioni e aggiungiamo il
        # termine specifico dell'azione e il piccolo termine costante.
        action_weights = self.weights[joint_width : joint_width + action_width]
        return (
            np.asarray(actions, dtype=np.float64) @ (state_effect + action_weights)
            + self.weights[joint_width + action_width]
        )

    def _tile_context(self, info: InformationState) -> tuple[object, ...]:
        """Prepara un riassunto compatto della situazione per le tile.

        Questi dati non diventano una tabella di chiavi. Vengono solo passati
        all'hash per scegliere alcune posizioni fisse, dando al modello un po'
        di memoria specifica senza ricreare una tabella CFR enorme.
        """

        # Qui conserviamo solo informazioni limitate e relative al giocatore,
        # così il numero di tile non dipende dal numero di partite già viste.
        phase = Phase(info.phase)
        own = tuple(sorted(self._bucket(card) for card in info.visible_cards))
        initial = tuple(sorted(self._bucket(card) for card in info.initial_private_observation))
        current = tuple(
            (relative_seat(actor, info.player), self._bucket(card))
            for actor, card in info.current_trick
        )
        players = len(info.bids)
        order = tuple((info.player + relative) % players for relative in range(players))
        bids = tuple(info.bids[index] for index in order)
        catches = tuple(info.catches[index] for index in order)

        # L'ordine è pensato per poter essere usato direttamente dalle diverse
        # combinazioni di tile qui sotto.
        return (
            int(phase),
            len(info.visible_cards),
            info.trick_index,
            relative_seat(info.starting_player, info.player),
            relative_seat(info.leader, info.player),
            bids,
            catches,
            own,
            initial,
            current,
            len(info.legal_actions),
        )

    def _tile_action(self, info: InformationState, action: int, index: int) -> tuple[object, ...]:
        """Prepara la parte dell'hash che descrive l'azione scelta."""

        # La forma dell'azione cambia in base alla fase: una carta, una puntata
        # o la scelta tra i due valori dell'asso.
        phase = Phase(info.phase)
        if phase == Phase.PLAY:
            return ("play", self._bucket(action), index, len(info.legal_actions))
        if phase == Phase.BID:
            return ("bid", action, info.hand_size)
        return ("ace", action)

    def _tile_hash(self, token: object) -> int:
        """Converte una descrizione in una posizione stabile nell'array dei pesi."""

        # ``hash()`` cambia tra un processo Python e l'altro. Un digest stabile
        # fa sì che il modello esportato resti uguale anche dopo il caricamento.
        digest = hashlib.blake2s(repr(token).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") & (self.config.tile_buckets - 1)

    def tile_indices(self, info: InformationState) -> np.ndarray:
        """Restituisce le posizioni delle tile, una riga breve per azione."""

        # Senza tile restituiamo subito una matrice vuota, evitando lavoro inutile
        # nelle configurazioni che usano solo la parte lineare.
        if not self.config.tile_buckets:
            return np.empty((len(info.legal_actions), 0), dtype=np.int64)
        context = self._tile_context(info)
        phase = Phase(info.phase)

        # Ogni firma guarda un'interazione diversa: così una collisione in una
        # posizione non rovina contemporaneamente tutte le informazioni.
        result = np.empty((len(info.legal_actions), 5), dtype=np.int64)
        for index, action in enumerate(info.legal_actions):
            action_token = self._tile_action(info, action, index)
            result[index] = (
                self._tile_hash(("full", context, action_token)),
                self._tile_hash(("hand", context[7], context[8], action_token)),
                self._tile_hash(("score", context[5], context[6], action_token)),
                self._tile_hash(("trick", context[2], context[9], action_token)),
                self._tile_hash(("phase", int(phase), len(info.visible_cards), action_token)),
            )
        return result

    def tile_logits_from_indices(self, indices: np.ndarray) -> np.ndarray:
        """Somma i pesi delle tile associate a ciascuna azione."""

        # Se le tile sono spente, ogni azione riceve un contributo nullo.
        if not indices.shape[1]:
            return np.zeros(indices.shape[0], dtype=np.float64)

        # Le posizioni vengono usate come indici nella parte finale del vettore
        # dei pesi, separata dai parametri della parte lineare.
        return np.sum(self.weights[self.base_parameter_count :][indices], axis=1)

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        """Restituisce le probabilità finali delle azioni consentite."""

        self.calls += 1

        # Usiamo il calcolo diretto: evita di creare una matrice completa che
        # servirebbe solo per ottenere un vettore di punteggi.
        state = self.state_features(info)
        actions = self.action_features(info)
        logits = self.linear_logits_from_features(state, actions)

        # Quando le tile sono spente non allochiamo nemmeno il vettore vuoto
        # del loro contributo: è il caso predefinito e più frequente.
        if self.config.tile_buckets:
            logits += self.tile_logits_from_indices(self.tile_indices(info))

        # La policy di base mantiene preferenze sensate anche con pesi iniziali
        # nulli e si somma ai contributi imparati.
        if self.config.compact_prior:
            logits += self.base_logits(info)
        return tuple(float(value) for value in _softmax(logits, self.config.temperature))

    def base_logits(self, info: InformationState) -> np.ndarray:
        """Converte le probabilità della policy di base in punteggi sommabili."""

        if not self.config.compact_prior:
            return np.zeros(len(info.legal_actions), dtype=np.float64)
        # Il logaritmo trasforma il prodotto implicito delle preferenze in una
        # somma, che si combina bene con il contributo dei pesi lineari.
        base = np.asarray(self.base.probabilities(info), dtype=np.float64)
        return np.log(np.maximum(base, 1e-12))

    def lookup(self, info: InformationState) -> tuple[float, ...]:
        """Interfaccia compatibile con le altre policy del progetto."""

        return self.probabilities(info)

    def copy(self) -> LinearBlueprintPolicy:
        """Crea una policy separata con gli stessi pesi e configurazione."""

        # La copia dei pesi evita che l'addestramento di un oggetto modifichi
        # accidentalmente la policy originale.
        return LinearBlueprintPolicy(self.config, weights=self.weights)

    def stats(self) -> dict[str, object]:
        """Restituisce numeri utili per controllare dimensione e utilizzo."""

        return {
            "kind": "linear_feature_blueprint",
            "stored_state_rows": 0,
            "parameters": int(self.weights.size),
            "tile_parameters": self.config.tile_buckets,
            "base": "compact_feature_blueprint",
            "calls": self.calls,
            "config": asdict(self.config),
        }


def save_linear_blueprint(
    game: GameConfig, policy: LinearBlueprintPolicy, path: Path
) -> dict[str, object]:
    """Salva configurazione e pesi in un file piccolo e ricaricabile."""

    # Il JSON resta leggibile; quando ci sono molte tile, i pesi vengono invece
    # compressi in byte float32 codificati in base64.
    payload = {
        "format": TILE_BLUEPRINT_FORMAT if policy.config.tile_buckets else LINEAR_BLUEPRINT_FORMAT,
        "game": game.to_dict(),
        "config": asdict(policy.config),
    }
    if policy.config.tile_buckets:
        payload["weights_f32_base64"] = base64.b64encode(
            policy.weights.astype("<f4", copy=False).tobytes()
        ).decode("ascii")
    else:
        payload["weights"] = policy.weights.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {
        "format": payload["format"],
        "parameters": int(policy.weights.size),
        "tile_parameters": policy.config.tile_buckets,
        "stored_state_rows": 0,
        "path": str(path.resolve()),
    }


def load_linear_blueprint(path: Path) -> tuple[GameConfig, LinearBlueprintPolicy]:
    """Ricarica dal file una configurazione di gioco e la relativa policy."""

    # Il formato viene controllato prima di interpretare i pesi, così un file
    # di un modello diverso non viene caricato in modo silenziosamente errato.
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") not in (LINEAR_BLUEPRINT_FORMAT, TILE_BLUEPRINT_FORMAT):
        raise ValueError("unsupported linear blueprint format")
    config = LinearFeatureConfig(**dict(payload.get("config", {})))
    if "weights_f32_base64" in payload:
        # Le tile usano float32 nel file per ridurre spazio e tempo di lettura;
        # nei calcoli torniamo a float64 per mantenere il comportamento attuale.
        raw = base64.b64decode(str(payload["weights_f32_base64"]))
        weights = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        # Il formato lineare semplice conserva invece la lista JSON dei pesi.
        weights = np.asarray(payload["weights"], dtype=np.float64)
    return GameConfig.from_dict(dict(payload["game"])), LinearBlueprintPolicy(
        config, weights=weights
    )
