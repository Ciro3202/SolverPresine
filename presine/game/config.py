"""Configurazione delle regole e dei parametri principali della partita.

Questo modulo definisce :class:`GameConfig`, una configurazione immutabile
usata dal motore di gioco per descrivere una partita: numero di giocatori,
dimensione delle mani, giocatore iniziale e metodo di calcolo del payoff.
La classe valida i valori ricevuti alla creazione dell'oggetto e offre anche
metodi per convertire la configurazione in un dizionario o ricostruirla da un
dizionario. In questo modo la stessa configurazione può essere usata sia dal
motore di gioco sia dalle componenti di addestramento e di ricerca.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Descrive le regole di una singola partita addestrata separatamente.

    Il motore di gioco tiene conto del numero di giocatori. I metodi di
    apprendimento e di ricerca effettuano i propri controlli di compatibilità,
    invece di dichiarare in modo implicito che ogni configurazione è supportata.
    """

    # Parametri principali della partita. Questi valori predefiniti descrivono
    # una partita standard a due giocatori con cinque carte ciascuno.
    players: int = 2
    hand_size: int = 5
    starting_player: int = 0
    payoff: str = "error_margin"

    def __post_init__(self) -> None:
        """Verifica che i parametri descrivano una partita valida."""

        # I controlli impediscono di creare configurazioni incompatibili con
        # le regole e con il mazzo utilizzato dal gioco.
        if not 2 <= self.players <= 6:
            raise ValueError("players must be in 2..6")
        if not 1 <= self.hand_size <= 5:
            raise ValueError("hand_size must be in 1..5")
        if self.players * self.hand_size > 40:
            raise ValueError("the requested deal contains more than 40 cards")
        if not 0 <= self.starting_player < self.players:
            raise ValueError("starting_player is out of range")
        if self.payoff != "error_margin":
            raise ValueError("only the separable error_margin payoff is implemented")

    def to_dict(self) -> dict[str, object]:
        """Restituisce la configurazione in una forma facilmente serializzabile."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> GameConfig:
        """Crea una configurazione a partire da un dizionario di parametri."""

        return cls(**values)
