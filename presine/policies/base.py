"""Definisce la forma comune che tutte le strategie devono rispettare.

Questo file non contiene una strategia concreta e non decide quale mossa
fare. Dice soltanto al resto del programma come parlare con una policy:
le si passa la situazione vista dal giocatore e si ricevono le probabilità
delle azioni possibili.

Tenere questa regola piccola rende facile provare strategie diverse durante
il training senza dover cambiare il motore della partita.
"""

from __future__ import annotations

from typing import Protocol

from presine.game.observation import InformationState


class Policy(Protocol):
    """Descrive il comportamento minimo richiesto a una strategia.

    Può essere rispettato da una policy casuale, tabellare, neurale o da una
    strategia di fallback. Non serve ereditare esplicitamente da questa classe:
    è sufficiente avere il metodo con la stessa forma.
    """

    # La policy riceve solo ciò che il giocatore può sapere della partita.
    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        """Restituisce una probabilità per ciascuna azione consentita."""

        ...
