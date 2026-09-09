"""Policy che cerca una situazione in una tabella di probabilità.

Quando trova la situazione richiesta, restituisce le probabilità già imparate.
Quando invece quella situazione non è presente, usa una distribuzione uniforme
tra le mosse consentite: è una scelta semplice e sicura per non bloccare il
resto del programma.

Il modulo è volutamente piccolo. Il lavoro più costoso, cioè costruire la
chiave della situazione, viene svolto da :class:`InformationState`; la chiave
ora viene conservata dopo la prima richiesta per evitare ricostruzioni inutili.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from presine.game.observation import InformationState
from presine.reporting import format_stats as format_project_stats

# La chiave descrive tutto ciò che il giocatore può sapere della situazione.
InformationKey: TypeAlias = tuple[object, ...]

# Ogni voce collega l'ordine delle azioni alle rispettive probabilità.
PolicyEntry: TypeAlias = tuple[tuple[int, ...], tuple[float, ...]]


class TabularPolicy:
    """Cerca esattamente una situazione e usa una scelta uniforme se manca.

    ``entries`` può essere un normale dizionario oppure una struttura che legge
    i dati da disco. Per questo la policy usa solo l'interfaccia ``Mapping`` e
    non assume che tutte le voci siano già in memoria.
    """

    def __init__(self, entries: Mapping[InformationKey, PolicyEntry]) -> None:
        # Conserviamo il contenitore così com'è: può essere grande o caricato
        # a pezzi, quindi non lo trasformiamo automaticamente in un dizionario.
        self.entries = entries

        # Questi contatori servono per capire quanto spesso la tabella copre
        # davvero le situazioni incontrate.
        self._exact_lookups = 0
        self._uniform_fallbacks = 0

        # Qui teniamo lo stesso riepilogo separato per giocatore e fase.
        self._by_seat_phase: dict[str, dict[str, int]] = {}

    def reset_stats(self) -> None:
        """Azzera i contatori senza toccare le probabilità salvate."""

        # È utile prima di una nuova sessione di valutazione o di una partita.
        self._exact_lookups = 0
        self._uniform_fallbacks = 0
        self._by_seat_phase.clear()

    def _record(self, info: InformationState, result: str) -> None:
        """Registra se la ricerca è stata precisa o ha usato il ripiego uniforme."""

        # La stringa rende il riepilogo leggibile anche quando viene mostrato
        # direttamente dalla riga di comando.
        label = f"seat_{info.player}/phase_{info.phase}"
        counters = self._by_seat_phase.setdefault(label, {"exact": 0, "uniform_fallback": 0})
        counters[result] += 1

        # Aggiorniamo anche i totali, così stats() non deve scorrere la cronologia.
        if result == "exact":
            self._exact_lookups += 1
        else:
            self._uniform_fallbacks += 1

    @staticmethod
    def _rates(exact: int, fallback: int) -> dict[str, float]:
        """Calcola le percentuali senza dividere per zero."""

        total = exact + fallback
        return {
            "exact_rate": exact / total if total else 0.0,
            "fallback_rate": fallback / total if total else 0.0,
        }

    def stats(self) -> dict[str, object]:
        """Restituisce un riepilogo dell'uso della tabella."""

        # I totali sono già mantenuti durante i lookup; qui li impacchettiamo
        # soltanto in una forma comoda da salvare o mostrare.
        total = self._exact_lookups + self._uniform_fallbacks
        return {
            "lookups": total,
            "exact_lookups": self._exact_lookups,
            "uniform_fallbacks": self._uniform_fallbacks,
            **self._rates(self._exact_lookups, self._uniform_fallbacks),
            "by_seat_phase": {
                label: {
                    **counts,
                    **self._rates(counts["exact"], counts["uniform_fallback"]),
                }
                for label, counts in self._by_seat_phase.items()
            },
        }

    def format_stats(self) -> str:
        """Prepara le statistiche con il formatter comune a tutto il progetto."""

        # La policy conserva questo piccolo collegamento per compatibilità;
        # la logica vera vive nel modulo reporting e vale per ogni risultato.
        return format_project_stats(self.stats(), title="Tabular policy - lookup statistics")

    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        """Restituisce una probabilità per ogni azione consentita."""

        # InformationState conserva la chiave dopo la prima costruzione, quindi
        # ripetere questo lookup non ricrea tutta la cronologia pubblica.
        key = info.key()
        entry = self.entries.get(key)
        if entry is None:
            # Una situazione mai vista non deve lasciare il chiamante senza
            # risposta: distribuiamo il peso in modo uguale tra le mosse legali.
            self._record(info, "uniform_fallback")
            return (1.0 / len(info.legal_actions),) * len(info.legal_actions)

        actions, probabilities = entry

        # L'ordine delle azioni è importante: una probabilità deve riferirsi
        # esattamente alla mossa per cui è stata imparata.
        if actions != info.legal_actions:
            raise ValueError("tabular policy action order differs from the game")

        # Questi controlli proteggono da file corrotti o strategie costruite male.
        # Sono il principale costo extra del lookup esatto.
        if len(probabilities) != len(actions):
            raise ValueError("tabular policy probability vector has the wrong length")
        if any(value < 0.0 for value in probabilities):
            raise ValueError("tabular policy contains a negative probability")
        if abs(sum(probabilities) - 1.0) > 1e-9:
            raise ValueError("tabular policy probabilities do not sum to one")

        # La voce è valida e può essere restituita senza creare una copia.
        self._record(info, "exact")
        return probabilities

    def probability_by_action(self, info: InformationState) -> dict[int, float]:
        """Restituisce le probabilità associate alle sole azioni consentite."""

        # Questo formato è comodo per chi vuole cercare direttamente per codice
        # dell'azione; per il training è più leggero usare probabilities().
        return dict(zip(info.legal_actions, self.probabilities(info)))
