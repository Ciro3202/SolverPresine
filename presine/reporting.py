"""Presenta le statistiche del progetto in una forma leggibile.

I moduli di gioco e di training continuano a produrre dizionari semplici,
perché sono il formato più comodo da salvare in JSON. Qui quei dizionari
possono essere trasformati in testo ordinato, senza parentesi graffe e senza
aggiungere librerie esterne.

L'idea è tenere separati i dati dal modo in cui vengono mostrati: una policy,
un trainer o una ricerca possono quindi usare lo stesso strumento.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_LABELS = {
    "by_seat_phase": "By seat / phase",
    "exact_lookups": "Exact",
    "uniform_fallbacks": "Fallback",
    "stored_state_rows": "Stored state rows",
    "tile_parameters": "Tile parameters",
    "estimated_table_bytes": "Estimated table bytes",
    "nodes_per_second": "Nodes per second",
    "elapsed_seconds": "Elapsed seconds",
    "information_states": "Information states",
}

_PHASE_NAMES = {1: "Bid", 2: "Play", 3: "Ace choice"}


def format_stats(stats: Mapping[str, object], *, title: str = "Statistics") -> str:
    """Formatta un dizionario di statistiche come testo leggibile.

    I nomi tecnici già usati dal progetto restano riconoscibili, mentre le
    strutture annidate vengono mostrate come sezioni. Le statistiche tabellari
    ricevono in più una piccola tabella per giocatore e fase.
    """

    # Il titolo separa chiaramente un risultato dall'altro quando più report
    # vengono stampati nella stessa sessione.
    lines = [title, "-" * len(title)]
    _append_mapping(lines, stats, "")
    return "\n".join(lines)


def _append_mapping(lines: list[str], mapping: Mapping[str, object], indent: str) -> None:
    # Le statistiche della policy tabellare hanno una forma ricorrente e si
    # leggono meglio con un riepilogo compatto invece che campo per campo.
    if _is_tabular_lookup(mapping):
        _append_tabular_lookup(lines, mapping, indent)
        return

    for key, value in mapping.items():
        label = _label(str(key))
        if isinstance(value, Mapping):
            lines.append(f"{indent}{label}:")
            _append_mapping(lines, value, indent + "  ")
        else:
            lines.append(f"{indent}{label:<24}: {_value_text(str(key), value)}")


def _is_tabular_lookup(mapping: Mapping[str, object]) -> bool:
    return all(key in mapping for key in ("lookups", "exact_lookups", "uniform_fallbacks"))


def _append_tabular_lookup(lines: list[str], mapping: Mapping[str, object], indent: str) -> None:
    # Qui mostriamo prima il colpo d'occhio generale: quante richieste sono
    # state coperte e quante hanno dovuto usare il fallback.
    exact = int(mapping["exact_lookups"])
    fallback = int(mapping["uniform_fallbacks"])
    lookups = int(mapping["lookups"])
    exact_rate = exact / lookups * 100.0 if lookups else 0.0
    fallback_rate = fallback / lookups * 100.0 if lookups else 0.0
    lines.extend(
        [
            f"{indent}{'Lookups':<24}: {lookups:,}",
            f"{indent}{'Exact':<24}: {exact:,} ({exact_rate:5.1f}%)",
            f"{indent}{'Fallback':<24}: {fallback:,} ({fallback_rate:5.1f}%)",
        ]
    )

    detail = mapping.get("by_seat_phase")
    if isinstance(detail, Mapping) and detail:
        _append_seat_phase_table(lines, detail, indent)


def _append_seat_phase_table(lines: list[str], detail: Mapping[str, object], indent: str) -> None:
    # Le chiavi interne restano del tipo seat_0/phase_1, ma a video diventano
    # una tabella più naturale da leggere.
    rows: list[tuple[int, int, Mapping[str, object]]] = []
    for label, value in detail.items():
        if not isinstance(value, Mapping):
            return
        try:
            seat_text, phase_text = str(label).split("/")
            seat = int(seat_text.removeprefix("seat_"))
            phase = int(phase_text.removeprefix("phase_"))
        except (ValueError, TypeError):
            return
        if "exact" not in value or "uniform_fallback" not in value:
            return
        rows.append((seat, phase, value))

    lines.extend(
        [
            "",
            f"{indent}By seat / phase",
            f"{indent}Seat  Phase        Lookups   Exact   Fallback   Coverage",
            f"{indent}----  -----------  --------  ------  ---------  ---------",
        ]
    )
    for seat, phase, counts in sorted(rows):
        exact = int(counts["exact"])
        fallback = int(counts["uniform_fallback"])
        lookups = exact + fallback
        coverage = exact / lookups * 100.0 if lookups else 0.0
        lines.append(
            f"{indent}{seat:>4}  {_PHASE_NAMES.get(phase, str(phase)):<11}  "
            f"{lookups:>8,}  {exact:>6,}  {fallback:>9,}  {coverage:>8.1f}%"
        )


def _label(key: str) -> str:
    # Per le chiavi nuove usiamo automaticamente parole separate, mentre le
    # più importanti hanno un nome scelto apposta per l'output.
    return _LABELS.get(key, key.replace("_", " ").capitalize())


def _value_text(key: str, value: object) -> str:
    # Le percentuali diventano subito leggibili; 0.93 è molto meno immediato
    # di 93.0% quando si guarda un report a colpo d'occhio.
    if isinstance(value, (int, float)) and key.endswith(("_rate", "_ratio")):
        return f"{float(value) * 100.0:.1f}%"
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)
