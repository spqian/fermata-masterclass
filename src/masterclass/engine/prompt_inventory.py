from __future__ import annotations

from typing import Any


def build_score_note_inventory(
    score_map: dict,
    *,
    first_measure: int | None = None,
    last_measure: int | None = None,
) -> str:
    """Compact bar-by-bar score note inventory for note_id citations.

    This intentionally mirrors the PoC teach.py _build_score_inventory output.
    """

    played_lo = _measure_bound(first_measure, score_map, ("first_measure", "played_lo"), default=None)
    played_hi = _measure_bound(last_measure, score_map, ("last_measure", "played_hi"), default=None)

    lines = []
    notes_by_bar: dict[int, list[dict[str, Any]]] = {}
    for n in score_map.get("notes", []):
        b = int(n.get("midi_measure", -1))
        if (played_lo is None or b >= played_lo) and (played_hi is None or b <= played_hi):
            notes_by_bar.setdefault(b, []).append(n)
    bar_meta = {int(b["midi_measure"]): b for b in score_map.get("bars", [])}
    for b in sorted(notes_by_bar):
        meta = bar_meta.get(b, {})
        ps = meta.get("perf_start_sec")
        pe = meta.get("perf_end_sec")
        sys_no = meta.get("system_on_page") or meta.get("system")
        page = meta.get("page", 1)
        lines.append(f"## Bar {b} — page {page} system {sys_no} — perf {ps}-{pe} sec")
        for n in sorted(notes_by_bar[b], key=lambda x: x.get("score_time", 0)):
            names = "+".join(n.get("names") or [])
            tag = " [chord]" if n.get("is_chord") else ""
            lines.append(f"  - `{n['note_id']}` beat {n.get('beat_in_bar')} : {names}{tag} — perf_time={n.get('perf_time')}s")
        lines.append("")
    return "\n".join(lines)


def _measure_bound(value: int | None, score_map: dict, keys: tuple[str, ...], *, default: int | None) -> int | None:
    if value is not None:
        return int(value)
    for key in keys:
        candidate = score_map.get(key)
        if candidate is not None:
            return int(candidate)
    return default
