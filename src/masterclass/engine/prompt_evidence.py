from __future__ import annotations

import re
from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage
from masterclass.engine.pitch_spelling import spell_pitch_name

_PITCH_RE = re.compile(r"\b([A-G]#)(-?\d*)\b")


def build_evidence_digest(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    score_key: str | None = None,
) -> str:
    """Build a compact text digest for the teach prompt from deterministic artifacts.

    Missing analysis artifacts are skipped so callers can use this at any point in
    the v2 pipeline without special casing partially analyzed sessions.
    """

    parts: list[str] = []
    first = manifest.metadata.get("first_measure")
    last = manifest.metadata.get("last_measure")
    played = _played_measures(first, last)

    parts.append(f"Repertoire: {manifest.repertoire}")
    parts.append(f"Movement: {manifest.movement}")
    if played:
        parts.append(f"Played measures: {played}")
    parts.append(f"Instrument: {manifest.instrument or manifest.instrument_profile}")
    parts.append("")

    rhythm = _read_json_artifact(storage, store, manifest, "analysis/polyphonic_rhythm.json")
    if rhythm:
        summary = rhythm.get("summary", {}) if isinstance(rhythm, dict) else {}
        rhythm_played = summary.get("played_measures")
        if rhythm_played and not played:
            played = _played_measures_from_value(rhythm_played)
            if played:
                parts.append(f"Played measures: {played}")
        bpm = summary.get("overall_player_quarter_bpm_median", summary.get("overall_tempo_bpm_quarter"))
        bar_duration = summary.get("bar_duration_median_sec")
        music_start = summary.get("music_start_sec")
        parts.append(
            "Tempo: median quarter-note BPM = "
            f"{bpm}, median bar duration = {bar_duration} sec, music starts at {music_start}s."
        )
        parts.append("")

    intonation = _read_json_artifact(storage, store, manifest, "analysis/polyphonic_intonation.json")
    if intonation:
        summary = intonation.get("summary", {}) if isinstance(intonation, dict) else {}
        in_tune = summary.get("high_confidence_count", summary.get("high_confidence_notes"))
        parts.append(
            "Intonation overall: "
            f"median {summary.get('overall_median_cents')}c, "
            f"abs-max {summary.get('overall_abs_max_cents')}c, "
            f"in-tune count {in_tune}."
        )
        spread_lines = []
        pcs = summary.get("by_pitch_class", {})
        if isinstance(pcs, dict):
            for pc, stats in pcs.items():
                if not isinstance(stats, dict):
                    continue
                if int(stats.get("count", 0) or 0) < 4:
                    continue
                spread = _spread(stats)
                pc_spelled = spell_pitch_name(f"{pc}0", score_key)[:-1] if score_key else str(pc)
                spread_lines.append(
                    f"{pc_spelled}: median {_signed(stats.get('median_cents'))}c "
                    f"spread {spread:.0f}c (n={stats.get('count')})"
                )
        if spread_lines:
            parts.append("By pitch class: " + "; ".join(spread_lines))
        parts.append("")

    comments = _read_json_artifact(storage, store, manifest, "analysis/mechanical_comments.json")
    if comments:
        rows = comments.get("comments", []) if isinstance(comments, dict) else comments
        rows = rows if isinstance(rows, list) else []
        parts.append(f"Mechanical pass produced {len(rows)} comments. Highlights:")
        for row in rows:
            if not isinstance(row, dict) or row.get("severity") not in ("warn", "alert"):
                continue
            cid = row.get("id")
            measure = row.get("measure")
            title = row.get("title", row.get("summary", ""))
            parts.append(f"  [{row.get('severity')}] {cid} bar {measure} — {_spell_text(str(title), score_key)}")
        parts.append("")

    return "\n".join(parts).rstrip()


def _read_json_artifact(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, relative_key: str) -> Any | None:
    candidates = []
    key = manifest.artifacts.get(relative_key)
    if key:
        candidates.append(key)
    candidates.append(store.artifact_key(manifest.session, relative_key))
    for candidate in dict.fromkeys(candidates):
        if storage.exists(candidate):
            try:
                return storage.read_json(candidate)
            except (FileNotFoundError, ValueError):
                return None
    return None


def _played_measures(first: Any, last: Any) -> str | None:
    if first is None or last is None:
        return None
    return f"{first}-{last}"


def _played_measures_from_value(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]}-{value[1]}"
    if isinstance(value, dict):
        return _played_measures(value.get("first_measure", value.get("first")), value.get("last_measure", value.get("last")))
    return str(value) if value else None


def _spread(stats: dict[str, Any]) -> float:
    if stats.get("spread_p10_p90") is not None:
        return float(stats.get("spread_p10_p90") or 0.0)
    return float(stats.get("p90", 0.0) or 0.0) - float(stats.get("p10", 0.0) or 0.0)


def _signed(value: Any) -> str:
    try:
        return f"{float(value):+g}"
    except (TypeError, ValueError):
        return str(value)


def _spell_text(text: str, key: str | None) -> str:
    if not key or "#" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        return spell_pitch_name(match.group(1) + match.group(2), key)

    return _PITCH_RE.sub(repl, text)
