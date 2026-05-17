from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.engine.instruments import intonation_enabled_for_profile, load_instrument_profile
from masterclass.storage.base import ObjectStorage


@dataclass(frozen=True)
class MechanicalCommentsConfig:
    intonation_warn_cents: float = 15.0
    intonation_alert_cents: float = 25.0
    rubato_info_pct: float = 15.0
    rubato_warn_pct: float = 30.0
    rubato_alert_pct: float = 50.0
    min_intonation_measure_count: int = 4
    min_pitch_class_count: int = 5
    pitch_class_warn_cents: float = 10.0
    pitch_class_spread_warn_cents: float = 50.0
    pitch_class_alert_cents: float = 20.0
    pitch_class_spread_alert_cents: float = 70.0
    max_off_pulse_per_bar: int = 2
    weak_melody_margin_db: float = 3.0
    buried_melody_margin_db: float = 0.0
    attack_spread_warn_ms: float = 80.0
    attack_spread_alert_ms: float = 150.0
    pedal_blur_warn_db: float = -8.0
    max_voicing_events: int = 12
    include_overview: bool = True


@dataclass
class MechanicalCommentsResult:
    comments: list[dict[str, Any]]
    summary: dict[str, Any]
    markdown: str
    config: dict[str, Any]


def generate_mechanical_comments(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    config: MechanicalCommentsConfig | None = None,
) -> MechanicalCommentsResult:
    """Generate deterministic, time-anchored baseline comments from analysis artifacts."""

    config = config or MechanicalCommentsConfig()
    profile = load_instrument_profile(manifest.instrument_profile)
    intonation_enabled = intonation_enabled_for_profile(profile)
    piano_family = _is_piano_family(manifest.instrument_profile, profile)

    rhythm = _read_required_json(storage, store, manifest, "analysis/polyphonic_rhythm.json")
    intonation = _read_optional_json(storage, store, manifest, "analysis/polyphonic_intonation.json") if intonation_enabled else None
    voicing = _read_optional_json(storage, store, manifest, "analysis/piano_voicing.json") if piano_family else None

    comments: list[dict[str, Any]] = []
    next_id = 1

    def add(
        start: float,
        end: float,
        category: str,
        severity: str,
        title: str,
        message: str,
        *,
        measure: int | None = None,
        beat: float | None = None,
        evidence_ref: str = "",
        note_refs: list[Any] | None = None,
    ) -> None:
        nonlocal next_id
        comments.append(
            {
                "id": f"c{next_id:03d}",
                "start_sec": round(max(0.0, float(start)), 3),
                "end_sec": round(max(float(start), float(end)), 3),
                "category": category,
                "severity": severity,
                "title": title,
                "message": message,
                "measure": measure,
                "beat": beat,
                "evidence_ref": evidence_ref,
                "note_refs": list(note_refs) if note_refs else [],
            }
        )
        next_id += 1

    rhythm_summary = rhythm.get("summary", {}) if isinstance(rhythm, dict) else {}
    bar_rows = _bar_rows(rhythm)
    bar_starts = _bar_starts(rhythm_summary, bar_rows)
    bar_durations = _bar_durations(rhythm_summary, bar_rows)

    if intonation:
        _add_intonation_comments(add, intonation, bar_starts, bar_durations, config)
    _add_rhythm_comments(add, rhythm, rhythm_summary, bar_starts, bar_durations, config)
    if voicing:
        _add_voicing_comments(add, voicing, bar_starts, bar_durations, config)
    if config.include_overview:
        _add_overview(add, manifest, rhythm_summary, intonation, voicing, bar_starts)

    comments.sort(key=lambda c: (float(c["start_sec"]), c["id"]))
    # Defense in depth: enforce the played-range sandbox even if an upstream
    # source (rhythm/intonation/voicing) leaked a row whose measure is
    # outside the lesson envelope. Rows without an explicit measure (e.g.
    # overview/global) pass through unchanged.
    from masterclass.core.played_range import derive_played_range
    played_range = derive_played_range(manifest, None)
    comments = [
        c for c in comments
        if c.get("measure") is None or played_range.contains(c.get("measure"))
    ]
    counts = _counts(comments)
    summary = {
        "session_id": manifest.session.session_id,
        "repertoire": manifest.repertoire,
        "movement": manifest.movement,
        "instrument_profile": manifest.instrument_profile,
        "comment_count": len(comments),
        "count_by_severity": counts,
        "source_artifacts": {
            "rhythm": _artifact_key(storage, store, manifest, "analysis/polyphonic_rhythm.json"),
            "intonation": _artifact_key(storage, store, manifest, "analysis/polyphonic_intonation.json") if intonation else None,
            "voicing": _artifact_key(storage, store, manifest, "analysis/piano_voicing.json") if voicing else None,
        },
    }
    return MechanicalCommentsResult(
        comments=comments,
        summary=summary,
        markdown=_render_markdown(summary, comments),
        config=asdict(config),
    )


def persist_mechanical_comments(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: MechanicalCommentsResult,
) -> None:
    """Persist baseline mechanical comments and stamp the manifest."""

    json_key = store.artifact_key(manifest.session, "analysis/mechanical_comments.json")
    md_key = store.artifact_key(manifest.session, "analysis/mechanical_comments.md")
    storage.write_json(json_key, result.comments)
    storage.write_bytes(md_key, result.markdown.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/mechanical_comments.json"] = json_key
    manifest.artifacts["analysis/mechanical_comments.md"] = md_key
    manifest.metadata["mechanical_comments_summary"] = result.summary
    store.save(manifest)


def _add_intonation_comments(add, intonation: dict[str, Any], bar_starts: dict[int, float], bar_durations: dict[int, float], config: MechanicalCommentsConfig) -> None:
    summary = intonation.get("summary", {})
    rows = _rows(intonation)
    by_measure = _intonation_by_measure(summary, rows)
    for bar, stats in sorted(by_measure.items()):
        if bar not in bar_starts or int(stats.get("count", 0)) < config.min_intonation_measure_count:
            continue
        median_cents = float(stats["median_cents"])
        abs_max = float(stats.get("abs_max", stats.get("abs_max_cents", 0.0)))
        count = int(stats["count"])
        start = bar_starts[bar]
        end = start + bar_durations.get(bar, 1.0)
        if abs(median_cents) >= config.intonation_alert_cents or abs_max >= config.intonation_alert_cents + 15:
            severity = "alert" if abs(median_cents) >= config.intonation_alert_cents else "warn"
            direction = "sharp" if median_cents > 0 else "flat"
            add(
                start,
                end,
                "intonation",
                severity,
                f"Bar {bar}: median {direction} {abs(median_cents):.0f}c",
                f"Across {count} tracked notes in this bar, intonation centers {direction} by {abs(median_cents):.0f} cents (max deviation {abs_max:.0f} cents). Drone the tonic and walk through this bar slowly.",
                measure=bar,
                evidence_ref="polyphonic_intonation.by_measure",
            )
        elif abs(median_cents) >= config.intonation_warn_cents:
            direction = "sharp" if median_cents > 0 else "flat"
            add(
                start,
                end,
                "intonation",
                "warn",
                f"Bar {bar}: leans {direction} {abs(median_cents):.0f}c",
                f"{count} tracked notes; median {direction} {abs(median_cents):.0f} cents. Worth a slow listen.",
                measure=bar,
                evidence_ref="polyphonic_intonation.by_measure",
            )

    for pc, stats in (summary.get("by_pitch_class") or {}).items():
        if int(stats.get("count", 0)) < config.min_pitch_class_count:
            continue
        median_cents = float(stats.get("median_cents", 0.0))
        spread = float(stats.get("spread_p10_p90", float(stats.get("p90", 0.0)) - float(stats.get("p10", 0.0))))
        if abs(median_cents) < config.pitch_class_warn_cents and spread < config.pitch_class_spread_warn_cents:
            continue
        best_row = _worst_pitch_class_row(rows, str(pc), bar_starts)
        if best_row is None:
            continue
        target_bar = _as_int(best_row.get("measure"))
        start = bar_starts.get(target_bar, _event_time(best_row) or 0.0) if target_bar is not None else (_event_time(best_row) or 0.0)
        end = start + bar_durations.get(target_bar or -1, 1.0) * 1.5
        severity = "alert" if abs(median_cents) >= config.pitch_class_alert_cents or spread >= config.pitch_class_spread_alert_cents else "warn"
        direction = "sharp" if median_cents > 0 else "flat" if median_cents < 0 else "centered"
        spread_note = f", spread {spread:.0f}c" if spread >= 30 else ""
        add(
            start,
            end,
            "intonation",
            severity,
            f"{pc} pattern: {direction} {abs(median_cents):.0f}c{spread_note}",
            f"Across {int(stats['count'])} tracked {pc}s in this take, the median is {median_cents:+.0f} cents (p10/p90 {float(stats.get('p10', 0)):+.0f}/{float(stats.get('p90', 0)):+.0f}). This is where the worst single {pc} occurs - listen here.",
            measure=target_bar,
            evidence_ref="polyphonic_intonation.by_pitch_class",
            note_refs=[{"midi_measure": target_bar, "pitch_name": best_row.get("expected_note")}],
        )


def _add_rhythm_comments(add, rhythm: dict[str, Any], summary: dict[str, Any], bar_starts: dict[int, float], bar_durations: dict[int, float], config: MechanicalCommentsConfig) -> None:
    bar_median = _as_float(summary.get("bar_duration_median_sec"))
    if bar_median is None:
        values = [dur for dur in bar_durations.values() if dur > 0]
        bar_median = median(values) if values else None
    if bar_median:
        for row in _bar_rows(rhythm):
            bar = _as_int(row.get("bar", row.get("measure")))
            duration = _as_float(row.get("duration_sec"))
            if bar is None or duration is None or bar not in bar_starts:
                continue
            pct = (duration - bar_median) / bar_median * 100.0
            start = bar_starts[bar]
            end = start + duration
            if abs(pct) >= config.rubato_warn_pct:
                severity = "alert" if abs(pct) >= config.rubato_alert_pct else "warn"
                direction = "longer" if pct > 0 else "shorter"
                arrow = "stretching" if pct > 0 else "rushing"
                add(
                    start,
                    end,
                    "rhythm",
                    severity,
                    f"Bar {bar}: {arrow} ({pct:+.0f}% vs median)",
                    f"This bar takes {duration:.1f}s, which is {abs(pct):.0f}% {direction} than your median bar. If this is structural rubato, keep it intentional; if mid-phrase, audit whether it is conscious.",
                    measure=bar,
                    evidence_ref="polyphonic_rhythm.per_bar",
                )
            elif abs(pct) >= config.rubato_info_pct:
                direction = "longer" if pct > 0 else "shorter"
                add(
                    start,
                    end,
                    "rhythm",
                    "info",
                    f"Bar {bar}: {pct:+.0f}% vs median",
                    f"Bar duration {duration:.1f}s, {abs(pct):.0f}% {direction} than median.",
                    measure=bar,
                    evidence_ref="polyphonic_rhythm.per_bar",
                )

    by_bar: dict[int, list[dict[str, Any]]] = {}
    for outlier in summary.get("off_pulse_outliers", []):
        bar = _as_int(outlier.get("measure"))
        if bar is not None:
            by_bar.setdefault(bar, []).append(outlier)
    for bar, items in sorted(by_bar.items()):
        if bar not in bar_starts:
            continue
        items.sort(key=lambda row: -abs(float(row.get("deviation_from_local_ms", 0.0) or 0.0)))
        for outlier in items[: config.max_off_pulse_per_bar]:
            beat = _as_float(outlier.get("beat")) or 1.0
            bar_dur = bar_durations.get(bar, 1.0)
            time_hint = _as_float(outlier.get("expected_performed_time"))
            t = time_hint if time_hint is not None else bar_starts[bar] + (beat - 1.0) / 4.0 * bar_dur
            dev_ms = float(outlier.get("deviation_from_local_ms", 0.0) or 0.0)
            direction = "late" if dev_ms > 0 else "early"
            pitch = outlier.get("expected_pitch", "?")
            add(
                t,
                t + 0.8,
                "rhythm",
                "info",
                f"Bar {bar} beat {beat:.1f}: {pitch} {direction} {abs(dev_ms):.0f}ms",
                f"This {pitch} arrives {abs(dev_ms):.0f}ms {direction} relative to the bar's local pulse.",
                measure=bar,
                beat=beat,
                evidence_ref="polyphonic_rhythm.off_pulse_outliers",
            )


def _add_voicing_comments(add, voicing: dict[str, Any], bar_starts: dict[int, float], bar_durations: dict[int, float], config: MechanicalCommentsConfig) -> None:
    rows = _rows(voicing)
    weak = [row for row in rows if _as_float(row.get("melody_margin_db")) is not None and _as_float(row.get("melody_margin_db")) < config.weak_melody_margin_db]
    weak.sort(key=lambda row: float(row.get("melody_margin_db", 0.0)))
    for row in weak[: config.max_voicing_events]:
        margin = float(row["melody_margin_db"])
        measure = _as_int(row.get("measure"))
        start = _event_time(row) or bar_starts.get(measure or -1, 0.0)
        severity = "alert" if margin < config.buried_melody_margin_db else "warn"
        label = "buried" if margin < config.buried_melody_margin_db else "weak"
        melody = row.get("melody_note") or row.get("top_note") or "top voice"
        add(
            start,
            start + 1.2,
            "voicing",
            severity,
            f"Bar {measure}: melody {label} ({margin:+.1f} dB)",
            f"The written melody note {melody} is only {margin:+.1f} dB against the strongest supporting voice here. Practice the chord with the top voice projected before adding pedal.",
            measure=measure,
            beat=_as_float(row.get("beat")),
            evidence_ref="piano_voicing.rows.melody_margin_db",
        )

    attacks = [row for row in rows if (_as_float(row.get("attack_spread_ms")) or 0.0) >= config.attack_spread_warn_ms]
    attacks.sort(key=lambda row: -float(row.get("attack_spread_ms", 0.0)))
    for row in attacks[: config.max_voicing_events]:
        spread = float(row["attack_spread_ms"])
        measure = _as_int(row.get("measure"))
        start = _event_time(row) or bar_starts.get(measure or -1, 0.0)
        severity = "alert" if spread >= config.attack_spread_alert_ms else "warn"
        add(
            start,
            start + 1.0,
            "voicing",
            severity,
            f"Bar {measure}: chord attack spread {spread:.0f}ms",
            f"The chord members do not speak together; measured attack spread is {spread:.0f} ms. Block it slowly and listen for one coordinated onset.",
            measure=measure,
            beat=_as_float(row.get("beat")),
            evidence_ref="piano_voicing.rows.attack_spread_ms",
        )

    residues = [row for row in rows if row.get("pedal_blur") or ((_as_float(row.get("pedal_residue_db_rel")) or -99.0) > config.pedal_blur_warn_db)]
    residues.sort(key=lambda row: -float(row.get("pedal_residue_db_rel", -99.0) or -99.0))
    for row in residues[: config.max_voicing_events]:
        residue = _as_float(row.get("pedal_residue_db_rel"))
        measure = _as_int(row.get("measure"))
        start = _event_time(row) or bar_starts.get(measure or -1, 0.0)
        add(
            start,
            start + 1.2,
            "voicing",
            "warn",
            f"Bar {measure}: pedal residue {residue:+.1f} dB",
            f"Previous-harmony energy is still present at {residue:+.1f} dB relative to this chord. Refresh the pedal at the harmonic change so the voicing stays clear.",
            measure=measure,
            beat=_as_float(row.get("beat")),
            evidence_ref="piano_voicing.rows.pedal_residue_db_rel",
        )


def _add_overview(add, manifest: SessionManifest, rhythm_summary: dict[str, Any], intonation: dict[str, Any] | None, voicing: dict[str, Any] | None, bar_starts: dict[int, float]) -> None:
    overall_bpm = rhythm_summary.get("overall_player_quarter_bpm_median")
    onset_rate = rhythm_summary.get("onset_alignment_rate")
    first_measure = min(bar_starts) if bar_starts else None
    last_measure = max(bar_starts) if bar_starts else None
    if intonation:
        int_summary = intonation.get("summary", {})
        median_int = int_summary.get("overall_median_cents", int_summary.get("median_cents_high_conf"))
        count = int_summary.get("high_confidence_notes") or int_summary.get("present_score_notes") or int_summary.get("harmonic_confirmed_notes")
        title = f"Take overview: ~{overall_bpm} quarter BPM, intonation centered at {median_int}c"
        msg = f"You played measures {first_measure}-{last_measure} of {manifest.movement} at a median {overall_bpm} quarter BPM. {count} notes have usable intonation evidence; onset alignment rate is {(float(onset_rate or 0) * 100):.0f}%."
    elif voicing:
        global_summary = voicing.get("summary", {}).get("global", {})
        median_margin = global_summary.get("median_melody_margin_db")
        weak_count = global_summary.get("buried_or_weak_melody_events")
        title = f"Take overview: ~{overall_bpm} quarter BPM, melody margin {median_margin} dB"
        msg = f"You played measures {first_measure}-{last_measure} of {manifest.movement} at a median {overall_bpm} quarter BPM. Piano voicing analysis found {weak_count} weak or buried melody events; onset alignment rate is {(float(onset_rate or 0) * 100):.0f}%."
    else:
        title = f"Take overview: ~{overall_bpm} quarter BPM"
        msg = f"You played measures {first_measure}-{last_measure} of {manifest.movement} at a median {overall_bpm} quarter BPM; onset alignment rate is {(float(onset_rate or 0) * 100):.0f}%."
    add(0.0, 5.0, "rhythm", "info", title, msg, measure=first_measure, evidence_ref="summary")


def _read_required_json(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, relative_key: str) -> dict[str, Any]:
    key = _artifact_key(storage, store, manifest, relative_key)
    if not key:
        raise ValueError(f"missing {relative_key}; run the corresponding analysis first")
    doc = storage.read_json(key)
    if not isinstance(doc, dict):
        raise ValueError(f"{relative_key} must contain a JSON object")
    return doc


def _read_optional_json(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, relative_key: str) -> dict[str, Any] | None:
    key = _artifact_key(storage, store, manifest, relative_key)
    if not key:
        return None
    doc = storage.read_json(key)
    return doc if isinstance(doc, dict) else None


def _artifact_key(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, relative_key: str) -> str | None:
    key = manifest.artifacts.get(relative_key)
    if key and storage.exists(key):
        return key
    candidate = store.artifact_key(manifest.session, relative_key)
    return candidate if storage.exists(candidate) else None


def _bar_rows(rhythm: dict[str, Any]) -> list[dict[str, Any]]:
    rows = rhythm.get("per_bar") or rhythm.get("summary", {}).get("by_bar") or []
    if rows:
        return [row for row in rows if isinstance(row, dict)]
    old = rhythm.get("summary", {}).get("bar_durations") or []
    return [{"bar": row.get("bar"), "measure": row.get("bar"), "duration_sec": row.get("duration_sec")} for row in old if isinstance(row, dict)]


def _bar_starts(summary: dict[str, Any], bar_rows: list[dict[str, Any]]) -> dict[int, float]:
    starts: dict[int, float] = {}
    for row in bar_rows:
        bar = _as_int(row.get("bar", row.get("measure")))
        start = _as_float(row.get("perf_start_sec", row.get("start_sec")))
        if bar is not None and start is not None:
            starts[bar] = start
    if starts:
        return starts
    cum = float(summary.get("music_start_sec", 0.0) or 0.0)
    for row in summary.get("bar_durations", []):
        bar = _as_int(row.get("bar", row.get("measure")))
        dur = _as_float(row.get("duration_sec"))
        if bar is not None:
            starts[bar] = cum
        if dur is not None:
            cum += dur
    return starts


def _bar_durations(summary: dict[str, Any], bar_rows: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in bar_rows:
        bar = _as_int(row.get("bar", row.get("measure")))
        dur = _as_float(row.get("duration_sec"))
        if bar is not None and dur is not None:
            out[bar] = dur
    for row in summary.get("bar_durations", []):
        bar = _as_int(row.get("bar", row.get("measure")))
        dur = _as_float(row.get("duration_sec"))
        if bar is not None and dur is not None:
            out.setdefault(bar, dur)
    return out


def _intonation_by_measure(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    existing = summary.get("by_measure") or []
    out = {_as_int(row.get("measure")): row for row in existing if isinstance(row, dict) and _as_int(row.get("measure")) is not None}
    if out:
        return {int(k): v for k, v in out.items() if k is not None}
    values: dict[int, list[float]] = {}
    for row in rows:
        measure = _as_int(row.get("measure", row.get("midi_measure")))
        cents = _cents(row)
        if measure is not None and cents is not None and row.get("present", True):
            values.setdefault(measure, []).append(cents)
    return {
        measure: {"measure": measure, "count": len(cents), "median_cents": median(cents), "abs_max": max(abs(v) for v in cents)}
        for measure, cents in values.items()
        if cents
    }


def _worst_pitch_class_row(rows: list[dict[str, Any]], pc: str, bar_starts: dict[int, float]) -> dict[str, Any] | None:
    best_row = None
    best_abs = -1.0
    for row in rows:
        note = str(row.get("expected_note", row.get("note_name", "")))
        if _pitch_class(note) != pc:
            continue
        measure = _as_int(row.get("measure", row.get("midi_measure")))
        if measure is not None and bar_starts and measure not in bar_starts:
            continue
        cents = _cents(row)
        if cents is None:
            continue
        if abs(cents) > best_abs:
            best_abs = abs(cents)
            best_row = row
    return best_row


def _rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    raw = doc.get("rows") or doc.get("events") or doc.get("comments") or []
    return [row for row in raw if isinstance(row, dict)]


def _cents(row: dict[str, Any]) -> float | None:
    for key in ("best_temperament_cents", "cents_offset", "cents_vs_12tet", "median_cents"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _pitch_class(note: str) -> str:
    note = note.strip()
    while note and (note[-1].isdigit() or note[-1] == "-"):
        note = note[:-1]
    return note


def _event_time(row: dict[str, Any]) -> float | None:
    return _as_float(row.get("perf_time", row.get("performed_time_sec", row.get("start_sec"))))


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_piano_family(profile_id: str | None, profile: Any) -> bool:
    text = " ".join(str(part or "").lower() for part in (profile_id, getattr(profile, "id", None), getattr(profile, "instrument", None), getattr(profile, "family", None)))
    return "piano" in text or "keyboard" in text


def _counts(comments: list[dict[str, Any]]) -> dict[str, int]:
    return {severity: sum(1 for c in comments if c.get("severity") == severity) for severity in ("info", "warn", "alert")}


def _render_markdown(summary: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    lines = [
        f"# Mechanical Comments - {summary.get('repertoire') or 'Untitled'}",
        "",
        f"- Session: `{summary.get('session_id')}`",
        f"- Comment count: `{summary.get('comment_count')}`",
        f"- Count by severity: `{summary.get('count_by_severity')}`",
        "",
        "| id | time | severity | category | measure | beat | title |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for comment in comments:
        lines.append(
            f"| {comment['id']} | {comment['start_sec']}-{comment['end_sec']} | {comment['severity']} | "
            f"{comment['category']} | {comment.get('measure')} | {comment.get('beat')} | {comment['title']} |"
        )
    return "\n".join(lines) + "\n"
