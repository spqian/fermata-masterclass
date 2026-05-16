"""Recording-derived piano voicing analysis.

Measures CQT energy at HMM-aligned score events to estimate melody projection,
chord balance, attack spread, and previous-harmony pedal residue.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


@dataclass(frozen=True)
class VoicingConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    bins_per_octave: int = 60
    n_octaves: int = 8
    fmin_note: str = "A0"
    radius_bins: int = 2
    onset_window_sec: float = 0.18
    sustain_window_sec: float = 0.75
    pre_attack_sec: float = 0.08
    attack_curve_sec: float = 0.30
    attack_threshold_ratio: float = 0.55
    present_db_threshold: float = -18.0
    present_floor_ratio: float = 1.4
    pedal_blur_db_threshold: float = -8.0


@dataclass
class VoicingResult:
    chords: list[dict[str, Any]]
    summary: dict[str, Any]
    markdown: str
    config: dict[str, Any]


def analyze_voicing(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    config: VoicingConfig | None = None,
) -> VoicingResult:
    """Analyze piano voicing from aligned notes and lesson audio."""

    del store  # Artifact keys are read from the manifest in the analysis step.
    config = config or VoicingConfig()
    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav; run ingestion first")

    import librosa
    import numpy as np

    from masterclass.engine.aligned_notes import load_aligned_notes
    notes = _normalized_aligned_notes(load_aligned_notes(storage, manifest))
    if not notes:
        raise RuntimeError("no aligned notes available for voicing analysis (audio_truth pipeline must run first)")

    measure_starts = _measure_starts_from_manifest(storage, manifest)
    audio_bytes = io.BytesIO(storage.read_bytes(audio_key))
    y, _ = librosa.load(audio_bytes, sr=config.sample_rate, mono=True)

    fmin_hz = float(librosa.note_to_hz(config.fmin_note))
    n_bins = config.bins_per_octave * config.n_octaves
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            fmin=fmin_hz,
            n_bins=n_bins,
            bins_per_octave=config.bins_per_octave,
        )
    )
    times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=config.sample_rate, hop_length=config.hop_length)

    chords = _event_measurements(notes, cqt, times, fmin_hz, config.bins_per_octave, config, measure_starts)
    summary = _summarize(chords)
    summary.update(
        {
            "session_id": manifest.session.session_id,
            "repertoire": manifest.repertoire,
            "movement": manifest.movement,
            "instrument_profile": manifest.instrument_profile,
            "audio": audio_key,
            "aligned_notes": notes_key,
            "method_notes": _method_notes(),
            "score_follower_note": (
                "This module consumes the global HMM aligned-note artifact. "
                "A separate piano chroma/bar-locked score follower is available in "
                "masterclass.engine.piano_score_follower; piano-specific HMM transitions "
                "could still be added to hmm_align.py later."
            ),
        }
    )
    return VoicingResult(chords=chords, summary=summary, markdown=_render_markdown(summary, chords), config=asdict(config))


def persist_voicing(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: VoicingResult,
) -> None:
    """Persist piano voicing JSON/Markdown and stamp the session manifest."""

    json_key = store.artifact_key(manifest.session, "analysis/piano_voicing.json")
    md_key = store.artifact_key(manifest.session, "analysis/piano_voicing.md")
    storage.write_json(
        json_key,
        {
            "schema_version": 1,
            "summary": result.summary,
            "rows": result.chords,
            "config": result.config,
        },
    )
    storage.write_bytes(md_key, result.markdown.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/piano_voicing.json"] = json_key
    manifest.artifacts["analysis/piano_voicing.md"] = md_key
    manifest.metadata["piano_voicing_summary"] = result.summary.get("global", {})
    manifest.metadata["piano_voicing_chord_count"] = len(result.chords)
    store.save(manifest)


def _normalized_aligned_notes(raw_notes: Any) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for raw in raw_notes or []:
        if not isinstance(raw, dict):
            continue
        pitches = raw.get("midi_pitches") or raw.get("pitches_midi") or raw.get("pitches") or []
        perf_time = raw.get("perf_time", raw.get("performed_time_sec"))
        if perf_time is None or not pitches:
            continue
        names = raw.get("names") or raw.get("note_names") or [str(p) for p in pitches]
        notes.append(
            {
                **raw,
                "perf_time": float(perf_time),
                "midi_pitches": [int(p) for p in pitches],
                "names": [str(n) for n in names],
            }
        )
    notes.sort(key=lambda n: float(n["perf_time"]))
    return notes


def _measure_starts_from_manifest(storage: ObjectStorage, manifest: SessionManifest) -> list[dict[str, float]]:
    key = manifest.artifacts.get("analysis/hmm_alignment.json")
    if not key or not storage.exists(key):
        return []
    try:
        alignment = storage.read_json(key)
    except Exception:
        return []
    starts = []
    for row in alignment.get("measure_timestamps") or alignment.get("bar_starts") or []:
        measure = row.get("measure")
        start = row.get("start", row.get("performed_time_sec"))
        if measure is not None and start is not None:
            starts.append({"measure": int(measure), "start": float(start)})
    starts.sort(key=lambda r: r["start"])
    return starts


def _infer_measure(note: dict[str, Any], perf_time: float, measure_starts: list[dict[str, float]]) -> int | None:
    explicit = note.get("midi_measure", note.get("measure"))
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    measure = None
    for row in measure_starts:
        if perf_time + 1e-6 >= float(row["start"]):
            measure = int(row["measure"])
        else:
            break
    return measure


def _midi_to_bin(midi_pitch: float, fmin_hz: float, bins_per_octave: int) -> float:
    import numpy as np

    freq = 440.0 * 2 ** ((midi_pitch - 69.0) / 12.0)
    return float(np.log2(freq / fmin_hz) * bins_per_octave)


def _band_energy(cqt, midi_pitch: int, frame_lo: int, frame_hi: int, fmin_hz: float, bins_per_octave: int, radius_bins: int) -> float:
    n_bins, n_frames = cqt.shape
    frame_lo = max(0, min(n_frames - 1, frame_lo))
    frame_hi = max(frame_lo + 1, min(n_frames, frame_hi))
    ctr = int(round(_midi_to_bin(midi_pitch, fmin_hz, bins_per_octave)))
    lo = max(0, ctr - radius_bins)
    hi = min(n_bins - 1, ctr + radius_bins)
    return float(cqt[lo : hi + 1, frame_lo:frame_hi].max(axis=0).mean())


def _band_curve(cqt, midi_pitch: int, frame_lo: int, frame_hi: int, fmin_hz: float, bins_per_octave: int, radius_bins: int):
    n_bins, n_frames = cqt.shape
    frame_lo = max(0, min(n_frames - 1, frame_lo))
    frame_hi = max(frame_lo + 1, min(n_frames, frame_hi))
    ctr = int(round(_midi_to_bin(midi_pitch, fmin_hz, bins_per_octave)))
    lo = max(0, ctr - radius_bins)
    hi = min(n_bins - 1, ctr + radius_bins)
    return cqt[lo : hi + 1, frame_lo:frame_hi].max(axis=0)


def _attack_offset_ms(curve, frame_dt: float, pre_frames: int, threshold_ratio: float) -> float | None:
    import numpy as np

    if len(curve) < 4 or float(curve.max()) <= 0:
        return None
    peak = float(curve.max())
    floor = float(np.percentile(curve, 10))
    threshold = floor + threshold_ratio * (peak - floor)
    candidates = np.where(curve >= threshold)[0]
    if len(candidates) == 0:
        return None
    return round((int(candidates[0]) - pre_frames) * frame_dt * 1000.0, 1)


def _event_measurements(
    notes: list[dict[str, Any]],
    cqt,
    times,
    fmin_hz: float,
    bins_per_octave: int,
    config: VoicingConfig,
    measure_starts: list[dict[str, float]],
) -> list[dict[str, Any]]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    frame_dt = float(times[1] - times[0]) if len(times) > 1 else config.hop_length / config.sample_rate
    prev_pitches: list[int] = []

    for idx, note in enumerate(notes):
        pitches = sorted({int(p) for p in note.get("midi_pitches", [])})
        if len(pitches) < 2:
            prev_pitches = pitches
            continue

        names_by_pitch = _names_by_pitch(note, pitches)
        t = float(note["perf_time"])
        next_t = _next_perf_time(notes, idx, t)
        onset_hi_t = min(t + config.onset_window_sec, next_t)
        sustain_hi_t = min(t + config.sustain_window_sec, max(onset_hi_t + 0.05, next_t))
        pre_frames = max(1, int(round(config.pre_attack_sec / frame_dt)))
        f0 = int(np.searchsorted(times, t))
        f_on_hi = max(f0 + 1, int(np.searchsorted(times, onset_hi_t)))
        f_sus_hi = max(f_on_hi + 1, int(np.searchsorted(times, sustain_hi_t)))
        f_curve_lo = max(0, f0 - pre_frames)
        f_curve_hi = min(cqt.shape[1], f0 + int(round(config.attack_curve_sec / frame_dt)))

        floor_slice = cqt[:, f0:f_on_hi]
        floor = float(np.median(floor_slice)) + 1e-9 if floor_slice.size else 1e-9
        members = []
        onset_energies = []
        attack_offsets = []
        for pitch in pitches:
            onset = _band_energy(cqt, pitch, f0, f_on_hi, fmin_hz, bins_per_octave, config.radius_bins)
            sustain = _band_energy(cqt, pitch, f_on_hi, f_sus_hi, fmin_hz, bins_per_octave, config.radius_bins)
            curve = _band_curve(cqt, pitch, f_curve_lo, f_curve_hi, fmin_hz, bins_per_octave, config.radius_bins)
            attack = _attack_offset_ms(curve, frame_dt, pre_frames, config.attack_threshold_ratio)
            onset_energies.append(onset)
            if attack is not None:
                attack_offsets.append(attack)
            members.append(
                {
                    "name": names_by_pitch.get(pitch, str(pitch)),
                    "midi": pitch,
                    "role": "melody" if pitch == max(pitches) else "bass" if pitch == min(pitches) else "inner",
                    "onset_energy": onset,
                    "sustain_energy": sustain,
                    "energy_ratio_to_floor": round(onset / floor, 2),
                    "attack_offset_ms": attack,
                }
            )

        max_onset = max(max(onset_energies), 1e-9)
        for member in members:
            member["onset_db_rel"] = round(20.0 * np.log10(max(member.pop("onset_energy"), 1e-9) / max_onset), 1)
            member["sustain_db_rel"] = round(20.0 * np.log10(max(member.pop("sustain_energy"), 1e-9) / max_onset), 1)
            member["present"] = bool(
                member["onset_db_rel"] >= config.present_db_threshold
                and member["energy_ratio_to_floor"] >= config.present_floor_ratio
            )

        melody = next(member for member in members if member["role"] == "melody")
        accomp = [member for member in members if member["role"] != "melody"]
        accomp_peak = max((float(member["onset_db_rel"]) for member in accomp), default=-99.0)
        bass = next((member for member in members if member["role"] == "bass"), None)
        melody_margin = float(melody["onset_db_rel"]) - accomp_peak
        residue_db = _pedal_residue_db(prev_pitches, pitches, cqt, f0, f_on_hi, fmin_hz, bins_per_octave, config, max_onset)
        attack_spread = round(max(attack_offsets) - min(attack_offsets), 1) if len(attack_offsets) >= 2 else None

        rows.append(
            {
                "note_id": note.get("note_id", note.get("state_idx")),
                "state_idx": note.get("state_idx"),
                "measure": _infer_measure(note, t, measure_starts),
                "beat": note.get("beat_in_bar"),
                "score_time_in_movement": note.get("score_time_in_movement"),
                "perf_time": round(t, 3),
                "names": [names_by_pitch.get(p, str(p)) for p in pitches],
                "midi_pitches": pitches,
                "members": members,
                "melody_note": melody["name"],
                "top_note": melody["name"],
                "melody_margin_db": round(float(melody_margin), 1),
                "melody_projection": _projection_label(melody_margin),
                "bass_db_rel": bass["onset_db_rel"] if bass else None,
                "attack_spread_ms": attack_spread,
                "pedal_residue_db_rel": residue_db,
                "pedal_blur": bool(residue_db is not None and residue_db > config.pedal_blur_db_threshold),
                "interpretation": "onset_db_rel is normalized within this score event; 0 dB is the strongest measured chord member in the recording.",
            }
        )
        prev_pitches = pitches
    return rows


def _next_perf_time(notes: list[dict[str, Any]], idx: int, current_t: float) -> float:
    for nxt in notes[idx + 1 :]:
        if nxt.get("perf_time") is not None:
            return float(nxt["perf_time"])
    return current_t + 0.6


def _names_by_pitch(note: dict[str, Any], pitches: list[int]) -> dict[int, str]:
    names = [str(n) for n in note.get("names", [])]
    raw_pitches = [int(p) for p in note.get("midi_pitches", [])]
    mapping = {p: name for p, name in zip(raw_pitches, names)}
    return {p: mapping.get(p, str(p)) for p in pitches}


def _pedal_residue_db(
    prev_pitches: list[int],
    pitches: list[int],
    cqt,
    f0: int,
    f_on_hi: int,
    fmin_hz: float,
    bins_per_octave: int,
    config: VoicingConfig,
    max_onset: float,
) -> float | None:
    import numpy as np

    prev_only = sorted(set(prev_pitches) - set(pitches))
    if not prev_only:
        return None
    residue = max(
        _band_energy(cqt, pitch, f0, f_on_hi, fmin_hz, bins_per_octave, config.radius_bins)
        for pitch in prev_only
    )
    return round(20.0 * np.log10(max(residue, 1e-9) / max(max_onset, 1e-9)), 1)


def _projection_label(melody_margin: float) -> str:
    if melody_margin < 0:
        return "buried"
    if melody_margin < 3:
        return "weak"
    if melody_margin < 7:
        return "clear"
    return "dominant"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    by_measure: dict[int | str, list[dict[str, Any]]] = {}
    for row in rows:
        measure = row.get("measure")
        by_measure.setdefault(int(measure) if measure is not None else "unknown", []).append(row)

    measures = []
    for measure, measure_rows in sorted(by_measure.items(), key=lambda item: (999999 if item[0] == "unknown" else int(item[0]))):
        margins = [float(r["melody_margin_db"]) for r in measure_rows]
        attacks = [float(r["attack_spread_ms"]) for r in measure_rows if r.get("attack_spread_ms") is not None]
        residues = [float(r["pedal_residue_db_rel"]) for r in measure_rows if r.get("pedal_residue_db_rel") is not None]
        measures.append(
            {
                "measure": measure,
                "events": len(measure_rows),
                "median_melody_margin_db": round(float(np.median(margins)), 1),
                "mean_melody_margin_db": round(float(np.mean(margins)), 1),
                "buried_or_weak_melody_events": sum(1 for r in measure_rows if r["melody_projection"] in ("buried", "weak")),
                "median_attack_spread_ms": round(float(np.median(attacks)), 1) if attacks else None,
                "max_attack_spread_ms": round(float(np.max(attacks)), 1) if attacks else None,
                "median_pedal_residue_db_rel": round(float(np.median(residues)), 1) if residues else None,
                "pedal_blur_events": sum(1 for r in measure_rows if r.get("pedal_blur")),
            }
        )

    margins = [float(r["melody_margin_db"]) for r in rows]
    worst_melody = sorted(rows, key=lambda r: float(r["melody_margin_db"]))[:12]
    worst_attack = sorted([r for r in rows if r.get("attack_spread_ms") is not None], key=lambda r: -float(r["attack_spread_ms"]))[:12]
    strongest_residue = sorted(
        [r for r in rows if r.get("pedal_residue_db_rel") is not None],
        key=lambda r: -float(r["pedal_residue_db_rel"]),
    )[:12]
    return {
        "events_analyzed": len(rows),
        "chord_count": len(rows),
        "by_measure": measures,
        "worst_melody_projection": worst_melody,
        "widest_attack_spread": worst_attack,
        "strongest_pedal_residue": strongest_residue,
        "global": {
            "mean_melody_margin_db": round(float(np.mean(margins)), 1) if margins else None,
            "median_melody_margin_db": round(float(np.median(margins)), 1) if margins else None,
            "buried_or_weak_melody_events": sum(1 for r in rows if r["melody_projection"] in ("buried", "weak")),
            "pedal_blur_events": sum(1 for r in rows if r.get("pedal_blur")),
        },
    }


def _method_notes() -> list[str]:
    return [
        "Recording-derived piano voicing: measures CQT energy at each HMM-aligned score event's written pitches.",
        "melody_margin_db compares the written top voice against the strongest non-melody chord member.",
        "attack_spread_ms estimates how far apart chord-member energy attacks arrive.",
        "pedal_residue_db_rel measures previous-harmony pitch energy still present at the next event.",
        "This is performance evidence, not reference-MIDI visualization.",
    ]


def _render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["# Piano Voicing Analysis", ""]
    lines.append(f"- Session: `{summary.get('session_id')}`")
    lines.append(f"- Events analyzed: `{summary['events_analyzed']}`")
    lines.append(f"- Mean melody margin: `{summary['global']['mean_melody_margin_db']}` dB")
    lines.append(f"- Median melody margin: `{summary['global']['median_melody_margin_db']}` dB")
    lines.append(f"- Buried/weak melody events: `{summary['global']['buried_or_weak_melody_events']}`")
    lines.append(f"- Pedal-blur events: `{summary['global']['pedal_blur_events']}`")
    lines.extend(["", "## Per measure", ""])
    lines.append("| Measure | events | median melody margin dB | weak/buried melody | median attack spread ms | pedal blur events |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for measure in summary["by_measure"]:
        lines.append(
            f"| {measure['measure']} | {measure['events']} | {measure['median_melody_margin_db']} | "
            f"{measure['buried_or_weak_melody_events']} | {measure['median_attack_spread_ms']} | {measure['pedal_blur_events']} |"
        )

    lines.extend(["", "## Worst melody projection", ""])
    lines.append("| Measure | beat | time | melody | margin dB | projection |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary["worst_melody_projection"][:10]:
        lines.append(
            f"| {row.get('measure')} | {row.get('beat')} | {row['perf_time']} | "
            f"{row['melody_note']} | {row['melody_margin_db']} | {row['melody_projection']} |"
        )

    lines.extend(["", "## Widest chord attack spreads", ""])
    lines.append("| Measure | beat | time | spread ms | notes |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in summary["widest_attack_spread"][:10]:
        lines.append(f"| {row.get('measure')} | {row.get('beat')} | {row['perf_time']} | {row['attack_spread_ms']} | {', '.join(row['names'])} |")

    lines.extend(["", "## Method"])
    for note in summary.get("method_notes", _method_notes()):
        lines.append(f"- {note}")
    lines.append(f"- {summary.get('score_follower_note')}")
    return "\n".join(lines) + "\n"
