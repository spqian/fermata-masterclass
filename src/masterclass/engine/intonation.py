"""Score-aware polyphonic intonation analysis.

For each HMM-aligned score event, this module measures CQT spectral energy near
each expected score pitch and reports the peak's cents deviation from equal
temperament.  The analysis is storage-scoped and writes v2 artifacts only.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from masterclass.core.artifact_catalog import ArtifactCatalog
from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class IntonationConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    bins_per_octave: int = 60
    n_octaves: int = 7
    fmin_note: str = "C2"
    search_cents: float = 50.0
    time_window_frames: int = 5
    presence_energy_ratio: float = 1.5
    harmonic_energy_ratio: float = 2.0
    high_conf_min_harmonics: int = 1
    temperament_root_midi: int = 59
    max_events: int = 5000


@dataclass
class IntonationResult:
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    markdown: str
    config: dict[str, Any]


def analyze_intonation(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    config: IntonationConfig | None = None,
) -> IntonationResult:
    """Analyze score-aware intonation from HMM-aligned note timestamps."""

    config = config or IntonationConfig()
    catalog = ArtifactCatalog(manifest)
    audio_key = catalog.audio_wav()
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav; run ingestion first")

    from masterclass.engine.aligned_notes import load_aligned_notes_source
    notes_key, raw_notes = load_aligned_notes_source(storage, manifest)
    aligned_notes = [n.to_dict() for n in raw_notes]
    if not aligned_notes:
        raise RuntimeError("no aligned notes available for intonation analysis (audio_truth pipeline must run first)")

    with tempfile.TemporaryDirectory(prefix="mc-intonation-") as tmp_raw:
        audio_path = Path(tmp_raw) / "audio.wav"
        storage.read_to_file(audio_key, audio_path)
        result = intonation_from_wav_file(
            audio_path,
            aligned_notes,
            config,
            manifest=manifest,
            audio_key=audio_key,
            notes_key=notes_key,
        )

    persist_intonation(storage=storage, store=store, manifest=manifest, result=result)
    return result


def persist_intonation(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: IntonationResult,
) -> None:
    """Write intonation artifacts and stamp the session manifest."""

    json_key = store.artifact_key(manifest.session, "analysis/polyphonic_intonation.json")
    md_key = store.artifact_key(manifest.session, "analysis/polyphonic_intonation.md")
    storage.write_json(
        json_key,
        {
            "schema_version": 1,
            "summary": result.summary,
            "events": result.events,
            "rows": result.events,
            "config": result.config,
        },
    )
    storage.write_bytes(md_key, result.markdown.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/polyphonic_intonation.json"] = json_key
    manifest.artifacts["analysis/polyphonic_intonation.md"] = md_key
    manifest.metadata["polyphonic_intonation_summary"] = result.summary
    store.save(manifest)


def intonation_from_wav_file(
    path: Path,
    aligned_notes: list[dict[str, Any]],
    config: IntonationConfig,
    *,
    manifest: SessionManifest | None = None,
    audio_key: str | None = None,
    notes_key: str | None = None,
) -> IntonationResult:
    import warnings

    import librosa
    import numpy as np

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    y, sr = librosa.load(str(path), sr=config.sample_rate, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    fmin_hz = float(librosa.note_to_hz(config.fmin_note))
    n_bins = config.bins_per_octave * config.n_octaves
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=config.hop_length,
            fmin=fmin_hz,
            n_bins=n_bins,
            bins_per_octave=config.bins_per_octave,
        )
    )
    cents_per_bin = 1200.0 / config.bins_per_octave

    events: list[dict[str, Any]] = []
    for chord_index, aligned in enumerate(aligned_notes):
        perf_time = _perf_time(aligned)
        if perf_time is None:
            continue
        time_idx = int(round(float(perf_time) * sr / config.hop_length))
        pitches = _pitches(aligned)
        names = _names(aligned, pitches, librosa)
        for note_index, pitch in enumerate(pitches):
            name = names[note_index] if note_index < len(names) else librosa.midi_to_note(int(pitch))
            measurement = _measure_polyphonic_pitch(
                cqt,
                time_idx,
                int(pitch),
                fmin_hz,
                config.bins_per_octave,
                cents_per_bin,
                search_cents=config.search_cents,
                time_window_frames=config.time_window_frames,
                presence_energy_ratio=config.presence_energy_ratio,
                harmonic_energy_ratio=config.harmonic_energy_ratio,
            )
            score_t = aligned.get("score_time_sec")
            row = {
                "aligned_note_index": int(chord_index),
                "state_idx": aligned.get("state_idx"),
                "score_time_sec": _round_or_none(score_t),
                "score_time_local": _round_or_none(score_t),
                "expected_pitch_midi": int(pitch),
                "expected_note": str(name),
                "note_name": str(name),
                "performed_time_sec": round(float(perf_time), 3),
                "perf_time": round(float(perf_time), 3),
                "timestamp_source": aligned.get("timestamp_source", "hmm_viterbi"),
                **measurement,
            }
            if row.get("present"):
                cents = float(row["cents_offset"])
                offsets = _temperament_offset(int(pitch), root=config.temperament_root_midi)
                deviations = {key: cents - offset for key, offset in offsets.items()}
                best = min(deviations, key=lambda key: abs(deviations[key]))
                row["cents_vs_12tet"] = round(deviations["12_tet"], 1)
                row["cents_vs_just"] = round(deviations["just"], 1)
                row["cents_vs_pythagorean"] = round(deviations["pythagorean"], 1)
                row["best_temperament"] = best
                row["best_temperament_cents"] = round(deviations[best], 1)
            row["confidence"] = _confidence(row, config)
            events.append(row)
            if len(events) >= config.max_events:
                break
        if len(events) >= config.max_events:
            break

    summary = _summarize(events, config, manifest, audio_key, notes_key, duration, cqt.shape, fmin_hz, cents_per_bin)
    return IntonationResult(
        events=events,
        summary=summary,
        markdown=intonation_markdown(summary, events),
        config=asdict(config),
    )


def _measure_polyphonic_pitch(
    cqt,
    time_idx: int,
    expected_midi: int,
    fmin_hz: float,
    bins_per_octave: int,
    cents_per_bin: float,
    *,
    search_cents: float,
    time_window_frames: int,
    presence_energy_ratio: float,
    harmonic_energy_ratio: float,
) -> dict[str, Any]:
    import numpy as np

    n_bins, n_frames = cqt.shape
    expected_bin = _midi_to_log_bin(expected_midi, fmin_hz, bins_per_octave)
    radius_bins = int(np.ceil(search_cents / cents_per_bin))
    bin_lo = max(0, int(np.floor(expected_bin)) - radius_bins)
    bin_hi = min(n_bins - 1, int(np.ceil(expected_bin)) + radius_bins)
    t_lo = max(0, time_idx - time_window_frames)
    t_hi = min(n_frames, time_idx + time_window_frames + 1)
    if bin_hi <= bin_lo or t_hi <= t_lo:
        return {"present": False, "reason": "out_of_bounds", "cents_offset": None}

    local_column = cqt[bin_lo : bin_hi + 1, t_lo:t_hi].mean(axis=1)
    if local_column.size < 3:
        return {"present": False, "reason": "no_local_window", "cents_offset": None}

    peak_local = int(np.argmax(local_column))
    if 0 < peak_local < len(local_column) - 1:
        y0, y1, y2 = local_column[peak_local - 1], local_column[peak_local], local_column[peak_local + 1]
        denom = y0 - 2 * y1 + y2
        sub_offset = 0.5 * (y0 - y2) / denom if abs(float(denom)) > 1e-9 else 0.0
        peak_local_refined = peak_local + float(sub_offset)
    else:
        peak_local_refined = float(peak_local)

    peak_global_bin = bin_lo + peak_local_refined
    cents_offset = (peak_global_bin - expected_bin) * cents_per_bin
    peak_energy = float(local_column[peak_local])
    full_column = cqt[:, t_lo:t_hi].mean(axis=1)
    floor = float(np.median(full_column)) + 1e-9
    energy_ratio = peak_energy / floor

    harmonic_confirmations = 0
    base_freq = 440.0 * 2 ** ((expected_midi - 69.0) / 12.0)
    harmonic_ratios: dict[str, float] = {}
    for harmonic in (2, 3, 4):
        h_bin_pos = np.log2(harmonic * base_freq / fmin_hz) * bins_per_octave
        if 0 <= h_bin_pos < n_bins - 1:
            hbin = int(round(h_bin_pos))
            h_energy = float(cqt[hbin, t_lo:t_hi].mean())
            ratio = h_energy / floor
            harmonic_ratios[str(harmonic)] = round(float(ratio), 2)
            if ratio >= harmonic_energy_ratio:
                harmonic_confirmations += 1

    present = energy_ratio >= presence_energy_ratio and abs(cents_offset) < search_cents - 1.0
    return {
        "present": bool(present),
        "cents_offset": round(float(cents_offset), 1),
        "peak_energy": round(peak_energy, 4),
        "energy_ratio_to_floor": round(float(energy_ratio), 2),
        "harmonic_confirmations": int(harmonic_confirmations),
        "harmonic_ratios_to_floor": harmonic_ratios,
    }


def _summarize(
    events: list[dict[str, Any]],
    config: IntonationConfig,
    manifest: SessionManifest | None,
    audio_key: str | None,
    notes_key: str | None,
    duration: float,
    cqt_shape: tuple[int, int],
    fmin_hz: float,
    cents_per_bin: float,
) -> dict[str, Any]:
    import numpy as np

    present = [event for event in events if event.get("present") and event.get("cents_offset") is not None]
    high_conf = [event for event in present if event.get("confidence") == "high"]
    basis = high_conf or present
    cents = [float(event["cents_offset"]) for event in basis]
    abs_values = [abs(value) for value in cents]

    by_pc_values: dict[str, list[float]] = {}
    for event in basis:
        pc = PITCH_CLASS_NAMES[int(event["expected_pitch_midi"]) % 12]
        by_pc_values.setdefault(pc, []).append(float(event["cents_offset"]))
    by_pc = {
        pc: {
            "count": len(values),
            "median_cents": round(float(np.median(values)), 1),
            "abs_max_cents": round(float(np.max(np.abs(values))), 1),
            "p10": round(float(np.percentile(values, 10)), 1),
            "p90": round(float(np.percentile(values, 90)), 1),
            "spread_p10_p90": round(float(np.percentile(values, 90) - np.percentile(values, 10)), 1),
        }
        for pc, values in sorted(by_pc_values.items(), key=lambda item: (-len(item[1]), item[0]))
    }

    return {
        "session_id": manifest.session.session_id if manifest else None,
        "repertoire": manifest.repertoire if manifest else None,
        "movement": manifest.movement if manifest else None,
        "instrument": manifest.instrument if manifest else None,
        "audio": audio_key,
        "hmm_aligned_notes": notes_key,
        "recording_duration_sec": round(duration, 3),
        "score_note_events": len(events),
        "present_score_notes": len(present),
        "high_confidence_notes": len(high_conf),
        "summary_basis": "high_confidence_notes" if high_conf else "present_score_notes",
        "presence_rate": round(len(present) / max(1, len(events)), 3),
        "high_confidence_rate": round(len(high_conf) / max(1, len(events)), 3),
        "overall_median_cents": round(float(np.median(cents)), 2) if cents else None,
        "overall_abs_max_cents": round(float(np.max(abs_values)), 1) if abs_values else None,
        "p10_cents": round(float(np.percentile(cents, 10)), 2) if cents else None,
        "p90_cents": round(float(np.percentile(cents, 90)), 2) if cents else None,
        "by_pitch_class": by_pc,
        "cqt_settings": {
            "bins_per_octave": config.bins_per_octave,
            "n_bins": config.bins_per_octave * config.n_octaves,
            "cents_per_bin": cents_per_bin,
            "fmin_hz": round(float(fmin_hz), 3),
            "shape": [int(cqt_shape[0]), int(cqt_shape[1])],
            "search_cents": config.search_cents,
            "time_window_frames": config.time_window_frames,
        },
        "method_notes": [
            "Polyphonic intonation: independent spectral peak lookup at each HMM-aligned score pitch.",
            "Cents are measured against the equal-tempered expected score pitch.",
            f"present = peak energy >= {config.presence_energy_ratio}x local CQT floor and peak inside +/-{config.search_cents - 1:.0f} cents.",
            f"high confidence = present plus at least {config.high_conf_min_harmonics} harmonic confirmation(s) among 2f/3f/4f.",
            "Use high_confidence_notes for intonation claims when available; otherwise present_score_notes are summarized.",
        ],
    }


def intonation_markdown(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    lines = [
        f"# Polyphonic Score-Aware Intonation - {summary.get('repertoire') or 'Untitled'}",
        "",
        f"- Session: `{summary.get('session_id')}`",
        f"- Movement: `{summary.get('movement')}`",
        f"- Instrument: `{summary.get('instrument')}`",
        f"- Score note events: `{summary.get('score_note_events')}`",
        f"- Present score notes: `{summary.get('present_score_notes')}` (`{float(summary.get('presence_rate') or 0) * 100:.1f}%`)",
        f"- High-confidence notes: `{summary.get('high_confidence_notes')}` (`{float(summary.get('high_confidence_rate') or 0) * 100:.1f}%`)",
        f"- Overall median cents: `{summary.get('overall_median_cents')}`",
        f"- Overall abs-max cents: `{summary.get('overall_abs_max_cents')}`",
        "",
        "## Per pitch class",
        "",
        "| PC | count | median c | abs max | p10 | p90 | spread |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pc, row in summary.get("by_pitch_class", {}).items():
        lines.append(
            f"| {pc} | {row['count']} | {row['median_cents']} | {row['abs_max_cents']} | "
            f"{row['p10']} | {row['p90']} | {row['spread_p10_p90']} |"
        )
    lines.extend(["", "## First measured notes"])
    for event in events[:40]:
        cents = event.get("cents_offset")
        cents_text = f"{float(cents):+.1f}c" if cents is not None else "n/a"
        lines.append(
            f"- `{event.get('perf_time'):.3f}s` {event.get('expected_note')} "
            f"{cents_text} ratio={event.get('energy_ratio_to_floor')} "
            f"harm={event.get('harmonic_confirmations')} conf={event.get('confidence')}"
        )
    lines.extend(["", "## Method"])
    for note in summary.get("method_notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _midi_to_log_bin(midi_pitch: float, fmin_hz: float, bins_per_octave: int) -> float:
    import numpy as np

    freq = 440.0 * 2 ** ((midi_pitch - 69.0) / 12.0)
    return float(np.log2(freq / fmin_hz) * bins_per_octave)


def _temperament_offset(p_midi: int, *, root: int) -> dict[str, float]:
    interval = (p_midi - root) % 12
    just = [0.0, 111.73, 203.91, 315.64, 386.31, 498.04, 590.22, 701.96, 813.69, 884.36, 996.09, 1088.27]
    pythagorean = [0.0, 113.69, 203.91, 294.13, 407.82, 498.04, 611.73, 701.96, 815.64, 905.87, 1019.55, 1109.78]
    tet = interval * 100.0
    return {"12_tet": 0.0, "just": just[interval] - tet, "pythagorean": pythagorean[interval] - tet}


def _perf_time(aligned: dict[str, Any]) -> float | None:
    for key in ("perf_time", "performed_time_sec", "time", "start"):
        value = aligned.get(key)
        if value is not None:
            return float(value)
    return None


def _pitches(aligned: dict[str, Any]) -> list[int]:
    raw = aligned.get("pitches_midi", aligned.get("pitches", aligned.get("pitch")))
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [int(raw)]
    return [int(value) for value in raw]


def _names(aligned: dict[str, Any], pitches: list[int], librosa) -> list[str]:
    raw = aligned.get("names", aligned.get("name", aligned.get("note")))
    if raw is None:
        return [librosa.midi_to_note(int(pitch)) for pitch in pitches]
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in raw]


def _confidence(row: dict[str, Any], config: IntonationConfig) -> str:
    if not row.get("present"):
        return "absent"
    harmonics = int(row.get("harmonic_confirmations") or 0)
    ratio = float(row.get("energy_ratio_to_floor") or 0.0)
    if harmonics >= config.high_conf_min_harmonics:
        return "high"
    if ratio >= config.presence_energy_ratio * 2.0:
        return "medium"
    return "low"


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)
