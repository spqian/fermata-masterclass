"""Audio-truth transcription + score matching.

The lesson pipeline used to alignment via the monophonic HMM, which falls
back to score-grid timestamps when the audio is polyphonic (see
hmm_align.py). That fallback is what produced the "is the C sharp?"
hallucinations in early teacher critiques: every claim was grounded in
timestamps that didn't correspond to any actual audio event.

This module replaces that with two purpose-built audio-to-MIDI models:

  - For piano lessons:
      ByteDance's piano_transcription_inference (Kong et al., ISMIR 2021).
      Trained on 200h of MAESTRO; 0.97 onset F1 on solo piano.
  - For everything else (violin, voice, wind, mixed ensembles):
      Spotify's basic-pitch (Bittner et al., ICASSP 2022).
      General-purpose polyphonic transcriber.

After transcription, we match each detected note against the reference
MIDI using greedy nearest-pitch-and-time matching with a sliding EMA
lag estimator. Matched notes inherit the score's staff_index, voice,
measure -- everything the teacher tools need for grounded comments.

Both models load heavy ML deps (TF for basic-pitch, PyTorch for PTI),
so we lazy-import them inside the functions.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage

_LOG = logging.getLogger(__name__)
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

PIANO_INSTRUMENTS = {
    # Both manifest.instrument and manifest.instrument_profile values that
    # map to PTI. The free-text field is rarely populated in production data;
    # the profile is the reliable source ("piano", "harpsichord_solo", etc).
    "piano", "harpsichord", "organ", "fortepiano", "keyboard",
    "piano_solo", "harpsichord_solo", "organ_solo", "fortepiano_solo",
}

# Harmonic offsets used by the optional harmonic-spur suppressor below.
_HARMONIC_OFFSETS = [12, 19, 24, 28, 31, 34, 36]


def _midi_to_name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def _pick_transcriber(instrument: str | None, instrument_profile: str | None = None) -> str:
    """Return 'piano_transcription' or 'basic_pitch' based on instrument hints.

    Looks at both manifest.instrument (free-text, often empty) and
    manifest.instrument_profile (populated in our pipeline). Either field
    matching a known keyboard family routes to PTI; everything else gets
    Spotify basic-pitch as the general-purpose transcriber.
    """
    for hint in (instrument_profile, instrument):
        name = (hint or "").strip().lower()
        if any(piano_kw in name for piano_kw in ("piano", "harpsichord", "organ", "fortepiano", "keyboard")):
            return "piano_transcription"
    return "basic_pitch"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def _transcribe_basic_pitch(audio_bytes: bytes) -> list[dict[str, Any]]:
    """Run Spotify's basic-pitch on a WAV byte string."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        _model_output, _midi_data, note_events = predict(
            str(tmp_path),
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=None,
            maximum_frequency=None,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )
    finally:
        try: tmp_path.unlink()
        except Exception: pass

    notes_out: list[dict[str, Any]] = []
    for i, ev in enumerate(note_events):
        start_t = float(ev[0])
        end_t = float(ev[1])
        pitch = int(ev[2])
        amp = float(ev[3]) if len(ev) > 3 else 0.0
        notes_out.append({
            "state_idx": i,
            "pitches_midi": [pitch],
            "names": [_midi_to_name(pitch)],
            "measure": None,
            "expected_perf_duration": round(end_t - start_t, 3),
            "performed_time_sec": round(start_t, 3),
            "perf_time": round(start_t, 3),
            "dwell_sec": round(end_t - start_t, 3),
            "amplitude": round(amp, 4),
            "confidence": "high" if amp >= 0.5 else "medium",
            "timestamp_source": "basic_pitch_audio",
        })
    notes_out.sort(key=lambda n: n["performed_time_sec"])
    return notes_out


def _transcribe_piano(audio_bytes: bytes) -> list[dict[str, Any]]:
    """Run ByteDance piano_transcription_inference on a WAV byte string."""
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        audio, _ = librosa.load(str(tmp_path), sr=sample_rate, mono=True)
        midi_out = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi_out.close()
        try:
            transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)
            result = transcriptor.transcribe(audio, midi_out.name)
        finally:
            try: Path(midi_out.name).unlink()
            except Exception: pass
    finally:
        try: tmp_path.unlink()
        except Exception: pass

    events = result.get("est_note_events", [])
    notes_out: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        onset = float(ev["onset_time"])
        offset = float(ev["offset_time"])
        pitch = int(ev["midi_note"])
        velocity = float(ev.get("velocity", 0))
        notes_out.append({
            "state_idx": i,
            "pitches_midi": [pitch],
            "names": [_midi_to_name(pitch)],
            "measure": None,
            "expected_perf_duration": round(offset - onset, 3),
            "performed_time_sec": round(onset, 3),
            "perf_time": round(onset, 3),
            "dwell_sec": round(offset - onset, 3),
            "amplitude": round(velocity / 127.0, 4),
            "confidence": "high",
            "timestamp_source": "piano_transcription_inference",
        })
    notes_out.sort(key=lambda n: n["performed_time_sec"])
    return notes_out


def transcribe(*, storage: ObjectStorage, manifest: SessionManifest) -> tuple[str, list[dict[str, Any]]]:
    """Return (method_name, notes) for the lesson, choosing the model by instrument."""
    audio_key = manifest.artifacts.get("artifacts/audio.wav") or manifest.artifacts.get("artifacts/audio_16k.wav")
    if not audio_key:
        raise ValueError("lesson manifest has no decoded audio.wav")
    audio_bytes = storage.read_bytes(audio_key)
    transcriber = _pick_transcriber(manifest.instrument, manifest.instrument_profile)
    _LOG.info(
        "audio-truth: instrument=%r profile=%r -> using %s",
        manifest.instrument, manifest.instrument_profile, transcriber,
    )
    if transcriber == "piano_transcription":
        return transcriber, _transcribe_piano(audio_bytes)
    return transcriber, _transcribe_basic_pitch(audio_bytes)


# ---------------------------------------------------------------------------
# Score matching (audio-truth notes -> score-anchored notes with staff/voice)
# ---------------------------------------------------------------------------
def _load_score_notes(midi_bytes: bytes) -> list[dict[str, Any]]:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    downbeats = list(pm.get_downbeats()) if pm.get_downbeats().size else []
    out: list[dict[str, Any]] = []
    for staff_idx, track in enumerate(pm.instruments):
        track_name = track.name or f"track{staff_idx}"
        for n in track.notes:
            measure = 1
            for i in range(len(downbeats) - 1):
                if downbeats[i] <= n.start < downbeats[i + 1]:
                    measure = i + 1
                    break
            else:
                if downbeats and n.start >= downbeats[-1]:
                    measure = len(downbeats)
            out.append({
                "score_time_sec": float(n.start),
                "midi_pitch": int(n.pitch),
                "staff_index": staff_idx,
                "track_name": track_name,
                "duration_sec": float(n.end - n.start),
                "measure": measure,
            })
    out.sort(key=lambda x: (x["score_time_sec"], x["midi_pitch"]))
    return out


def match_to_score(
    perf_notes: list[dict[str, Any]],
    score_notes: list[dict[str, Any]],
    *,
    time_window_sec: float = 4.0,
    lag_smoothing: float = 0.85,
) -> list[dict[str, Any]]:
    """Greedy left-to-right matcher with EMA-tracked perf-vs-score lag.

    Returns a new list (same length as ``perf_notes``) with each note
    enriched by the matched score note's ``staff_index``, ``track_name``,
    ``measure``, ``score_time_sec``, ``score_midi_pitch``, and a derived
    ``timing_offset_ms``. Unmatched perf notes keep their original data
    but get ``matched=False`` and ``staff_index=None`` so the overlay can
    render them in a neutral colour.
    """
    by_pitch: dict[int, list[int]] = {}
    for idx, sn in enumerate(score_notes):
        by_pitch.setdefault(sn["midi_pitch"], []).append(idx)
    score_claimed = [False] * len(score_notes)
    lag_estimate: float | None = None
    out: list[dict[str, Any]] = []
    for pn in perf_notes:
        enriched = dict(pn)
        pitches = pn.get("pitches_midi") or []
        if not pitches:
            enriched["matched"] = False
            enriched["staff_index"] = None
            out.append(enriched)
            continue
        pitch = int(pitches[0])
        perf_time = float(pn.get("performed_time_sec", pn.get("perf_time", 0.0)))
        best_idx = None
        best_cost = float("inf")
        for cand_idx in by_pitch.get(pitch, []):
            if score_claimed[cand_idx]:
                continue
            sn = score_notes[cand_idx]
            expected = sn["score_time_sec"] + (lag_estimate or 0.0)
            dt = abs(perf_time - expected)
            if dt > time_window_sec:
                continue
            if dt < best_cost:
                best_cost = dt
                best_idx = cand_idx
        if best_idx is None:
            enriched["matched"] = False
            enriched["staff_index"] = None
            out.append(enriched)
            continue
        score_claimed[best_idx] = True
        m = score_notes[best_idx]
        enriched["matched"] = True
        enriched["staff_index"] = m["staff_index"]
        enriched["track_name"] = m["track_name"]
        enriched["measure"] = m["measure"]
        enriched["score_time_sec"] = m["score_time_sec"]
        enriched["score_midi_pitch"] = m["midi_pitch"]
        enriched["timing_offset_ms"] = round((perf_time - m["score_time_sec"]) * 1000.0, 1)
        new_lag = perf_time - m["score_time_sec"]
        lag_estimate = new_lag if lag_estimate is None else (lag_smoothing * lag_estimate + (1 - lag_smoothing) * new_lag)
        out.append(enriched)
    return out


# ---------------------------------------------------------------------------
# End-to-end orchestration: write artifacts the technical viewer + future
# teacher tools consume.
# ---------------------------------------------------------------------------
def run_audio_truth_pipeline(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
) -> dict[str, Any]:
    """Transcribe audio + match to score, persisting both artifacts.

    Writes:
        analysis/audio_truth_notes.json            - raw transcriber output
        analysis/audio_truth_matched_notes.json    - notes enriched with score

    The "audio_truth" naming is deliberately model-agnostic so downstream
    consumers don't need to know whether PTI or basic-pitch produced the
    data. The method name is preserved in the JSON payload.
    """
    method, notes = transcribe(storage=storage, manifest=manifest)
    raw_key = store.artifact_key(manifest.session, "analysis/audio_truth_notes.json")
    storage.write_json(raw_key, {
        "schema_version": 1,
        "method": method,
        "notes": notes,
        "summary": {
            "total_notes": len(notes),
            "first_note_sec": notes[0]["performed_time_sec"] if notes else None,
            "last_note_sec": notes[-1]["performed_time_sec"] if notes else None,
        },
    })
    manifest.artifacts["analysis/audio_truth_notes.json"] = raw_key

    midi_key = manifest.artifacts.get("masterclass/reference/midi")
    matched_count = 0
    if midi_key and storage.exists(midi_key):
        score_notes = _load_score_notes(storage.read_bytes(midi_key))
        enriched = match_to_score(notes, score_notes)
        matched_count = sum(1 for n in enriched if n.get("matched"))
        matched_key = store.artifact_key(manifest.session, "analysis/audio_truth_matched_notes.json")
        storage.write_json(matched_key, {
            "schema_version": 1,
            "method": f"{method}_score_matched",
            "match_rate": round(matched_count / max(1, len(enriched)), 3),
            "notes": enriched,
            "summary": {
                "perf_notes_total": len(enriched),
                "perf_notes_matched": matched_count,
                "score_notes_total": len(score_notes),
                "staves": 1 + max((s["staff_index"] for s in score_notes), default=0),
            },
        })
        manifest.artifacts["analysis/audio_truth_matched_notes.json"] = matched_key
    else:
        _LOG.warning("no reference MIDI on lesson %s; skipping score-match step",
                     manifest.session.session_id)

    store.save(manifest)
    return {
        "method": method,
        "notes": len(notes),
        "matched": matched_count,
        "has_score_match": bool(midi_key),
    }
