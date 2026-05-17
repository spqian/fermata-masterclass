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
MIDI using banded Needleman-Wunsch sequence alignment with a tempo prior.
Matched notes inherit the score's staff_index, voice, measure --
everything the teacher tools need for grounded comments.

Both models load heavy ML deps (TF for basic-pitch, PyTorch for PTI),
so we lazy-import them inside the functions.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from masterclass.core.artifact_catalog import ArtifactCatalog
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
def _load_score_notes_from_musicxml(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Extract note events from a MusicXML (.xml/.musicxml) or MXL (.mxl) file.

    Returns the same shape as :func:`_load_score_notes` so the matcher can
    consume either source interchangeably. Wall-clock time is derived from
    metronome marks (``<sound tempo="...">``) -- when MusicXML has no tempo
    annotation, defaults to 120 qpm (the matcher's EMA lag tracker absorbs
    moderate tempo error anyway since it follows the actual performance).
    """
    import xml.etree.ElementTree as ET
    import zipfile

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    # MXL files are zip archives containing one or more .xml documents.
    if zipfile.is_zipfile(io.BytesIO(xml_bytes)):
        with zipfile.ZipFile(io.BytesIO(xml_bytes)) as archive:
            xml_names = [
                name for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml")) and "container.xml" not in name.lower()
            ]
            if not xml_names:
                raise ValueError("MXL archive contains no MusicXML document")
            xml_bytes = archive.read(xml_names[0])

    root = ET.fromstring(xml_bytes)
    out: list[dict[str, Any]] = []
    # The MusicXML <part> element groups all measures for one staff system.
    # In a piano score there's usually one <part> with two <staff> entries
    # inside each <note> for treble/bass; in a violin score there's one
    # <part> with one staff. We tag each note with its part index AND staff
    # number so multi-part scores work too.
    parts = [el for el in root.iter() if _local(el.tag) == "part"]
    for part_idx, part in enumerate(parts):
        divisions_per_qtr = 1
        tempo_qpm = 120.0
        # Time signature: defaults to 4/4. measure_length_qtr is the *nominal*
        # length of a measure in quarter notes; we use this to advance the
        # cumulative score-time cursor at the end of each measure rather than
        # trusting `cursor_qtr` (which is wherever the last <backup>/<forward>
        # left it -- often back at 0 because MusicXML emits a trailing
        # <backup> to reset for the next voice).
        beats = 4
        beat_type = 4
        measure_length_qtr = 4.0
        current_time_qtr = 0.0  # cumulative time in quarter notes within the part
        measure_number = 0
        for measure in (m for m in list(part) if _local(m.tag) == "measure"):
            measure_number = int(measure.get("number") or measure_number + 1)
            measure_start_qtr = current_time_qtr
            cursor_qtr = current_time_qtr  # position within this measure for the active "voice"
            # `prev_note_start_qtr` is the start time of the most recent
            # non-chord, non-grace note. Chord notes share THAT start time,
            # not `cursor_qtr` (which has already advanced past the lead note).
            prev_note_start_qtr = cursor_qtr
            for el in list(measure):
                tag = _local(el.tag)
                if tag == "attributes":
                    for child in list(el):
                        ctag = _local(child.tag)
                        if ctag == "divisions" and child.text:
                            divisions_per_qtr = int(child.text) or 1
                        elif ctag == "time":
                            beats_el = next((c for c in list(child) if _local(c.tag) == "beats"), None)
                            beat_type_el = next((c for c in list(child) if _local(c.tag) == "beat-type"), None)
                            try:
                                if beats_el is not None and beats_el.text:
                                    beats = int(beats_el.text)
                                if beat_type_el is not None and beat_type_el.text:
                                    beat_type = int(beat_type_el.text)
                                measure_length_qtr = float(beats) * 4.0 / float(beat_type)
                            except (TypeError, ValueError):
                                pass
                elif tag == "direction":
                    for sound in (s for s in el.iter() if _local(s.tag) == "sound"):
                        if sound.get("tempo"):
                            try:
                                tempo_qpm = float(sound.get("tempo"))
                            except (TypeError, ValueError):
                                pass
                elif tag == "sound" and el.get("tempo"):
                    try:
                        tempo_qpm = float(el.get("tempo"))
                    except (TypeError, ValueError):
                        pass
                elif tag == "backup":
                    dur_el = next((c for c in list(el) if _local(c.tag) == "duration"), None)
                    if dur_el is not None and dur_el.text:
                        cursor_qtr -= int(dur_el.text) / divisions_per_qtr
                elif tag == "forward":
                    dur_el = next((c for c in list(el) if _local(c.tag) == "duration"), None)
                    if dur_el is not None and dur_el.text:
                        cursor_qtr += int(dur_el.text) / divisions_per_qtr
                elif tag == "note":
                    is_rest = any(_local(c.tag) == "rest" for c in list(el))
                    is_chord = any(_local(c.tag) == "chord" for c in list(el))
                    is_grace = any(_local(c.tag) == "grace" for c in list(el))
                    dur_el = next((c for c in list(el) if _local(c.tag) == "duration"), None)
                    dur_qtr = (int(dur_el.text) / divisions_per_qtr) if (dur_el is not None and dur_el.text) else 0.0
                    voice_el = next((c for c in list(el) if _local(c.tag) == "voice"), None)
                    voice = voice_el.text if (voice_el is not None and voice_el.text) else "1"
                    staff_el = next((c for c in list(el) if _local(c.tag) == "staff"), None)
                    staff_index = (int(staff_el.text) - 1) if (staff_el is not None and staff_el.text and staff_el.text.isdigit()) else part_idx
                    pitch_el = next((c for c in list(el) if _local(c.tag) == "pitch"), None)
                    if pitch_el is not None and not is_rest:
                        step_el = next((c for c in list(pitch_el) if _local(c.tag) == "step"), None)
                        alter_el = next((c for c in list(pitch_el) if _local(c.tag) == "alter"), None)
                        octave_el = next((c for c in list(pitch_el) if _local(c.tag) == "octave"), None)
                        if step_el is not None and octave_el is not None and step_el.text and octave_el.text:
                            step_to_pc = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
                            pc = step_to_pc.get(step_el.text.upper(), 0)
                            alter = int(alter_el.text) if (alter_el is not None and alter_el.text) else 0
                            octave = int(octave_el.text)
                            midi = 12 * (octave + 1) + pc + alter
                            # Chord notes share the start time of the preceding
                            # lead note (not the post-advance cursor position).
                            note_start_qtr = prev_note_start_qtr if is_chord else cursor_qtr
                            score_time_sec = (note_start_qtr * 60.0) / tempo_qpm
                            out.append({
                                "score_time_sec": score_time_sec,
                                "midi_pitch": int(midi),
                                "staff_index": staff_index,
                                "track_name": f"part{part_idx}_voice{voice}_staff{staff_index}",
                                "duration_sec": (dur_qtr * 60.0) / tempo_qpm,
                                "measure": measure_number,
                            })
                    if not is_chord and not is_grace:
                        prev_note_start_qtr = cursor_qtr
                        cursor_qtr += dur_qtr
            # Always advance the cumulative cursor by exactly one nominal
            # measure, regardless of where the per-voice backup/forward dance
            # left `cursor_qtr`. Trusting `cursor_qtr` makes m.2 start back at
            # measure 1's position whenever the last voice ends with a
            # trailing <backup>.
            current_time_qtr = measure_start_qtr + measure_length_qtr
    out.sort(key=lambda x: (x["score_time_sec"], x["midi_pitch"]))
    return out


def _load_score_notes_auto(storage: ObjectStorage, manifest: SessionManifest) -> tuple[str, list[dict[str, Any]]]:
    """Load score notes from MusicXML if available, otherwise from MIDI."""
    catalog = ArtifactCatalog(manifest)
    musicxml_key = catalog.musicxml()
    if musicxml_key and storage.exists(musicxml_key):
        return "musicxml", _load_score_notes_from_musicxml(storage.read_bytes(musicxml_key))
    midi_key = manifest.artifacts.get("masterclass/reference/midi")
    if midi_key and storage.exists(midi_key):
        return "midi", _load_score_notes(storage.read_bytes(midi_key))
    return "none", []


def _scope_score_notes_to_played_range(
    score_notes: list[dict[str, Any]],
    manifest: SessionManifest,
) -> list[dict[str, Any]]:
    """Filter score notes to the played measure range.

    Multi-movement MusicXML contains ALL movements (e.g. 117 measures for
    a 4-movement Bach sonata). If the student only played the Adagio
    (m.1-22), matching their 2-minute recording against all 117 measures
    produces nonsensical results. We scope to:

    1. Explicit first/last_measure from manifest metadata (user-specified)
    2. auto_detected_first/last_measure (from score_prep)
    3. Full score (no filter) as fallback

    We add a generous margin (+4 measures) because the matcher's monotonic
    constraint + EMA lag will handle slight over-shooting, but matching
    against measures 80-117 when the student played m.1-22 is catastrophic.
    """
    if not score_notes:
        return score_notes

    first = None
    last = None
    for key_first, key_last in (
        ("first_measure", "last_measure"),
        ("auto_detected_first_measure", "auto_detected_last_measure"),
    ):
        f = manifest.metadata.get(key_first)
        l = manifest.metadata.get(key_last)
        if f is not None and l is not None:
            try:
                first, last = int(f), int(l)
                break
            except (TypeError, ValueError):
                continue

    if first is None or last is None:
        return score_notes

    margin = 4
    lo = max(1, first - margin)
    hi = last + margin
    filtered = [n for n in score_notes if lo <= int(n.get("measure", 0)) <= hi]
    _LOG.info(
        "scoped score notes from %d to %d (measures %d-%d, margin %d): %d -> %d notes",
        first, last, lo, hi, margin, len(score_notes), len(filtered),
    )
    return filtered if filtered else score_notes  # never return empty if we had notes


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
    tempo_ratio_init: float | None = None,
    time_prior_weight: float = 0.5,
    band_sec: float = 3.0,
    gap_perf: float = -0.5,
    gap_score: float = -2.0,
    match_reward: float = 5.0,
    near_reward: float = 1.0,
    time_window_sec: float = 4.0,
    lag_smoothing: float = 0.85,
    backward_tolerance_sec: float = 0.5,
    backward_penalty_sec: float = 6.0,
) -> list[dict[str, Any]]:
    """Globally align detected notes to score notes with banded Needleman-Wunsch.

    Returns a new list (same length as ``perf_notes``) with each note
    enriched by the matched score note's ``staff_index``, ``track_name``,
    ``measure``, ``score_time_sec``, ``score_midi_pitch``, and a derived
    ``timing_offset_ms``. Unmatched perf notes keep their original data
    but get ``matched=False`` and ``staff_index=None`` so the overlay can
    render them in a neutral colour.

    The old greedy matcher could alias repeated pitch material across the
    piece. This implementation instead uses a semi-global Needleman-Wunsch
    dynamic program: pitch-compatible diagonal matches compete with cheap
    performed-note gaps (extra detections/harmonics) and more expensive
    score-note gaps (missed notes). A tempo prior scores each candidate
    against ``tempo_ratio * score_time_sec + start_offset``, and a score-time
    band around that prior prevents repeated material far away in the piece
    from being considered. We run the alignment twice: first with a duration
    ratio estimate, then with a linear-regression tempo/intercept estimated
    from first-pass matches.

    ``timing_offset_ms`` is measured against the final tempo-mapped expected
    performed time, not raw score time. Legacy kwargs
    (``time_window_sec``, ``lag_smoothing``, ``backward_tolerance_sec``,
    ``backward_penalty_sec``) remain accepted for caller compatibility but are
    intentionally ignored by the NW matcher.
    """
    del time_window_sec, lag_smoothing, backward_tolerance_sec, backward_penalty_sec

    def _perf_time(note: dict[str, Any]) -> float:
        return float(note.get("performed_time_sec", note.get("perf_time", 0.0)) or 0.0)

    def _unmatched(note: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(note)
        enriched["matched"] = False
        enriched["staff_index"] = None
        return enriched

    if not perf_notes:
        return []
    if not score_notes:
        return [_unmatched(pn) for pn in perf_notes]

    sorted_perf: list[tuple[int, dict[str, Any], int, float]] = []
    for orig_idx, pn in sorted(enumerate(perf_notes), key=lambda item: _perf_time(item[1])):
        pitches = pn.get("pitches_midi") or []
        if pitches:
            sorted_perf.append((orig_idx, pn, int(pitches[0]), _perf_time(pn)))

    if not sorted_perf:
        return [_unmatched(pn) for pn in perf_notes]

    sorted_score = sorted(
        enumerate(score_notes),
        key=lambda item: (float(item[1].get("score_time_sec", 0.0) or 0.0), int(item[1].get("midi_pitch", 0) or 0)),
    )

    score_times = [float(sn.get("score_time_sec", 0.0) or 0.0) for _, sn in sorted_score]
    score_pitches = [int(sn.get("midi_pitch", 0) or 0) for _, sn in sorted_score]
    perf_times = [pt for _, _, _, pt in sorted_perf]
    perf_pitches = [pitch for _, _, pitch, _ in sorted_perf]

    last_score_time = max(score_times) if score_times else 0.0
    last_perf_time = max(perf_times) if perf_times else 0.0
    tempo_ratio = float(tempo_ratio_init) if tempo_ratio_init and tempo_ratio_init > 0 else (
        last_perf_time / last_score_time if last_score_time > 0.1 else 1.0
    )
    if tempo_ratio <= 0:
        tempo_ratio = 1.0
    start_offset = 0.0

    def _align_once(ratio: float, offset: float) -> list[tuple[int, int]]:
        n = len(perf_pitches)
        m = len(score_pitches)
        neg_inf = -1.0e15
        dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
        ptr: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
        in_band = [[False] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            if i == 0:
                lo, hi = 0, m
            else:
                center_score_time = (perf_times[i - 1] - offset) / ratio if ratio > 0 else perf_times[i - 1]
                lo = 1
                while lo <= m and score_times[lo - 1] < center_score_time - band_sec:
                    lo += 1
                hi = lo - 1
                while hi < m and score_times[hi] <= center_score_time + band_sec:
                    hi += 1
                lo = max(1, min(lo, m))
                hi = max(lo, min(hi, m))
            in_band[i][0] = True
            for j in range(lo, hi + 1):
                in_band[i][j] = True

        dp[0][0] = 0.0
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + gap_perf
            ptr[i][0] = "up"
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + gap_score
            ptr[0][j] = "left"

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if not in_band[i][j]:
                    continue
                best = neg_inf
                best_ptr: str | None = None

                if in_band[i - 1][j] and dp[i - 1][j] > neg_inf / 2:
                    score = dp[i - 1][j] + gap_perf
                    if score > best:
                        best = score
                        best_ptr = "up"

                if in_band[i][j - 1] and dp[i][j - 1] > neg_inf / 2:
                    score = dp[i][j - 1] + gap_score
                    if score > best:
                        best = score
                        best_ptr = "left"

                if in_band[i - 1][j - 1] and dp[i - 1][j - 1] > neg_inf / 2:
                    pitch_delta = abs(perf_pitches[i - 1] - score_pitches[j - 1])
                    if pitch_delta == 0:
                        pitch_score = match_reward
                    elif pitch_delta == 1:
                        pitch_score = near_reward
                    else:
                        pitch_score = -1000.0
                    expected_perf_time = ratio * score_times[j - 1] + offset
                    time_penalty = time_prior_weight * min(abs(perf_times[i - 1] - expected_perf_time), 20.0)
                    score = dp[i - 1][j - 1] + pitch_score - time_penalty
                    if score > best:
                        best = score
                        best_ptr = "diag"

                dp[i][j] = best
                ptr[i][j] = best_ptr

        min_final_j = 0
        final_j = max(range(min_final_j, m + 1), key=lambda j: dp[n][j])
        matches: list[tuple[int, int]] = []
        i, j = n, final_j
        while i > 0 or j > 0:
            step = ptr[i][j] if i >= 0 and j >= 0 else None
            if step == "diag":
                pitch_delta = abs(perf_pitches[i - 1] - score_pitches[j - 1])
                if pitch_delta <= 1:
                    matches.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif step == "up":
                i -= 1
            elif step == "left":
                j -= 1
            else:
                break
        matches.reverse()
        return matches

    matches = _align_once(tempo_ratio, start_offset)
    if len(matches) >= 2:
        xs = [score_times[j] for _, j in matches]
        ys = [perf_times[i] for i, _ in matches]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom > 1.0e-9:
            refined_ratio = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
            refined_offset = mean_y - refined_ratio * mean_x
            if refined_ratio > 0:
                tempo_ratio = refined_ratio
                start_offset = refined_offset
                matches = _align_once(tempo_ratio, start_offset)

    match_by_orig_idx: dict[int, dict[str, Any]] = {}

    def _matched_note(pn: dict[str, Any], sn: dict[str, Any], perf_time: float) -> dict[str, Any]:
        enriched = dict(pn)
        enriched["matched"] = True
        enriched["staff_index"] = sn.get("staff_index")
        enriched["track_name"] = sn.get("track_name")
        enriched["measure"] = sn.get("measure")
        enriched["score_time_sec"] = sn.get("score_time_sec")
        enriched["score_midi_pitch"] = sn.get("midi_pitch")
        expected_perf_time = tempo_ratio * float(sn.get("score_time_sec", 0.0) or 0.0) + start_offset
        enriched["timing_offset_ms"] = round((perf_time - expected_perf_time) * 1000.0, 1)
        return enriched

    for perf_i, score_j in matches:
        orig_idx, pn, _, perf_time = sorted_perf[perf_i]
        _, sn = sorted_score[score_j]
        match_by_orig_idx[orig_idx] = _matched_note(pn, sn, perf_time)

    # Basic-pitch often emits several same-pitch fragments around one notated
    # note. NW correctly finds the score path, then this bounded fill marks
    # near-tempo duplicate fragments as score-grounded without letting distant
    # repeated material alias across the piece.
    duplicate_fill_tolerance_sec = 2.0
    for orig_idx, pn, pitch, perf_time in sorted_perf:
        if orig_idx in match_by_orig_idx:
            continue
        best_sn: dict[str, Any] | None = None
        best_err = float("inf")
        for _, sn in sorted_score:
            if abs(pitch - int(sn.get("midi_pitch", 0) or 0)) > 1:
                continue
            expected_perf_time = tempo_ratio * float(sn.get("score_time_sec", 0.0) or 0.0) + start_offset
            err = abs(perf_time - expected_perf_time)
            if err < best_err:
                best_err = err
                best_sn = sn
        if best_sn is not None and best_err <= duplicate_fill_tolerance_sec:
            match_by_orig_idx[orig_idx] = _matched_note(pn, best_sn, perf_time)

    return [match_by_orig_idx.get(i, _unmatched(pn)) for i, pn in enumerate(perf_notes)]


# ---------------------------------------------------------------------------
# End-to-end orchestration: write artifacts the technical viewer + future
# teacher tools consume.
# ---------------------------------------------------------------------------
def _enrich_for_legacy_consumers(note: dict[str, Any]) -> dict[str, Any]:
    """Add HMM-era field aliases to an audio-truth note so legacy consumers
    (voicing/rhythm/intonation/score_map/inspect_*) see the names they grew
    up reading. Each alias is derived from data we already have; nothing
    fabricated.
    """
    enriched = dict(note)
    # Score-time aliases: HMM-era code looks at score_time_in_movement and
    # score_time_local on every note row; we have the same value under
    # score_time_sec from the matcher.
    score_t = enriched.get("score_time_sec")
    if score_t is not None:
        enriched.setdefault("score_time_in_movement", round(float(score_t), 3))
        enriched.setdefault("score_time_local", round(float(score_t), 3))
    # obs_log_prob: HMM used Viterbi observation log-prob as a quality scalar.
    # We don't have one; expose amplitude (basic-pitch) or velocity-normalized
    # (PTI) so consumers that gate on "is this note loud enough to trust"
    # have something to read instead of None.
    amp = enriched.get("amplitude")
    if amp is not None and "obs_log_prob" not in enriched:
        enriched["obs_log_prob"] = float(amp)
    # expected_pitch hint: rhythm.py et al sometimes look at this to render
    # mismatch info. For matched notes, the score pitch IS the expected pitch.
    score_pitch = enriched.get("score_midi_pitch")
    if score_pitch is not None and "expected_pitch" not in enriched:
        enriched["expected_pitch"] = int(score_pitch)
        # Name form too (e.g. "C5") -- mirrors the existing "names" list shape.
        midi_int = int(score_pitch)
        name = f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi_int % 12]}{midi_int // 12 - 1}"
        enriched["expected_pitch_name"] = name
    return enriched


def _build_legacy_hmm_artifacts(
    *,
    detected_notes: list[dict[str, Any]],
    matched_notes: list[dict[str, Any]] | None,
    method: str,
    score_source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce HMM-shaped artifacts from audio-truth data so legacy consumers
    (voicing.py, rhythm.py, intonation.py, score_map.py, agent_tools/*) keep
    working unchanged while we delete the old HMM pipeline.

    Each note is run through _enrich_for_legacy_consumers so the historical
    field names (score_time_in_movement, score_time_local, obs_log_prob,
    expected_pitch) are present alongside the modern ones.

    Returns ``(aligned_notes_doc, alignment_doc)`` matching the historical
    shape of ``analysis/hmm_aligned_notes.json`` and ``analysis/hmm_alignment.json``.
    """
    raw_for_shim = matched_notes or detected_notes
    notes_for_shim = [_enrich_for_legacy_consumers(n) for n in raw_for_shim]
    # bar_starts: for each measure mentioned in the matched notes, the first
    # performed_time_sec for that measure. Sorted by measure so the legacy
    # consumers can iterate in order.
    by_measure: dict[int, float] = {}
    for n in notes_for_shim:
        m = n.get("measure")
        t = n.get("performed_time_sec", n.get("perf_time"))
        if m is None or t is None:
            continue
        if m not in by_measure or t < by_measure[m]:
            by_measure[m] = float(t)
    bar_starts = [
        {
            "measure": m,
            "performed_time_sec": round(t, 3),
            "start": round(t, 3),
            "first_visited_pitches": [],
            "is_score_bar_first_state": True,
            "method": "audio_truth_first_note",
        }
        for m, t in sorted(by_measure.items())
    ]
    # measure_timestamps: rhythm.py reads this as a list of {measure, start}
    # dicts (NOT a dict-of-floats -- that breaks _normalized_measures which
    # iterates assuming each row exposes .get). Mirror bar_starts in that
    # exact shape so rhythm picks it up without code change.
    measure_timestamps = [
        {"measure": b["measure"], "start": b["performed_time_sec"]}
        for b in bar_starts
    ]
    music_start = bar_starts[0]["performed_time_sec"] if bar_starts else (
        notes_for_shim[0].get("performed_time_sec", 0.0) if notes_for_shim else 0.0
    )
    played_measures = sorted(by_measure.keys())
    aligned_notes_doc = {
        "schema_version": 1,
        "notes": notes_for_shim,
        "method": f"audio_truth:{method}:{score_source}",
        "source_compat": "hmm_aligned_notes_v1",
    }
    alignment_doc = {
        "schema_version": 1,
        "summary": {
            "method": f"audio_truth:{method}:{score_source}",
            "music_start_sec": music_start,
            "n_states": len(notes_for_shim),
            "note_count": len(notes_for_shim),
            "tempo_factor_perf_over_midi": 1.0,
            "played_measures": played_measures,
            "effective_first_measure": played_measures[0] if played_measures else None,
            "effective_last_measure": played_measures[-1] if played_measures else None,
            "method_notes": [
                "Legacy HMM-shaped artifact synthesized from audio_truth output.",
                "Underlying transcriber: " + method,
                "Score-matching source: " + score_source,
            ],
        },
        "bar_starts": bar_starts,
        "measure_timestamps": measure_timestamps,
        "notes": notes_for_shim,
        "note_alignments": notes_for_shim,
        "measure_count": len(bar_starts),
        "source_compat": "hmm_alignment_v1",
    }
    return aligned_notes_doc, alignment_doc


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
        analysis/hmm_aligned_notes.json            - legacy-shim for old consumers
        analysis/hmm_alignment.json                - legacy-shim with bar_starts

    The "audio_truth" naming is deliberately model-agnostic so downstream
    consumers don't need to know whether PTI or basic-pitch produced the
    data. The HMM-shaped shim is a transitional output -- once every
    consumer (voicing, rhythm, intonation, score_map, inspect_bar,
    inspect_note) is migrated to read audio_truth_matched_notes directly,
    we delete the shim and the hmm_align module along with it.
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

    matched_count = 0
    score_source, score_notes = _load_score_notes_auto(storage, manifest)
    # Scope score notes to the played measure range. Without this,
    # multi-movement MusicXML (e.g. all 4 movements of a Bach sonata)
    # causes the matcher to spread a 2-minute Adagio recording across
    # 117 measures of the full sonata, producing nonsensical timestamps.
    score_notes = _scope_score_notes_to_played_range(score_notes, manifest)
    enriched: list[dict[str, Any]] | None = None
    if score_notes:
        # Estimate tempo ratio so the matcher can widen its time window.
        # The MusicXML score_time_sec assumes the OMR's default tempo (~120bpm)
        # but the performance may be 2-4x slower. Without adapting, the 4s
        # time_window can't bridge a 3x slowdown and match rate plummets.
        perf_duration = notes[-1]["performed_time_sec"] - notes[0]["performed_time_sec"] if len(notes) >= 2 else 0.0
        score_duration = max(n["score_time_sec"] for n in score_notes) - min(n["score_time_sec"] for n in score_notes) if score_notes else 0.0
        tempo_ratio = (perf_duration / score_duration) if score_duration > 0.1 else 1.0
        adaptive_window = max(4.0, min(30.0, 4.0 * tempo_ratio))
        _LOG.info(
            "matcher window: perf=%.1fs score=%.1fs ratio=%.2f -> window=%.1fs",
            perf_duration, score_duration, tempo_ratio, adaptive_window,
        )
        enriched = match_to_score(notes, score_notes, time_window_sec=adaptive_window)
        matched_count = sum(1 for n in enriched if n.get("matched"))
        matched_key = store.artifact_key(manifest.session, "analysis/audio_truth_matched_notes.json")
        storage.write_json(matched_key, {
            "schema_version": 1,
            "method": f"{method}_score_matched",
            "score_source": score_source,
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
        _LOG.warning("no reference score (MusicXML or MIDI) on lesson %s; skipping score-match step",
                     manifest.session.session_id)

    # Compat shim: write HMM-shaped artifacts so the old consumers keep
    # working until they are individually migrated to audio_truth_matched.
    aligned_doc, alignment_doc = _build_legacy_hmm_artifacts(
        detected_notes=notes,
        matched_notes=enriched,
        method=method,
        score_source=score_source,
    )
    aligned_key = store.artifact_key(manifest.session, "analysis/hmm_aligned_notes.json")
    storage.write_json(aligned_key, aligned_doc)
    manifest.artifacts["analysis/hmm_aligned_notes.json"] = aligned_key
    # Vocabulary-clean alias: write the same document under the new name so
    # consumers can migrate off the ``hmm_`` prefix at their own pace. See
    # ``aligned_notes._CANDIDATE_KEYS`` for the deprecation timeline.
    aligned_alias_key = store.artifact_key(manifest.session, "analysis/aligned_notes.json")
    storage.write_json(aligned_alias_key, aligned_doc)
    manifest.artifacts["analysis/aligned_notes.json"] = aligned_alias_key
    align_key = store.artifact_key(manifest.session, "analysis/hmm_alignment.json")
    storage.write_json(align_key, alignment_doc)
    manifest.artifacts["analysis/hmm_alignment.json"] = align_key

    store.save(manifest)
    return {
        "method": method,
        "score_source": score_source,
        "notes": len(notes),
        "matched": matched_count,
        "has_score_match": bool(score_notes),
        "legacy_shim_written": True,
    }
