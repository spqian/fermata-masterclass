# Lesson Processing Pipeline

This is what runs after a user uploads a video. All stages live in `apps/api/main.py:_run_lesson_jobs` (~250 lines), which calls into `engine/*` modules.

Stages run **sequentially** — each writes its outputs to storage and updates `manifest.metadata[f"{stage}_state"]` so the wizard's processing-status view can poll progress. If a stage fails, the orchestrator catches the exception and continues to the next stage; the lesson can still produce useful output even if (e.g.) intonation fails.

## Stage map

| # | Stage | Engine module | Output artifacts | Failure mode |
|---|---|---|---|---|
| 1 | `extract_media` | `engine/ingest.py:extract_media_artifacts` | `artifacts/audio.wav`, `artifacts/frames/*.jpg` | fatal — pipeline stops |
| 2 | `analyze` | `engine/analysis.py:analyze_session` | `analysis/analysis.json`, `analysis/pitch_events.json` | fatal |
| 3 | `evidence_packet` | `engine/analysis.py:build_evidence_packet` | `analysis/evidence_packet.md` | fatal |
| 4 | `onsets` | `engine/onsets.py` (called inside hmm_align) | `analysis/rich_onsets.json` | best-effort |
| 5 | `hmm_align` | `engine/hmm_align.py:align_lesson_with_midi_hmm` | `analysis/hmm_alignment.json`, `analysis/hmm_aligned_notes.json` | best-effort |
| 6 | `score_map` | `engine/score_map.py:build_score_map` + `persist_score_map` | `score/score_map.json` | best-effort |
| 7 | `intonation` | `engine/intonation.py:analyze_intonation` | `analysis/polyphonic_intonation.json` | best-effort, skipped for piano |
| 8 | `rhythm` | `engine/rhythm.py:analyze_rhythm` | `analysis/polyphonic_rhythm.json` | best-effort |
| 9 | `voicing` | `engine/voicing.py:analyze_voicing` | `analysis/voicing.json` | best-effort, keyboard-only |
| 10 | `mechanical_comments` | `engine/mechanical_comments.py` | `analysis/mechanical_comments.json` | best-effort |
| 11 | `teach` | `engine/teach_lesson.py:teach_lesson` | `lesson/comments_enriched.json`, `analysis/teach_tool_calls.json` | best-effort — lesson marked READY even if teacher fails |
| — | `READY` | (manifest state set) | session.json updated | terminal |

## Stage 1: extract_media

ffmpeg extracts:

- **`audio.wav`**: 22050 Hz, mono, 16-bit PCM. The canonical analysis sample rate.
- **`frames/frame_NNNN.jpg`**: one frame every 10 seconds (configurable). Used by the agentic teacher for visual technique critique.

The original video stays at `input/{filename}.mp4`. A separate `audio_16k.wav` is generated later (just before the teacher call) for cheaper Files API upload to Gemini.

## Stage 2: analyze

`librosa` does:

- CQT (Constant-Q Transform) at 60 bins/octave, 7 octaves
- Chromagram (for chroma DTW fallback)
- Monophonic pitch tracking (CREPE-style heuristic on CQT energy)
- Per-event pitch detection: emits `pitch_events.json` — one entry per detected note onset with `start_sec`, `note` (pitch+octave name), `pitch_class`, `confidence`, `median_loudness_db`

Used downstream by intonation, HMM alignment, mechanical comments.

## Stage 3: evidence_packet

Synthesizes the `analyze` output plus the score (from masterclass) into a Markdown summary fed to the teacher. Format: per-bar capsule with detected pitches, loudness, off-pulse outliers, and high-level analysis flags. ~1-3 KB depending on lesson length.

## Stage 4: onsets

`engine/onsets.py` runs spectral-flux onset detection on the raw audio. For each onset:
- `time` (sec)
- `note_estimate` (pitch+octave at peak energy)
- `loudness_db`
- `is_strong` (boolean — top-quintile spectral flux)

Used by HMM refinement to anchor bar boundaries on real audio attacks (see `alignment.md`).

## Stage 5: hmm_align

The deepest stage. See `alignment.md` for full details. Briefly:

1. Trim MIDI to the played measure range (auto-detected if user didn't specify)
2. Build a state machine: 1 silence + N note-event states (chords merged)
3. Forward-only sparse Viterbi over CQT pitch energies
4. Refine: snap to detected pitch onsets, smooth residuals
5. Auto-detect actual played `last_measure` from confidence walk
6. Re-trim and re-refine
7. Globally optimize bar boundaries via chain-DP with onset anchors

Output: `hmm_alignment.json` (measure_timestamps, bar_starts with method=`global_dp|expected_only|music_start`, summary stats) + `hmm_aligned_notes.json` (per-note alignment).

## Stage 6: score_map

`build_score_map` joins:
- The masterclass's `score_prep.json` (pages, systems, per-bar bboxes from Audiveris+visual)
- The MIDI score (pitches, beats)
- The HMM alignment (per-note `perf_time`)

Produces `score/score_map.json` with:
- `systems[]`: page, system_index, bbox, bars[]
- `bars[]`: measure, page, system, bbox, system_bbox, highlight_x_frac, alignment_source
- `notes[]`: stable `note_id` (`m{measure}_b{beat:.2f}_{pitch_names_joined}`), measure, beat_in_bar, score_time, perf_time, x_frac, x_frac_end, names, pitch_midi, hmm_confidence, is_chord, is_bar_anchor

This is the structure the player consumes. Note IDs are stable so the teacher's `note_refs` survive a re-alignment.

## Stage 7: intonation

`analyze_intonation` does score-aware CQT intonation analysis. For each played note in `hmm_aligned_notes.json`:
- Locate the corresponding CQT frame range
- Find the dominant pitch peak within ±50 cents of the score pitch
- Report the cents deviation
- Aggregate per-bar (mean cents, std cents, outlier counts)
- Try multiple temperaments (12-TET, just intonation in the piece's key, Pythagorean) and pick the closest

Skipped entirely for keyboard instruments (`profile.pitch_class == "fixed"`).

## Stage 8: rhythm

`analyze_rhythm` per-bar tempo + off-pulse outlier analysis:
- Compute per-bar tempo from bar timestamp deltas
- Compare against MIDI's expected tempo
- Find notes more than 80ms (configurable per profile) off the expected metric position
- Report rubato % per bar, attack-spread distribution, hesitation flags

## Stage 9: voicing (keyboard only)

`analyze_voicing` runs only when `profile.family == "keyboard"`. For each chord in the score:
- Locate the audio window via HMM
- Run a fine-grained CQT on the window
- Rank chord pitches by their spectral peak amplitude
- Compute "top-voice projection" metric: dB difference between the melody note and the next-loudest chord member
- Compute "attack spread" metric: time from earliest to latest detected onset within the chord (in ms)
- Compute "pedal residue" metric: how long the chord's energy persists into the next bar

## Stage 10: mechanical_comments

`generate_mechanical_comments` walks every analysis artifact and emits ~80-120 deterministic per-bar comments (`c001`...`c118`):

- "Bar 5: A4 measured 30 cents sharp"
- "Bar 12: top voice (G5) is 8 dB below the bass"
- "Bar 17: 3 notes off-pulse (>80ms after expected position)"
- "Bar 20: tempo dropped to 0.65× of established pulse"

These are crude but ground-truthed. The teacher reads them in the evidence packet, picks the 8-15 worth elevating, and writes proper musical commentary citing them. The unselected ones land in the `dropped` array of `comments_enriched.json` with one-line reasons (so the user can audit what the teacher decided to skip).

## Stage 11: teach

The agentic Gemini teacher. See `teacher.md` for full details. Briefly:

1. Load the instrument profile → system instruction template
2. Build the 5-layer prompt:
   - Persona (instrument-specific teacher voice + video-critique checklist)
   - Recording briefing (piece, movement, instrument, played measures, student notes, prior lessons)
   - Audio (Files API upload of `audio_16k.wav`)
   - Score system images (Files API upload of relevant pages)
   - Evidence digest + score-note inventory + tool catalog
3. Run a multi-turn `generate_with_tools` loop with 11 tools available
4. Parse the final JSON code block out of the response → `comments_enriched.json`
5. Persist tool-call audit → `teach_tool_calls.json`

## Failure handling

Every per-stage call inside `_run_lesson_jobs` is wrapped in a `run_best_effort(stage, fn)` helper that:
- Catches all exceptions
- Stamps `manifest.metadata[f"{stage}_state"] = "failed"`
- Records the error in `manifest.errors`
- Continues to the next stage

The exceptions to "best-effort" are stages 1-3 (extract_media, analyze, evidence_packet) — those raise to the outer handler which marks the entire session FAILED. Without audio there's nothing to do.

The teacher (stage 11) is also best-effort — if Gemini 503s out the lesson is still marked READY with the deterministic comments, mechanical comments, and (broken) score map. The wizard's processing view shows individual stage states with glyphs (✓ / ✗ / spinner) so the user can see exactly what worked.
