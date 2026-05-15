# Audio ↔ Score Alignment

The hardest engineering problem in v2. We need to know, for every moment of the user's audio, which note (or chord) of the score they're playing. Two derived quantities follow:

- **Bar timestamps**: when does each bar start in the audio? (Used by player to highlight the active system.)
- **Per-note `perf_time`**: when was each score note played? (Used by intonation, rhythm, voicing analyses.)

## Architecture: HMM Viterbi + onset refinement + auto-scope detection

```
align_lesson_with_midi_hmm()
    │
    ├── 1. Load reference MIDI, extract notes
    │       Trim to user-supplied [first_measure, last_measure] OR full MIDI
    │
    ├── 2. Build state machine
    │       1 silence state + 1 state per unique score-time event
    │       (chords = simultaneous notes are merged into 1 state)
    │       For Bach Adagio (22 bars): ~140 states
    │
    ├── 3. Compute observation likelihoods per audio frame
    │       For each frame: high-resolution CQT pitch energies
    │       For each state: log P(observation | state) via:
    │         - direct pitch match (60 bins/octave, ±search_bins tolerance)
    │         - harmonic support (boost score if 2nd-5th harmonics present)
    │
    ├── 4. Sparse forward-only Viterbi
    │       Transitions: self / +1 state / +2 / +3 (no backward jumps)
    │       Hesitation factor: penalize self-transitions less than the prior
    │       Skip penalty: small probability of skipping 1-2 states (missed notes)
    │       Output: most-likely state-per-frame path
    │
    ├── 5. Convert path → per-note alignment
    │       For each score event, find the median frame in its assigned span
    │       perf_time = frame_index * hop_length / sample_rate
    │       Stamp confidence: high (large dwell) / medium / low (interpolated)
    │
    ├── 6. AUTO-DETECT effective played range  ──── NEW (was missing in original PoC)
    │       Walk bars forward from first_measure
    │       For each bar, confidence_ratio = (high+medium notes) / total_notes
    │       Find the LAST contiguous bar with ratio >= 0.4
    │       If last_high_conf_note_time + 8s < audio_total_seconds → trim there
    │       Otherwise keep full range (audio went all the way through)
    │
    ├── 7. If auto-detect trimmed the range → trim alignments and bar_starts to match
    │
    ├── 8. Refine with detected onsets  ──── PORTED FROM PoC
    │       Load pitch_events.json + rich_onsets.json
    │       For each high/medium-confidence aligned note:
    │         find nearest detected onset with same pitch within ±2.5s
    │         compute residual = onset_time - perf_time
    │       Smooth residuals via median over 7-note window
    │       Forward/backward fill stretches with no residual
    │       Apply: new_perf_time = perf_time + smoothed_residual
    │       Effect: HMM's relative timing preserved, global drift corrected
    │
    └── 9. Globally optimize bar boundaries  ──── PORTED FROM PoC
            For each bar in turn, pick the onset that scores highest:
              score = loudness - distance_from_expected*4 - distance_from_HMM*1.5
              constrained to [prev_bar_end + 0.5*expected, prev_bar_end + 1.6*expected]
            Bar 1 anchored to music_start_sec
            Stamp method per bar: "music_start" | "global_dp" | "expected_only"
```

Output: `analysis/hmm_alignment.json` (measure_timestamps + bar_starts + summary diagnostics) and `analysis/hmm_aligned_notes.json` (per-note record).

## Why HMM and not chroma DTW

The original v1 used end-to-end chroma DTW (`engine/alignment.py` is the v2 port, kept as fallback). DTW degenerates badly on:

- Polyphonic music (chord chroma is ambiguous)
- Repeated patterns (bar 5 looks like bar 13 in chroma — DTW happily aligns to either)
- Slow tempo changes / rubato (DTW snaps to the nearest local optimum)

HMM with note-level states + harmonic-support observation model gives ~50ms timing resolution and resists ambiguity by maintaining a posterior. The PoC discovered this empirically; v2 inherits it.

## Auto-detect played range

Before this fix, the user **had to specify `last_measure`** in the upload form, otherwise HMM aligned the user's 9-bar performance against all 22 bars of the Adagio, stretching everything proportionally. The user's bar 9 ended up at audio time 58s (instead of 124s), and the score-following highlighter advanced way ahead of what the user was actually playing.

The fix: walk score bars forward, count high/medium-confidence aligned notes per bar. The last bar before confidence drops to <40% is the effective last bar. Then apply a "trailing silence" check — if the last high-confidence note is more than 8s before the audio ends, trust the trim; otherwise the player went all the way through (and apparent low confidence is just an HMM bug, not silence).

For the Bach BWV 1001 test case (22 bars in MIDI, 9 bars actually played, 132s audio):
- `effective_last_measure: 9` ✓
- Bar 9 starts at 124.5s of 132s ✓
- 8/9 bars anchored to detected strong onsets

For Chopin Op 9 No 2 (full 38 bars played):
- `effective_last_measure: 38` ✓ (auto-detect didn't false-trim)

## Onset refinement details

The HMM's bar timestamps were "perfectly even" (e.g. 6 seconds per bar) before refinement was added. That's because in low-confidence regions Viterbi just emits uniformly-distributed timestamps. The refinement passes anchor the bars to real audio events.

`refine_with_onsets`:
- Uses `pitch_events.json` (monophonic pitch tracker) AND `rich_onsets.json` (spectral-flux onsets) merged. Spectral-flux catches sustained-bow attacks the pitch tracker misses.
- Filters to high+medium confidence + loudness >= -28 dB (drops noise)
- Smoothing window of 7 notes prevents single-pitch ambiguity (an A4 pitch has many candidates within ±2.5s)
- Output: `onset_correction_ms` per note, mean correction reported in summary

`find_bar_boundaries_global`:
- Chain-DP: bar N's start depends on bar N-1's start (no backward refinement)
- Search window: `[prev + 0.5*expected_dur, prev + 1.6*expected_dur]` allows 50%-160% rubato
- Loudness threshold: -22 dB (strong attacks only)
- Reports `method` per bar: `global_dp` (anchored to a real onset), `expected_only` (no onset in window — fell back to evenly-spaced expected time)

For Bach Adagio refined: `notes_corrected: 399/416`, `mean_onset_correction_ms: -62`, `bars_anchored: 8/9`. Before refinement bar timestamps were perfectly even at 6s spacing; after they reflect the actual rubato (4.9s, 14.0s, 19.7s, 25.5s, 34.7s, ...).

## Configuration knobs

`HmmAlignConfig` (in `engine/hmm_align.py`):

```python
sample_rate: int = 22050
hop_length: int = 512  # ~23ms per frame at 22050 Hz
bins_per_octave: int = 60  # ±10 cents per bin
n_octaves: int = 7
search_bins: int = 2  # ±20 cents pitch tolerance per state
harmonic_weight: float = 0.3  # weight for 2nd-5th harmonic energy in observation
hesitation_factor: float = 2.5  # bias toward staying on current state
skip_penalty: float = 0.005  # small probability of skipping 1-2 states
max_note_alignments: int = 2000  # safety cap

# New refinement knobs
refine_with_onsets: bool = True
auto_detect_played_range: bool = True
played_range_confidence_threshold: float = 0.4
played_range_trailing_silence_sec: float = 8.0
```

Per-instrument overrides come from `engine/instruments.py:InstrumentProfile.hmm` — piano gets `hesitation_factor=1.5` (faster transitions), violin gets `2.5` (more sustained notes).

## Output diagnostics

`HmmAlignResult.summary` includes:

```
note_count: int
state_count: int
state_coverage: float (visited / total)
refinement_applied: bool
notes_with_onset_correction: int
mean_onset_correction_ms: float
bars_anchored_to_onsets: int (method=global_dp count)
bars_no_onset_match: int (method=expected_only count)
effective_first_measure: int
effective_last_measure: int
played_range_auto_detected: bool
played_range_method: "user_supplied" | "auto_confidence"
```

The `played_range_method` field tells you whether the user explicitly specified the range or we detected it. The score_map builder respects `effective_last_measure` so notes beyond the played range are not included in `score_map.json`.

## Repair scripts

If alignment looks wrong post-hoc, you can re-run just the HMM stage without re-extracting media:

```powershell
tools\python\python.exe scripts\rerun_hmm_align.py SESSION_ID
```

And rebuild the score_map after that:

```powershell
tools\python\python.exe scripts\rebuild_score_map.py --tenant pqian --user pqian SESSION_ID
```

To regenerate the teacher's comments after fixing alignment:

```powershell
tools\python\python.exe scripts\rerun_teacher.py SESSION_ID
```
