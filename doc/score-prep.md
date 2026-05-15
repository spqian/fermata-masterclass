# Score Preparation: PDF → Layout

The score-prep stage runs once per masterclass (when the PDF is uploaded). Output: `reference/score_prep.json` with movements, pages, systems, and per-bar bboxes — everything the player needs to highlight a measure on the rendered page.

This module went through three architectures before settling. The current one is a cascade.

## Final pipeline (cascading)

```
score_prep.py:prepare_score()
    │
    ├── 1. Rasterize PDF → page-NNN.png at 150 DPI (PyMuPDF)
    │
    ├── 2. Wait up to 90s for midi_find (parallel job) so MIDI cross-check has data
    │
    ├── 3. Try Audiveris OMR ──┐
    │       │                  │ on success: deterministic, ~90s, no LLM cost
    │       │                  ▼
    │       │              MusicXML parsed → movements/systems/measures
    │       │              (engine/score_layout_from_musicxml.py)
    │       │
    │       └── on failure: Gemini 2.5 Pro fallback (legacy path)
    │           Gemini reads page rasters and emits SCORE_PREP_RESPONSE_SCHEMA
    │           Used to be primary; replaced because it failed on dense scores
    │
    ├── 4. Visual barline detection per page (engine/barline_detection.py)
    │       ├── Horizontal projection → staff lines → systems with non-overlapping bboxes
    │       ├── Per-system: vertical projection → barline columns
    │       ├── Music-start-x detection (clef + keysig prefix valley)
    │       └── REPLACES Audiveris's broken bboxes; keeps Audiveris note content
    │
    ├── 5. MIDI cross-check + redistribute_movement_bars_to_midi
    │       ├── Identify played movement (URL hint: bwv-1001_1.mid → mvmt 1)
    │       ├── If Audiveris's bar count for the played movement disagrees with MIDI ±2,
    │       │   redistribute MIDI's count uniformly across detected systems
    │       └── Sets manifest.metadata["played_movement_id"]
    │
    └── 6. Stamp metadata
            ├── score_prep_source: "audiveris" | "gemini"
            ├── score_prep_layout_measure_count
            ├── score_prep_played_movement_measure_count
            ├── score_prep_midi_measure_count
            ├── played_movement_id
            └── score_prep_redistributed_to_midi
```

## Audiveris (`engine/audiveris_omr.py`)

Wrapper around the bundled Audiveris 5.6.2 + Eclipse Temurin JDK 21 (under `tools/audiveris/`, `tools/jre/`). Run via subprocess in batch mode:

```
java -jar audiveris.jar -batch -export -output {tmpdir} {pdf}
```

Output: `.mxl` (compressed MusicXML, ~50 KB for a 7-page violin sonata). Wall-clock ~90 seconds for a 7-page PDF on CPU.

Failure modes:
- PDF has unusual fonts → some symbols misclassified
- Extremely dense Baroque ornament passages → spurious barlines
- Hand-marked / scanned facsimile editions → degraded accuracy

When Audiveris errors out, the wrapper raises `RuntimeError` with stderr captured; `score_prep.py` catches and falls back to the Gemini path.

## MusicXML → score_prep schema (`engine/score_layout_from_musicxml.py`)

Audiveris's MusicXML is parsed with `lxml`. We extract:

- **Movements**: detected by tempo/time/key signature changes within a single concatenated part. For Bach BWV 1001 this correctly produces 4 movements (Adagio, Fuga, Siciliana, Presto). Each movement gets its own `id`, `title` (from text marker or "Movement N"), `tempo_marking`, `time_signature`, `key_signature`, `start_page`, `end_page`, `first_measure`, `last_measure`, `measure_count`. **Measure numbering restarts per movement** (Adagio = 1..22, Fuga = 1..91, etc.).

- **Spurious barline filtering**: drops any "measure" with 0 notes, duration < 25% of the time-signature's barlength, or `<barline location="middle">`. Surviving measures are renumbered contiguously.

- **System bboxes**: from `<system-layout>` y-coordinates. **These are unreliable** — Audiveris reports just the staff strip (~4% of page height), not the visual extent (stems, slurs, ornament marks). The visual barline detector replaces these.

- **Per-bar pixel positions**: Audiveris does NOT emit `<measure-layout>`. We have no per-bar pixel info from MusicXML. Bars are positioned by visual barline detection (below).

## Visual barline detection (`engine/barline_detection.py`)

The standalone CV pass that's the actual layout source-of-truth. **Algo A** ("Algorithm A") was the winner of a 4-algorithm bake-off (horizontal+vertical projection vs Hough transform vs connected components vs LSD; the prototype script is not committed):

```
For each page raster:

1. Convert to grayscale numpy array.

2. Horizontal projection → staff lines:
     row_density[y] = sum(dark_pixels in row y)
     Find peaks above threshold = staff line candidates
     Group peaks into 5-line clusters = single staves
     Group adjacent staves (within ~80px) into "systems" (handles piano grand staff)
     For each system, define non-overlapping bbox:
       y_top = staff_top - margin
       y_bot = staff_bottom + margin
       where margin = (gap_to_next_system / 2)
       so bboxes never overlap

3. Per-system: vertical projection → barline columns:
     column_density[x] = sum(dark_pixels in column x within staff y-range)
     Find peaks above threshold = barline candidates
     For each peak: walk left and right while density > 30% of peak
       → column_extent = [left_x, right_x]
     Use right_x as the divider position (next bar starts immediately after)

4. Music-start-x detection per system:
     Walk column_density from left, find first "valley":
       density < 15% of max, sustained for >= staff_line_spacing * 2
     This is the boundary between clef+keysig prefix and music
     Fallback: 8% of system width if no valley found

5. Bar count validation against MIDI:
     If detected bars per system summed != MIDI bar count for the movement
     fall back to uniform redistribution within [music_start_x, system_end_x]
```

### Performance

- ~40ms per page on CPU
- Bach BWV 1001 page 1: 10 systems detected, bars distributed [2,3,2,2,2,2,2,2,2,3] across 22 MIDI bars
- Chopin Op 9 No 2 page 2: 6 systems, 38 MIDI bars across 2 pages

### Algorithm bake-off results (Bach + Chopin page 1)

| Algorithm | Bach (10 sys, 22 bars) | Chopin (6 sys, 38 bars) | Speed |
|---|---|---|---|
| **A: H+V projection** | 22/22 ✓, 0 overlaps | 21/38 (uniform fallback fired), 0 overlaps | 17-40ms |
| B: Hough lines | 22/22, 0 overlaps | 16/38 | 50-67ms |
| C: Connected-components | 20/22 | 12/38 | 21-22ms |
| D: LSD line detector | 20/22 + 2 false-positives | 12/38 | 62-76ms |

Algo A won on cross-page accuracy + zero-overlap guarantee + speed.

### Known limitations of barline detection

- Bar-label placement is accurate to **±5-15 pixels** on most systems. Within 1 bar, never wrong by more than 1/3 of the bar width.
- The 8% music-start fallback is wrong for Bach single-staff with treble-clef-only (real prefix ~10%) and Chopin grand staff with brace+clefs+keysig (real prefix ~12%). Adaptive valley detection handles this most of the time but degrades on dense leftmost notes.
- Half-width / cautionary barlines (mid-bar style) can be picked as full bars; the MIDI-count cross-check usually catches these.
- See `limitations.md` for what to invest in next (per-bar coords from Audiveris JSON export, or a custom OMR pass).

## MIDI URL → played movement hint

Mutopia URLs follow the pattern `bwv-1001_N.mid` where N is the movement number. `score_prep.py` parses this:

```python
def _parse_movement_hint_from_midi_url(url: str) -> int | None:
    m = re.search(r'_(\d+)\.midi?$', url)
    return int(m.group(1)) if m else None
```

If a hint exists AND the corresponding movement was detected by Audiveris, we set `played_movement_id = N` regardless of whether bar counts match. Otherwise we fall back to "movement whose bar count is closest to MIDI's bar count". This was an iterative fix — earlier versions blindly picked closest-count which sent Bach BWV 1001's MIDI (22 bars Adagio) to Siciliana (21 bars).

## Cascading fallback contract

`prepare_score()` always tries Audiveris first and Gemini second. If both fail, it raises and the masterclass is marked failed. Gemini fallback is preserved because:

- Audiveris depends on Java + ~250 MB of bundled assets
- Gemini's score reading, while unreliable for dense music, works on PDFs Audiveris chokes on (e.g. modern non-engraved scores, hand-marked scores)
- Some users may want to disable Audiveris entirely (env: `MASTERCLASS_DISABLE_AUDIVERIS=1`)
