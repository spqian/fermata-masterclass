# UI

Two single-page apps, both vanilla HTML/CSS/JS (no build step, no framework). They live in `apps/api/static/`.

## Wizard ingest UI (`ingest.html`)

Hash-routed multi-page wizard. Routes:

| Route | Purpose |
|---|---|
| `#/` | Landing — Start new class series OR Continue previous |
| `#/new` | Create new masterclass: piece name, movement, instrument profile, score PDF, optional MIDI |
| `#/class/{id}` | Class home — masterclass status, list of lessons, "+ Add a new lesson" |
| `#/class/{id}/lesson` | Lesson upload form — video + optional first/last measure + student notes |
| `#/class/{id}/lesson/{sid}/processing` | 12-stage progress view with glyphs and substages |

Routing is plain `window.addEventListener("hashchange", route)` + a `goto(path)` helper. No framework.

### Friendly status mapping

The 12 pipeline stages have technical names (`hmm_align`, `evidence_packet`, `mechanical_comments`). The wizard maps them to user-friendly phrases via `FRIENDLY_REF` and `STAGE_DEFS`:

```
extract_media         → "Extracting audio and video frames"
analyze               → "Analyzing pitch, rhythm, dynamics"
evidence_packet       → "Compiling the evidence packet"
onsets                → "Detecting note onsets"
hmm_align             → "Matching your performance to the score"
score_map             → "Building the score follower"
intonation            → "Measuring intonation"
rhythm                → "Measuring rhythm and tempo"
voicing               → "Analyzing chord voicing"
mechanical_comments   → "Generating per-measure feedback"
teach                 → "Teacher reviewing your performance"
```

Substages (e.g. score_prep showing "asking gemini-2.5-pro to read the score") are surfaced inline below the stage label, in monospace, dimmed.

### Self-healing polling

Each `processing` view polls `/sessions/{id}` every 2-3 seconds. On focus events (user switches back to the tab), it forces a refresh. If a stage transitions from "ready" back to "running" (rare, indicates a re-trigger), the view picks up the new state. When all stages report ready/failed/skipped, polling stops and a "Open player →" button appears.

### MIDI auto-find UI

When the user creates a new masterclass without uploading a MIDI:
- Status starts "Looking up reference recording online..."
- Substage shows the current Mutopia query being tried
- On success: "Reference found from mutopia." with a green checkmark
- On failure: "No reference recording found — you can upload your own MIDI."

The whole MIDI-find typically takes 5-15 seconds. The wizard never blocks on it — it shows in parallel with score_prep.

### "Which measures did you play?" panel

The lesson upload form has a panel that auto-suggests the bar range:

- Reads the played-movement bar count from `/masterclasses/{id}/score-prep` if available
- Pre-fills "From bar: 1, To bar: {auto-detected count}"
- Both fields are **optional** with placeholder "auto" — the pipeline auto-detects from HMM confidence (see `alignment.md`)
- Help text: "We'll detect the played range from your recording. Set this only if auto-detection guesses wrong."

This was iterated multiple times. Originally the user had to fill it in (alignment was unreliable without). Now it's optional thanks to auto-detect.

## Player UI (`player.html`)

3-column desktop layout. Shows audio + score + comments + teacher's takeaway all simultaneously without scrolling.

```
+-----------------------------------------------------------+
|  HEADER: piece title · class breadcrumb · status          |
+-----------------------------------------------------------+
| TEACHER       |  VIDEO PLAYER          |  COMMENTS        |
| TAKEAWAY      |  + timeline strip      |  [severity      |
|               |                        |   filter chips]  |
| (left col,    +------------------------+  [comment 1]     |
|  ~25%,        |  SCORE                 |  [comment 2]     |
|  scrollable)  |  + page thumbs strip   |  ...             |
|               |  (auto-fit bbox)       |  (right col,     |
|               |                        |   ~30%,          |
|               |                        |   scrollable)    |
+---------------+------------------------+------------------+
| TOOL CALLS / DROPPED COMMENTS (collapsed by default)      |
+-----------------------------------------------------------+
```

Below 1280px: comments wrap to a row below.
Below 900px: full vertical stack.

### Score-following

The score panel shows one system at a time with:

- A yellow bbox highlighting the active system
- A bar number label inside each system at the start of each bar
- The currently-playing bar gets a thicker highlight
- "Now: bar X · Page Y · System Z · ♪ m{measure} {chord_names}" info-bar shows live HMM-matched note as the audio plays

`followCurrentTime(t)` runs on every video `timeupdate` event:
- `currentMeasureForTime(t)` walks `measure_timestamps` to find the most recent bar
- `currentNoteForTime(t)` walks `score_map.notes` to find the currently-sounding note
- `findPageAndSystemForMeasure(m)` looks up which page+system the bar is on
- If the page/system changes, `showSystem(page, sys)` swaps the score image and animates the bbox

The "Following ✓" button toggles auto-follow off if the user wants to manually scrub the score independently of audio.

### Bbox cropping

The score image is rendered inside a fixed-aspect stage. CSS:

```
.score-track .stage {
  flex: 0 0 auto;
  position: relative;
  overflow: hidden;
  background: #fff;
  min-height: 140px;
  transition: height 200ms ease;
}
```

When a system is selected, `updateOverlayAndCrop(system)`:
1. Reads system bbox from `score_prep.json` (visual-detector-derived)
2. Computes scale to fit bbox width to stage width with 4% padding
3. Sets stage height to `bboxPx.h * scale * (1 + padding * 2)` clamped to [140, 560]
4. Translates the full-page score image so the system is centered
5. Draws the overlay rectangle

This was iterated multiple times. Earlier versions had a fixed-height stage (1500px tall) with the system at small scale leaving huge whitespace, then a CSS padding hack that bled into adjacent systems. The current bbox is from the visual barline detector which gives non-overlapping system bboxes by construction, so the stage just sizes to the bbox aspect ratio.

### Comment hover

Hovering a comment in the right column:
1. Reads `comment.references[0]` (the LLM's explicit page+system_index)
2. Calls `showSystem(page, system_index)` if different from current
3. Calls `paintNoteRefBands(notes)` to highlight specific notes within the bar via x_frac

Critical bug fix: the original implementation derived page+system via measure-only lookup against `score_map.notes`. But measures restart per movement (Adagio's bar 6 = Fuga's bar 6 = Siciliana's bar 6 = Presto's bar 6). The lookup picked one ambiguously, often Presto (last movement). Fix: always prefer the LLM's explicit `references[].page/system_index` over derived values.

### Comment severity filter

Three chips: `info`, `warn`, `alert`. Each toggles visibility for that severity. `info` comments are hidden by default (low signal-to-noise — usually camera-angle notes). The chip count updates "9/9 notes" → "7/9 notes" as filters apply.

### Teacher's takeaway panel

Renders the `lesson` block from `comments_enriched.json`:

- Artistic summary (1-2 paragraph overview of the piece's musical character)
- What works (bullet list)
- Areas to develop (priority-tagged list with focus + exercise)
- This week's practice (concrete daily-routine items)
- For next take (camera-angle / repertoire suggestion)

This is "the masterclass" — the high-level pedagogy a real teacher would open and close with. Always visible in the left column.

### Tool call audit

Below the main grid, a collapsible panel showing every tool the teacher invoked:

```
[turn 3]  inspect_chord  ok  · 0.42s    {start_sec: 4.5, end_sec: 6.0}
[turn 5]  measure_tempo  ok  · 0.18s    {start_sec: 30, end_sec: 50}
[turn 7]  watch          ok  · 4.21s    {start_sec: 18, end_sec: 22, question: "bow speed"}
[turn 9]  listen         err · 0.50s    "clip too long..."
```

Status, duration, args (truncated). Plus the teacher's response text per turn (collapsed). Useful for debugging — "did the teacher actually look at the video?" answered immediately.

### Dropped comments expander

Bottom expander showing each mechanical comment the teacher dropped, with the one-line reason. Format:

```
c008   single-note off-pulse outlier — below threshold
c012   absorbed into g_004 (intonation discussion)
c031   tempo measurement noise — practice tempo, not performative
```

Useful for power users debugging "why didn't the teacher comment on bar 17?".

## Brand and styling

Single-file design system using CSS custom properties:

```
--gold       #c9a96a   (primary accent — bar labels, active highlights)
--gold-soft  rgba(...) (secondary accent — borders, dividers)
--ink        #f0f0f0   (primary text)
--ink-soft   #c8c8c8   (secondary text)
--ink-mute   #888       (tertiary text)
--accent     #a4b8d6   (info severity, music notes)
--danger     #d97a6c    (alert severity, errors)
--panel      #161618   (panel background)
--panel-edge rgba(...) (panel borders)
--bg         #0d0d0e   (page background)
```

Typography:
- Headlines: Inter, 600/700 weight
- Body: Inter, 400
- Code/data: JetBrains Mono (bar labels, note names, timestamps)

No framework. No CSS-in-JS. No SASS. Just plain CSS with custom properties + flexbox + grid.

## Why no framework

The whole UI is < 2300 lines of HTML/CSS/JS in 2 files. Adding React + Vite + TypeScript would 10× the build/install footprint and add zero functionality. The state management is simple (URL hash route + a few JS variables); the rendering is mostly static templates with `${...}` interpolation. Pure DOM works fine.

The trade-off: any UI change requires manual DOM manipulation, no hot-reload, no component reuse. For a tool with two routes and one main view, this is fine.
