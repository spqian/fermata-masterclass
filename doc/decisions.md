# Decision Log

A chronological record of meaningful design decisions, the alternatives considered, and what we learned. Useful when revisiting "why is this thing the way it is?"

## Score reading: Gemini → Audiveris

**Decision**: switched from Gemini-as-OMR to Audiveris OMR + Gemini as fallback only.

**Why**: Gemini Pro repeatedly failed on standard scores. Misclassified piano grand staff as 6 separate systems (treble + bass counted independently). Took 80+ seconds per page. Gave the wrong measure count by 40%+ on Chopin Op 9 No 2. Timed out entirely on Bach BWV 1001 (7-page Bärenreiter edition).

**Alternative considered**: KernScores or OpenScore for catalog MusicXML — would have skipped OMR entirely. Coverage too patchy outside core repertoire.

**Trade-off**: Audiveris adds ~430 MB to bundled tools (Java + Audiveris). Cold-start ~5-10s per call. But it's deterministic, free at runtime, and works.

## Layout source: Audiveris MusicXML → visual barline detection

**Decision**: Audiveris emits unreliable system bboxes (just the staff strip, no padding) and **no per-bar pixel positions** in MusicXML. We replaced its layout output with a visual CV pass that does horizontal+vertical projection on the page raster.

**Why**: Audiveris's `<system-layout>` reports y-coords for staff lines only. We need the visual extent including stems, slurs, ornament marks — otherwise the player's highlight bbox bleeds into adjacent systems. And we need per-bar coords for bar labels.

**Alternative considered**:
- Audiveris JSON export (more verbose, includes per-glyph positions) — would require writing a JSON parser; deferred.
- oemer (different OMR with explicit barline positions) — newer, less mature.
- CNN-based barline detector — would need training data + ML pipeline.

**Trade-off**: visual detector is accurate to ±5-15 pixels per bar position. User accepted this is a tech limitation worth deferring. The system is good enough for score-following but visibly off if you look closely.

## Alignment: chroma DTW → HMM Viterbi (inherited from PoC)

**Decision**: HMM Viterbi note-level alignment, not end-to-end chroma DTW.

**Why**: DTW degenerates on polyphonic music with repeated patterns. PoC discovered this empirically. HMM with note-level states + harmonic-support observation model gives ~50ms timing resolution and resists ambiguity by maintaining a posterior.

**Alternative considered**: end-to-end neural alignment (e.g. nnAudio + transformer) — overkill for our use case.

## HMM refinement: bar-boundary chain DP added

**Decision**: after raw Viterbi, run a chain-DP that re-anchors bars to detected strong onsets.

**Why**: raw Viterbi in low-confidence regions produced perfectly-even bar timestamps (6.0s spacing across an Adagio with rubato). The user's bar 9 ended up at audio time 58s instead of 124s. Refinement closes this gap by anchoring to real audio events.

**Inherited from PoC**: yes, but never ported to v2 originally. Found this regression late in the session, ported `refine_with_onsets` and `find_bar_boundaries_global` from `score_follower_hmm.py`.

## Auto-detect played range

**Decision**: walk HMM-aligned notes by bar, find the last bar with confidence_ratio >= 40%, trim alignment + score_map to that range.

**Why**: user shouldn't have to specify "I played bars 1-9" in the upload form. Original v1 required this. v2 originally also required this. Got rebuilt as auto-detection because the user kept forgetting and getting bad alignment as a result.

**Trade-off**: false-trims if the player flubs the last bars (low confidence even though audio exists). User can override via the explicit `last_measure` field.

## MIDI auto-finder: hybrid Mutopia + Gemini Flash pick

**Decision**: programmatic Mutopia HTML scraping + Gemini Flash to select best candidate from a structured list.

**Why originally tried Gemini grounded search**: thought Gemini-as-MIDI-finder would Just Work. It hung for minutes on flaky 504s and gave hallucinated URLs.

**Why current approach works**: Mutopia's `make-table.cgi` returns a deterministic candidate list. Gemini Flash gets the candidates as JSON and is constrained to pick one of the URLs verbatim — no hallucination. ~$0.001 per masterclass, 11s end-to-end.

**Iteration history**:
1. Initial parser misaligned fields across pieces (Bach title paired with Beethoven URL). Fixed by switching to "one nested table per piece" parser.
2. Original normalize-query prompt was too generic. Refined to suggest catalog-id queries (BWV/KV/Op.) which Mutopia indexes well.
3. Originally stopped at first hit. Now accumulates across multiple queries for richer LLM choice.

## Background jobs: BackgroundTasks → threading.Thread

**Decision**: replaced FastAPI's `BackgroundTasks` with daemon `threading.Thread` instances spawned via a `_spawn` helper.

**Why**: `BackgroundTasks` runs jobs **sequentially** within a single request's background queue. Score_prep was waiting up to 90s for midi_find to complete (so MIDI cross-check could fire). With sequential execution they deadlocked.

**Trade-off**: real threads have to manage their own exception handling (we use a try/except in each thread target). No retry semantics.

## Wizard UI: turbotax-style multi-page

**Decision**: hash-routed wizard with one focused page per step (start → new class → class home → upload lesson → processing).

**Why**: Original single-page UI was busy and confusing. User asked for "turbotax style" — one decision per page.

**Trade-off**: more clicks. But each page is unambiguous; users don't misclick.

## Player UI: 3-column layout

**Decision**: video + score in middle column, teacher's takeaway left, comments right. All visible without scrolling.

**Why**: User explicitly required "comment and video and score should be visible at the same time. They cross reference each other." Earlier 2-column layout buried comments below the fold.

**Iteration**: started single-column stack → 2-column (video+score top, teacher+comments side-by-side below the fold) → 3-column (current). Each iteration was driven by user feedback.

## Continuity context: server-builds, force-RAG into prompt

**Decision**: when a user uploads a 2nd lesson, server builds `context/prior_lessons.json` from sibling sessions and force-includes it in the teacher's prompt. NOT exposed as a tool.

**Why force-RAG instead of tool**: continuity is too important to leave to the model deciding whether to ask. The teacher would routinely skip "did you check what you said last time?" if it could.

**Iteration**:
1. Initial implementation built the file but `_read_past_comments` looked for the wrong storage key (`player/comments_enriched.json` vs actual `lesson/comments_enriched.json`). Result: `teacher_comments: []` always.
2. Even after fixing the key, the prior context was missing the lesson `lesson` block (artistic_summary, what_works, areas_to_develop, etc.) — only individual comments. Lost the most important continuity signal.
3. Format was raw JSON dump. Reformatted as structured Markdown the teacher can act on.
4. Added explicit "Continuity" section to system instruction directing the teacher to: open with diff, write progress_notes as before/unchanged/new, address last week's prescribed practice, escalate unresolved areas with new drills.

## Comment hover: reference-page resolution bug

**Bug**: hovering on Bach Adagio comments sent the score view to Presto (page 6).

**Root cause**: the `/lessons/{id}/comments` endpoint was overwriting the LLM's explicit `references[].page/system_index` with a measure-only lookup against the full score_map. Since "bar 6" exists in 4 movements (Adagio, Fuga, Siciliana, Presto), the lookup returned an arbitrary one — usually the last (Presto).

**Fix**: preserve the LLM's explicit refs, only fall back to lookup if missing. Plus client-side `note.system` composite fallback (`100*page + system_on_page`) was also using the composite as a sys_on_page in some code paths — fixed.

## Video critique: dedicated `watch` tool

**Decision**: added a `watch` tool that extracts a short MP4 (≤10s, downscaled, no audio) and sends to Gemini.

**Why**: User asked "how does it tell bow speed from looking at a frame?" Stills can't show motion. The new `watch` tool gives the teacher real video for motion-based judgments.

**Per-instrument video checklists**: added to InstrumentProfile so each instrument's teacher prompt includes specific things to look for (bow contact for strings, hand position + pedal for piano, etc.).

## Teacher prompt: 5-layer construction

**Decision**: every teacher invocation sends 5 layers — persona system instruction (instrument-aware), recording briefing, audio (Files API), score images (Files API), evidence digest + score-note inventory + tool catalog.

**Why this many layers**: each carries different information the teacher needs. Splitting them lets the model attend to the right thing at the right moment. Tested compressing into one block — quality dropped (model lost track of which audio file vs which score image).

**Cost**: typical lesson $0.30-2.00 per teacher call. Pro multimodal is expensive. Worth it for the output quality.

## Bundled toolchain in `tools/`

**Decision**: bundle Python + ffmpeg + JRE + Audiveris under `tools/`, gitignored.

**Why**: makes the whole repo runnable on a fresh Windows machine without managing deps. Total ~1 GB download but only once.

**Alternative considered**: Docker — would simplify deps but adds Docker Desktop as a requirement. Most music-software users don't have Docker.

## Storage: file-system, no database

**Decision**: `LocalObjectStorage` rooted at `local_adls/` with a tenant/user/{masterclass|session}/{id} key shape.

**Why**: app is a self-contained tool. No multi-user serving today, no need for a query layer. JSON manifests on disk are easy to inspect with `Get-Content`/`cat`.

**Future**: ADLS adapter exists but isn't wired in. When we go multi-tenant SaaS, we'll switch.

## Documentation: doc/ markdown, not website

**Decision**: write architecture docs as Markdown under `doc/`, not as a Sphinx/MkDocs site.

**Why**: code is the source of truth; docs are an entry point. Markdown rendered in IDE/GitHub is enough for a repo-internal audience. No build step.
