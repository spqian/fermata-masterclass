# Known Limitations & What to Invest In Next

This is the "honest list" of where the system breaks, where we accepted technical debt, and what would meaningfully improve quality.

## Score layout / barline alignment

**Bar labels are accurate to ±5-15 pixels on most systems.** Within 1 bar, never wrong by more than 1/3 of the bar width. This is good enough for score-following but visibly off if you look closely.

Root cause: Audiveris's MusicXML doesn't include `<measure-layout>` (per-bar pixel coordinates). We have:
- System bboxes from Audiveris (broken — too tight, replaced)
- System bboxes from visual barline detection (good)
- Per-bar pixel positions from visual detection (decent — ±5-15px)

What would fix it:
1. **Use Audiveris JSON export instead of MusicXML** — Audiveris's internal format includes per-glyph pixel positions. Would require writing a JSON parser instead of a MusicXML parser.
2. **Switch to oemer** — different open-source OMR that explicitly emits per-bar bboxes. Newer, less mature than Audiveris.
3. **Custom Audiveris plugin** — invasive, requires Java skill.
4. **Refine the visual detector** — currently uses fixed 8% music-start fallback. Adaptive valley detection helps but still 5-10px off in places. A trained CNN for barline detection would crush this but adds ML weight + training pipeline.

User explicitly accepted current state ("this maybe tech limitation. We can invest in this later").

## Score-prep wall-clock time

90 seconds per masterclass to run Audiveris on a 7-page PDF. This is CPU-bound, single-threaded inside the JVM. Acceptable for lesson uploads (the user already waits 3-5 min for the full pipeline) but feels long when the user is creating a masterclass and just wants to start uploading lessons.

What would help:
- Pre-warm the JVM (currently we cold-start `java.exe` per invocation)
- Run Audiveris incrementally per page in parallel — the Audiveris CLI doesn't support this directly
- Skip Audiveris entirely if MusicXML is found in a catalog (none of the catalogs we use ship MusicXML, but if KernScores / OpenScore expand coverage this becomes viable)

## Gemini 503 capacity throttling

Gemini 2.5 Pro hits 503 UNAVAILABLE during peak hours. Our retry logic does 3 attempts with 5/10/15s backoff but if all 3 fail the teacher is marked failed. The user has to manually retry via `scripts/rerun_teacher.py` after capacity returns.

Could be improved:
- Longer exponential backoff (currently caps at 15s; could go to 60-120s)
- Auto-fallback to Gemini Flash on persistent 503 (Flash has more capacity, ~80% quality)
- Persist a "retry queue" and have a background loop attempt failed teachers periodically

Currently no automation — user-driven retry only.

## Auto-detect played range — false trims

The HMM confidence walk works well for clean audio (Bach BWV 1001 user case: trims to bar 9 correctly). But it can false-trim if:
- Player attempts a passage but flubs every note (low confidence even though audio exists)
- Score has multi-bar rests that the player observes (high silence in middle of piece)
- Player has very quiet practice passages

Mitigations in place:
- "Trailing silence check" — if last high-conf note is within 8s of audio end, we don't trim
- User can override via the `last_measure` field in the upload form
- `played_range_method` field in metadata signals "user_supplied" vs "auto_confidence" so this is auditable

Not yet handled:
- **Mid-piece skipped sections** — algorithm assumes contiguous play from first_measure to last_measure. A player who plays bars 1-9 then jumps to 14-22 will get bars 10-13 phantom-aligned to whatever audio exists at those time positions.

## Teacher's bar references can drift after alignment changes

If you re-run alignment after the teacher already ran, the teacher's `references[].page/system_index/note_id` may now point to slightly different score positions. The teacher used the OLD score_map to write its critique.

We mitigated by ensuring `note_id` is stable across alignment changes — `m1_b1.00_G5+Bb4+D4+G3` always means the same chord regardless of how the alignment shifts. So hover navigation is correct.

What's not perfect: the teacher's TEXT might say "bar 5" but with new alignment that audio time is now in bar 4. The user has to re-run the teacher (`scripts/rerun_teacher.py`) to regenerate text matching the new alignment.

## MIDI auto-finder coverage

Mutopia has ~2000 pieces, mostly classical. If the user requests something not on Mutopia (jazz, contemporary, transcriptions, lesser-known repertoire), MIDI find returns `not_found` and the user has to upload their own MIDI.

Not implemented:
- IMSLP MIDI search (IMSLP has many MIDIs but no clean search API; would need scraping)
- Open MuseScore.com (MIDI for free public-domain works)
- KernScores → MusicXML → MIDI conversion (Stanford's **kern catalog has near-complete classical repertoire)

## Single-process, single-machine

The whole app runs in one Python process. No worker pool, no message queue, no distribution.

Implications:
- Can only run one lesson at a time per machine without resource contention (Audiveris alone uses ~2 GB JVM heap)
- A crash during the teach stage takes the whole API down
- No retries on background-thread failure
- File-system storage means no shared deployments — each user has their own local copy

For a v3 that targets multi-user / production deployment, this would need:
- Object storage (ADLS adapter exists but isn't wired in)
- Worker queue (RabbitMQ or similar)
- Multi-tenant RBAC layer
- Probably containerization

## UI limitations

- **No mobile layout** below 900px screen width. Player gracefully stacks but the score is too wide for phones.
- **No real-time updates** — wizard polls every 2-3s rather than using SSE/WebSockets.
- **No keyboard shortcuts** for play/pause, scrub, comment navigation.
- **No edit mode** for the teacher's comments — read-only output.
- **No export** to PDF / shareable URL.
- **Chat has no streaming** — replies arrive as one response after the synchronous Gemini turn completes.
- **No chat edit/regenerate flow** — past messages are immutable JSON records unless the whole conversation is deleted.
- **Single-thread chat UX** — the backend supports multiple lesson chat threads, but the player loads the most recent one and does not expose a thread switcher yet.
- **Chat context is lesson-only** — by design it does not pull prior lessons or other masterclasses into the follow-up prompt.

## Test coverage

There is no committed test suite. The author's working tree has ~30 one-off `scripts/test_*.py` smoke runners (each tied to a specific local recording / IMSLP download) — these are gitignored because the paths are hardcoded. There is no `pytest`, no CI.

What would help:
- Unit tests for `engine/barline_detection.py` (deterministic, no LLM)
- Snapshot tests for `engine/score_prep.py` output on the canonical Bach + Chopin PDFs
- HMM alignment regression tests with known-good measure_timestamps
- A `make test` target that runs these in 2-3 minutes

## What to invest in next (priority order)

1. **Better barline placement** (per-bar from Audiveris JSON OR a CNN). High user-visible impact, current state "tolerable but visibly off".

2. **Auto-retry on Gemini 503** — current state "user has to manually rerun". Easy fix: persist failed-teach state, background loop with longer exponential backoff (5min, 15min, 1hr).

3. **MIDI catalog expansion** — KernScores integration would 5-10× the coverage. Each new catalog source is ~1 day of work.

4. **Mid-piece skip detection** — extend `detect_effective_last_measure` to find contiguous segments instead of one final cutoff. ~1 day of HMM work.

5. **Real test suite** — `pytest` + a few snapshot tests. ~1 day to set up, pays back forever.

6. **Mobile-friendly player layout** — repurpose existing 3-column grid. 1-2 days.

7. **Audiveris JVM warmup** — keep a JVM process alive between requests. Avoids 5-10s startup cost per score-prep. ~1 day.

8. **Gemini Pro/Flash A/B comparison** — currently always Pro for teacher; Flash is 4× cheaper and might be 80% as good for routine lessons. Worth empirical study.

