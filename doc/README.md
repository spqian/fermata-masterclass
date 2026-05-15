# Music Masterclass v2 — Documentation

This folder documents the v2 production app: what it does, how it's built, where the bodies are buried.

| Doc | Subject |
|---|---|
| [`architecture.md`](./architecture.md) | Top-level system architecture, components, storage layout, request flow |
| [`pipeline.md`](./pipeline.md) | Full lesson processing pipeline, stage by stage |
| [`score-prep.md`](./score-prep.md) | PDF → MusicXML → score layout (Audiveris + visual barline detection) |
| [`alignment.md`](./alignment.md) | HMM Viterbi note-level alignment, onset refinement, auto-detect played range |
| [`teacher.md`](./teacher.md) | Agentic Gemini teacher: prompt layers, tool catalog, agentic loop, continuity |
| [`ui.md`](./ui.md) | Wizard ingest UI + player UI |
| [`tooling.md`](./tooling.md) | Bundled toolchain (Python, ffmpeg, Audiveris, JRE), MIDI auto-finder |
| [`data-model.md`](./data-model.md) | Every persistent JSON shape: manifests + engine artifacts |
| [`extending.md`](./extending.md) | How to add new instruments, agent tools, score sources, pipeline stages |
| [`operations.md`](./operations.md) | How to run, environment variables, authentication, debugging, repair scripts |
| [`byo-key.md`](./byo-key.md) | BYO Gemini API key model, billing, model choices, encryption |
| [`limitations.md`](./limitations.md) | Known issues, hacks, design tradeoffs, what to invest in next |
| [`decisions.md`](./decisions.md) | Decision log: meaningful design choices and what we learned |
| [`roadmap.md`](./roadmap.md) | What's next: auth, BYO Gemini key, chat-with-teacher, Azure hosting |

## TL;DR

A user uploads:

- A **piece description** (e.g. "Bach BWV 1001 Adagio for violin solo")
- A **score PDF** (e.g. an IMSLP edition)
- A **performance video** (their own playing)

In ~3-5 minutes the system produces a structured **masterclass critique**:

- A high-level "teacher's takeaway" (artistic_summary, what_works, areas_to_develop, this_week_practice, next_take)
- 5-15 time-anchored **comments** with severity, references to specific notes/bars
- A **player** that follows the audio along the score with bar-level highlighting
- Per-comment audit of what tools the teacher used (intonation measurements, dynamics, listen, watch)

The reference MIDI is **automatically found** on Mutopia.org (no user upload needed in the common case).

## Headline numbers

- ~14,300 lines of Python under `src/masterclass`
- ~2,300 lines of HTML/CSS/JS for the wizard + player
- 4 storage entities: tenant, user, masterclass, session (lesson)
- 11 agent tools the teacher can call mid-conversation
- 5 prompt layers fed to the teacher per lesson
- 12-stage processing pipeline per lesson
- 3 LLM models in use: `gemini-2.5-pro` (teacher + score-prep), `gemini-2.5-flash` (MIDI picker, query normalizer), Audiveris OMR (deterministic, no LLM)

## Where to start reading

1. **New to the codebase?** Start with [`architecture.md`](./architecture.md) for the high-level system shape.
2. **Want to understand a lesson run?** Read [`pipeline.md`](./pipeline.md) — it walks through every stage from upload to player.
3. **Debugging score alignment?** [`score-prep.md`](./score-prep.md) and [`alignment.md`](./alignment.md).
4. **Curious about how the LLM teacher works?** [`teacher.md`](./teacher.md).
5. **Operating the app?** [`operations.md`](./operations.md).
