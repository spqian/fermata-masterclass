# Fermata Masterclass

> An LLM-powered masterclass coach for musicians — multimodal video + audio + score analysis.

A user uploads a **score PDF**, a **performance video**, and a short description of the piece. In a few minutes, Fermata returns a structured masterclass critique:

- a teacher's **takeaway** (artistic summary, what works, areas to develop, this week's practice, plan for next take)
- 5–15 time-anchored **comments** with severity, references to specific bars/notes, and the audio/video evidence the teacher used
- a **player** that follows the audio along the score with bar-level highlighting
- **continuity across takes** — subsequent recordings are critiqued in the context of prior lessons

The reference MIDI is automatically located on Mutopia.org for most catalog repertoire — no upload needed in the common case.

## How it works (10-second version)

```
PDF score   ──►  Audiveris OMR + visual barline detection  ──►  score layout
performance ──►  ffmpeg → audio + frames                   ──►  HMM Viterbi alignment
                                                                    │
                                                                    ▼
                              Gemini 2.5 Pro agentic teacher  ──►  comments + lesson
                              (with tools: listen / watch /
                               intonation / dynamics / ...)
```

Three model classes work together:

- **Audiveris** — deterministic optical music recognition (PDF → MusicXML)
- **Gemini 2.5 Flash** — fast utility tasks (MIDI picker, query normalizer)
- **Gemini 2.5 Pro** — the agentic teacher, given audio, score images, video frames, mechanical measurements, and prior-lesson context

## Documentation

Full docs live in [`doc/`](./doc/README.md). Recommended reading order:

1. [`doc/architecture.md`](./doc/architecture.md) — system shape, storage, components
2. [`doc/pipeline.md`](./doc/pipeline.md) — the 12-stage lesson pipeline
3. [`doc/score-prep.md`](./doc/score-prep.md) — Audiveris + visual barline detection
4. [`doc/alignment.md`](./doc/alignment.md) — HMM Viterbi + onset refinement
5. [`doc/teacher.md`](./doc/teacher.md) — the agentic Gemini teacher (5 prompt layers, 11 tools, continuity)
6. [`doc/operations.md`](./doc/operations.md) — running, debugging, repair scripts

Reference material:

- [`doc/data-model.md`](./doc/data-model.md) — every persistent JSON shape
- [`doc/extending.md`](./doc/extending.md) — adding instruments, tools, score sources, models
- [`doc/decisions.md`](./doc/decisions.md) — design decision log + bug histories
- [`doc/limitations.md`](./doc/limitations.md) — known issues, what to invest in next

## Quick start

```powershell
# 1. Clone
git clone https://github.com/spqian/fermata-masterclass.git
cd fermata-masterclass

# 2. Install bundled toolchain (Python, ffmpeg, Audiveris, JRE)
#    Downloads ~2 GB. See tools/README.md.
.\scripts\install_tools.ps1

# 3. Install Python deps
tools\python\python.exe -m pip install -e ".[api,llm]"

# 4. Set your Gemini API key
"GEMINI_API_KEY=your-key-here" | Out-File -Encoding ascii .env

# 5. Run the API + UI
tools\python\python.exe -m uvicorn masterclass.apps.api.main:app --host 127.0.0.1 --port 8770
```

Open <http://127.0.0.1:8770> for the wizard.

See [`doc/operations.md`](./doc/operations.md) for environment variables, debugging, and repair scripts.

## Repository layout

```
src/masterclass/         Application code (~14k lines)
  engine/                Score prep, HMM alignment, teacher orchestration
  agent_tools/           Tools the LLM teacher can call (listen, watch, intonation, ...)
  apps/api/              FastAPI app + wizard + player UI
  apps/cli/              CLI entry point
scripts/                 Repair / re-run / debug scripts
doc/                     All documentation
tools/                   Bundled binaries (gitignored — installed via install_tools.ps1)
```

## Status

Working v2 with end-to-end Bach BWV 1001 + Chopin Op 9 No 2 lessons verified. See [`doc/limitations.md`](./doc/limitations.md) for what's still rough.

## License

Apache 2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

Third-party tools (Audiveris, JRE, ffmpeg) are downloaded at install time and retain their respective licenses; they are not redistributed in this repository.
