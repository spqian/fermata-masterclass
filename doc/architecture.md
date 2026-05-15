# Architecture

## Overview

Music Masterclass v2 is a single-process FastAPI app with a heavy bundled toolchain. There is no separate worker, no message queue, no database — just file-system storage, threaded background jobs, and direct LLM calls.

```
                              ┌──────────────────────┐
                              │  Browser (wizard +   │
                              │   player UIs)        │
                              └──────────┬───────────┘
                                         │  HTTP (multipart upload, JSON)
                                         ▼
                              ┌──────────────────────┐
                              │  FastAPI app         │
                              │  apps/api/main.py    │
                              └──────────┬───────────┘
                                         │  threading.Thread per job
                                         ▼
                              ┌──────────────────────┐
                              │  Engine pipeline     │
                              │  src/masterclass/    │
                              │   engine/*           │
                              └──────────┬───────────┘
                                         │
              ┌──────────┬───────────────┼──────────────┬─────────────────┐
              ▼          ▼               ▼              ▼                 ▼
        ┌──────────┐ ┌─────────┐ ┌──────────────┐ ┌─────────────┐ ┌─────────────┐
        │ ffmpeg   │ │ librosa │ │ Audiveris    │ │ Mutopia HTTP│ │ Gemini API  │
        │ (audio + │ │ (CQT,   │ │ + Java JRE   │ │ + Gemini    │ │ (teach,     │
        │  frames  │ │ chroma, │ │ (OMR)        │ │ Flash       │ │  score-prep,│
        │  + clip  │ │ onset)  │ │              │ │ (MIDI find) │ │  watch,     │
        │  encode) │ │         │ │              │ │             │ │  listen)    │
        └──────────┘ └─────────┘ └──────────────┘ └─────────────┘ └─────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  Object storage      │
                              │  local_adls/         │
                              │  (key-value, file-   │
                              │   system backed)     │
                              └──────────────────────┘
```

## Component map

| Path | Purpose |
|---|---|
| `apps/api/main.py` | FastAPI app — REST endpoints, background-job dispatcher, static-file routes |
| `apps/api/static/ingest.html` | Wizard UI for creating masterclasses + uploading lessons |
| `apps/api/static/player.html` | 3-column player UI: teacher's takeaway, video+score, comments |
| `apps/cli/main.py` | CLI for offline/worker-style invocations of individual stages |
| `core/models.py` | Shared dataclasses: `TenantContext`, `MasterclassRef`, `SessionRef`, `MasterclassManifest`, `SessionManifest` |
| `core/sessions.py` | `SessionStore` — load/save lesson session manifests |
| `core/masterclasses.py` | `MasterclassStore` — load/save masterclass (class series) manifests |
| `storage/base.py`, `storage/local.py` | `ObjectStorage` interface + `LocalObjectStorage` filesystem implementation |
| `engine/*` | All deterministic processing stages (see `pipeline.md`) |
| `agent_tools/*` | 11 investigation tools the agentic teacher can call |
| `agent/gemini.py` | Wrapped Gemini client (`generate_json`, `generate_with_tools`, retry, files API) |
| `toolchain/ffmpeg.py` | Bundled-ffmpeg invoker for media extraction |
| `tools/python/` | Bundled Python 3.12 with all deps installed (gitignored) |
| `tools/ffmpeg/bin/` | Bundled ffmpeg static binary (gitignored) |
| `tools/audiveris/` | Bundled Audiveris 5.6.2 (gitignored) |
| `tools/jre/` | Bundled Eclipse Temurin JDK 21 (gitignored) |

## Storage layout

All storage is key-value, served by `LocalObjectStorage`. Keys follow a tenant-scoped hierarchy:

```
tenant/{tenant_id}/users/{user_id}/
├── masterclasses/{masterclass_id}/
│   ├── masterclass.json           ← MasterclassManifest
│   └── reference/
│       ├── score_pdf              ← original PDF upload
│       ├── score_pages/page-NNN.png  ← rasterized at 150 DPI
│       ├── score_prep.json        ← Audiveris+barline output: pages, systems, movements, bars
│       ├── score_musicxml.mxl     ← raw Audiveris output
│       ├── midi                   ← reference MIDI (Mutopia or user-uploaded)
│       └── midi_find.json         ← audit of MIDI auto-find
└── sessions/{session_id}/
    ├── session.json               ← SessionManifest
    ├── input/{filename}.mp4       ← original video upload (canonical key: `input/source_video`)
    ├── artifacts/
    │   ├── audio.wav              ← extracted audio (22050 Hz, mono)
    │   ├── audio_16k.wav          ← downsampled for teacher Files API upload
    │   ├── frames/frame_NNNN.jpg  ← video frames every 10s
    │   ├── listen_clips/*.wav     ← cached audio clips for `listen` tool
    │   ├── watch_clips/*.mp4      ← cached video clips for `watch` tool
    │   └── metadata.json
    ├── analysis/
    │   ├── analysis.json          ← analyze_session output (chroma, pitch features)
    │   ├── evidence_packet.md     ← summary text fed to teacher
    │   ├── pitch_events.json      ← monophonic pitch-track events
    │   ├── rich_onsets.json       ← spectral-flux onsets with note_estimate
    │   ├── hmm_alignment.json     ← measure_timestamps + bar_starts + summary
    │   ├── hmm_aligned_notes.json ← per-note (pitch, perf_time, score_time, confidence)
    │   ├── polyphonic_intonation.json
    │   ├── polyphonic_rhythm.json
    │   ├── voicing.json           ← keyboard-only: chord-balance + dynamics envelope
    │   ├── mechanical_comments.json   ← deterministic per-bar c001..c118 comments
    │   └── teach_tool_calls.json  ← audit of every tool the teacher invoked
    ├── score/
    │   ├── score_map.json         ← combined: notes + bars + systems for player overlay
    │   └── page-NNN.png           ← copy of masterclass score pages (per-session for cache)
    ├── lesson/
    │   ├── comments.json          ← raw teacher JSON before normalization
    │   └── comments_enriched.json ← final v2 teacher output (lesson + comments + dropped)
    ├── llm/raw_teacher_response.json
    └── jobs/{uuid}.json           ← queued-job records (legacy from worker model)
```

## Process model

**No external worker process.** Background jobs run as daemon `threading.Thread` instances spawned by a `_spawn(target, *args)` helper inside the FastAPI app process. Originally this was `BackgroundTasks` from FastAPI but those run sequentially, which deadlocked score_prep waiting for midi_find — see [`limitations.md`](./limitations.md).

Three classes of background work:

1. **Per-masterclass**: `_run_score_prep` (Audiveris OMR + layout), `_run_midi_find` (Mutopia search + Gemini pick + download). These run in parallel when a masterclass is created.

2. **Per-lesson**: `_run_lesson_jobs` runs the full 12-stage pipeline sequentially (see `pipeline.md`). One thread per uploaded lesson.

3. **Per-API-request**: agent tool invocations (`watch`, `listen`, `inspect_*`) run synchronously inside the teacher's tool-call loop, blocking the parent thread. Each tool itself may shell out to ffmpeg or call Gemini.

## Multi-tenancy

Every request must carry `X-Tenant-Id` and `X-User-Id` headers (defaulting to `pqian`/`pqian` for local dev). All storage keys are scoped to the tenant. There's no auth — the headers are trusted, intended for a future identity layer.

A masterclass is a **class series** for one piece (e.g. "Chopin Nocturne Op 9 No 2"). It owns the score PDF + reference MIDI. Multiple lesson sessions can be uploaded against the same masterclass; each session has its own audio/video/analysis/comments but shares the masterclass's score and MIDI.

## Why no database

The app is a self-contained tool you run locally on a developer or musician's machine. All artifacts are immutable on creation (a session's audio.wav doesn't change). File-system + JSON manifests is enough; adds zero ops burden, easy to inspect with `Get-Content`/`cat`. Storage adapters exist for ADLS but are not wired in yet — the `LocalObjectStorage` is the only implementation in production.

## What v2 inherits from v1 (the PoC)

The original PoC is at `C:\Users\pqian\Source\music-masterclass\` (~3,150 lines in `masterclass.py` plus 13 sidecar modules). v2 ports every meaningful capability:

| PoC module | v2 module |
|---|---|
| `score_follower_hmm.py` (HMM Viterbi + onset refine) | `engine/hmm_align.py` |
| `score_follow.py` (chroma DTW fallback) | `engine/alignment.py` |
| `score_align.py` (per-take key-aware align) | merged into `engine/hmm_align.py` |
| `polyphonic_intonation.py` | `engine/intonation.py` |
| `polyphonic_rhythm.py` | `engine/rhythm.py` |
| `piano_voicing.py` + `piano_score_follower.py` | `engine/voicing.py` + `engine/piano_score_follower.py` |
| `rich_onsets.py` | `engine/onsets.py` |
| `tools.py` (11 investigation tools) | `agent_tools/*` |
| `teach.py` (agentic teacher) | `engine/teach_lesson.py` |
| `generate_comments.py` (mechanical) | `engine/mechanical_comments.py` |
| `enrich_comments_prompt.py` (Markdown enrichment) | `engine/enrich_prompt.py` |
| `masterclass.py` (3,150-line orchestrator) | replaced by `apps/api/main.py:_run_lesson_jobs` |

What v2 adds that v1 didn't have:

- **Wizard UI** + **player UI** with score-following overlays (PoC was CLI-only)
- **Audiveris OMR** for score reading (PoC used hand-coded MIDI bar maps)
- **Visual barline detection** (PoC had no per-bar pixel positions)
- **Auto-detected played range** via HMM confidence walking (PoC required user to specify last_measure)
- **MIDI auto-finder** via Mutopia + Gemini Flash hybrid (PoC required user to download MIDI)
- **`watch` tool** for video-clip critique (PoC had only `get_frames` for stills)
- **Multi-tenant storage layout** (PoC was single-user)
- **Cascading score-prep** (Audiveris first, Gemini fallback)
