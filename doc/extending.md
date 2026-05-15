# Extending v2

How to add new instruments, agent tools, score sources, or pipeline stages without breaking the existing flow.

## Adding a new instrument profile

Edit `engine/instruments.py:BUILTIN_PROFILES`. Each profile is an `InstrumentProfile` dataclass with:

```python
"clarinet_solo": InstrumentProfile(
    id="clarinet_solo",
    instrument="clarinet",
    family="woodwind",                    # used for tool gating
    pitch_class="variable",                # "fixed" disables intonation tools
    polyphony="low",                       # "low|medium|high" — affects HMM tuning
    max_simultaneous_voices=1,
    intonation={"enabled": True, "search_cents": 50, "temperaments": ["12_tet"]},
    rhythm={"min_onset_spacing_ms": 100},
    hmm={"search_bins": 2, "harmonic_weight": 0.4, "hesitation_factor": 2.0, "skip_penalty": 0.005},
    onset_detection={"peak_delta": 0.06, "min_spacing_ms": 100},
    comment_generator={"intonation_threshold_cents": [15, 25], "rubato_threshold_pct": [15, 30, 50]},
    teacher_examples="Sabine Meyer, Anthony McGill, Martin Fröst",
    voicing_focus=DEFAULT_VOICING_FOCUS,
    category_guidance=DEFAULT_CATEGORY_GUIDANCE,
    measurements_available="intonation cents, HMM alignment, onsets, breath events, dynamics envelope",
    measurable_claims_rule=DEFAULT_MEASURABLE_CLAIMS_RULE,
    intonation_or_voicing="intonation",
    video_checklist=DEFAULT_VIDEO_CHECKLIST,  # or write a clarinet-specific one
    disabled_tools=("inspect_voicing",),       # disable piano-only tools
)
```

Also add to `_ALIASES` if you want shorthand resolution (e.g. `"clarinet": "clarinet_solo"`).

If your instrument needs special teacher framing (e.g. embouchure for winds, breath support), write a custom `video_checklist` and `category_guidance` constant and reference them in the profile.

The wizard's instrument-profile dropdown is populated from the registry. New profiles automatically appear after restart.

## Adding a new agent tool

Each tool is one file under `agent_tools/`. Convention:

```python
# agent_tools/measure_breath.py

from typing import Any
from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import session_key, read_json

MEASURE_BREATH_SCHEMA = {
    "type": "object",
    "properties": {
        "start_sec": {"type": "number"},
        "end_sec": {"type": "number"},
    },
    "required": ["start_sec", "end_sec"],
}
DESCRIPTION = "Detect breath events (inhalations) in a window. args: {start_sec, end_sec}"

def measure_breath(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    start = float(args["start_sec"])
    end = float(args["end_sec"])
    # ... your implementation, e.g. using existing audio analysis
    return {
        "window_sec": [start, end],
        "breaths_detected": [...],
        "notes": "..."
    }
```

Then register in `agent_tools/registry.py`:

```python
from masterclass.agent_tools.measure_breath import DESCRIPTION as MEASURE_BREATH_DESCRIPTION, MEASURE_BREATH_SCHEMA, measure_breath

ALL_TOOL_SPECS = (
    ...,
    ToolSpec("measure_breath", MEASURE_BREATH_DESCRIPTION, MEASURE_BREATH_SCHEMA, measure_breath),
)

# Optional gating
WOODWIND_ONLY_TOOLS = {"measure_breath"}

def _enabled_for_profile(spec: ToolSpec, profile: InstrumentProfile | None) -> bool:
    if profile and spec.name in WOODWIND_ONLY_TOOLS and profile.family != "woodwind":
        return False
    # ... existing gating
```

The teacher will automatically receive the tool in its catalog (per-profile filtered) and can invoke it during the agentic loop.

### Tool design guidelines

- **Return JSON-serializable dicts.** No numpy types, no `datetime` objects — use floats and ISO strings.
- **Catch your own errors and return `{"error": "..."}`.** Don't raise; the teacher receives the error JSON and adapts.
- **Cache expensive results.** See `agent_tools/listen.py` for the pattern (writes clip to `artifacts/listen_clips/` and reuses on subsequent calls with same args).
- **Keep schemas minimal.** Required fields only. Add `description` strings to schema properties for the teacher's benefit.
- **Document what's MEASURABLE vs PERCEPTUAL.** Tools that call Gemini (`listen`, `watch`) are perceptual — say so in the description.

## Adding a new pipeline stage

Append to `_run_lesson_jobs` in `apps/api/main.py`. The pattern:

```python
def run_best_effort(stage: str, fn) -> None:
    try:
        mark_stage(stage, "running")
        fn()
        mark_stage(stage, "ready")
    except Exception as exc:
        mark_stage(stage, "failed", error=f"{type(exc).__name__}: {exc}")
        manifest.errors.append({"stage": stage, "warning": True, "error": str(exc), "at": datetime.now(UTC).isoformat()})

# After existing stages...
run_best_effort(
    "my_new_stage",
    lambda: my_new_engine_module.do_thing(storage=storage, store=store, manifest=manifest),
)
```

Then add the stage to `apps/api/static/ingest.html`'s `STAGE_DEFS` array so the wizard's processing view shows progress:

```js
STAGE_DEFS = [
    ...,
    { key: "my_new_stage", label: "Doing the thing" },
]
```

If your stage produces an artifact the teacher should read, also:
1. Add it to the evidence digest builder in `engine/prompt_evidence.py`
2. Or expose it as a tool the teacher can invoke (preferred — keeps the prompt small)

### Stage failure modes

- **Fatal stages** (extract_media, analyze, evidence_packet) — wrap in `try/except` and re-raise. Lesson is marked FAILED.
- **Best-effort stages** (everything else) — use `run_best_effort` so a failure doesn't block downstream work.
- **Conditional stages** (voicing for keyboards only) — check the profile and call `mark_stage(stage, "skipped")` if not applicable.

## Adding a new score catalog source

Currently we use Mutopia. To add KernScores, OpenScore, IMSLP, MuseScore, etc.:

1. Add a parser to `engine/midi_finder.py`:

```python
def kernscores_search(piece_name: str, *, timeout: int) -> list[dict[str, Any]]:
    # Hit the catalog's search endpoint
    # Parse the response (HTML, JSON, whatever)
    # Return [{"source": "kernscores", "title": "...", "composer": "...", "midi_url": "...", "musicxml_url": "..."}, ...]
```

2. Wire it into `find_and_download_midi`:

```python
results_mutopia = mutopia_search(query, timeout=...)
results_kern = kernscores_search(query, timeout=...)
candidates = (results_mutopia + results_kern)[: config.max_candidates_per_source * 2]
```

3. The Gemini picker (`gemini_pick_best`) needs no changes — it already accepts a heterogeneous candidate list.

4. Update the audit JSON shape in `MidiFindResult` to include the new source name.

If the catalog ships MusicXML (most don't, but KernScores does via verovio conversion), wire it into a **new** "catalog MusicXML lookup" stage that runs BEFORE Audiveris in `engine/score_prep.py:prepare_score`. Skip Audiveris entirely if the catalog returned a usable MusicXML.

## Adding a new LLM model

To swap the teacher model from Gemini Pro to (e.g.) Claude Sonnet:

1. Implement `LlmProvider` interface (in `agent/llm.py`) for the new vendor:
   - `generate_json(model, system_instruction, contents, response_schema)` 
   - `generate_with_tools(model, system_instruction, contents, tool_declarations, max_tool_calls)`
   - File-upload support if the model handles multimodal

2. Add a provider class to `agent/`:

```python
class AnthropicProvider(LlmProvider):
    def __init__(self, config: AnthropicConfig): ...
    def generate_json(self, ...): ...
    def generate_with_tools(self, ...): ...
```

3. Wire into `apps/api/main.py:_build_llm_provider`:

```python
def _build_llm_provider() -> LlmProvider | None:
    backend = os.environ.get("MASTERCLASS_LLM_PROVIDER", "gemini")
    if backend == "anthropic":
        return AnthropicProvider(...)
    if backend == "gemini":
        return SharedKeyGeminiProvider(...)
    if backend == "dry-run":
        return DryRunProvider()
    return None
```

4. Test with a smoke script before swapping production traffic.

## Repository conventions

- **No new pip dependencies without strong justification.** The bundled Python is a curated set; adding a new dep means re-bundling.
- **Pure Python preferred.** No Cython, no C extensions besides what's already in librosa/numpy.
- **One module = one responsibility.** `engine/X.py` should have a clear noun-phrase purpose.
- **Smoke scripts not committed.** The author's working tree contains many one-off `scripts/test_*.py` smoke runners with hardcoded local paths; they are gitignored. If you write similar one-offs, drop them in `scripts/` and they'll be ignored automatically. There is no `pytest` suite yet — see `doc/limitations.md`.
- **No frameworks for the UI.** Plain HTML/CSS/JS in 2 files. New routes go through the existing hash-router in `ingest.html`.
- **Storage keys are tenant-scoped.** Never write to absolute paths; always go through `store.artifact_key` or `session_key` helpers.
- **Manifest metadata is free-form.** Use it for state tracking, diagnostic info, anything you'd want to query later. Don't overload schema fields.
