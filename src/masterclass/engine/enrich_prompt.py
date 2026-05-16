from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from masterclass.agent_tools.catalog import tool_catalog_text as build_tool_catalog_text
from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.engine.instruments import load_instrument_profile, system_instruction_for_profile
from masterclass.engine.prompt_candidates import attach_candidate_notes
from masterclass.engine.prompt_evidence import build_evidence_digest
from masterclass.storage.base import ObjectStorage


VOICE_GUIDANCE_TEMPLATE = """You are a world-class {instrument} masterclass instructor — think {teacher_examples} — reviewing a student's recording.

- Speak as the teacher in the room: direct, encouraging, specific. Use second person ("you", "your hand/bow").
- Every comment must be **actionable** OR **musically illuminating**. If a comment does not pass that bar, drop it.
- Prefer one excellent observation over three mediocre ones. **Aim for 8-15 total comments**, not 30+.
- Imitate the masterclass tradition: name what you hear, name why it matters musically, prescribe a concrete experiment.
- Real masterclass critique is heavy on **dynamics within phrases** — which notes lead, which support, where the swells go, and how each long phrase is shaped. {voicing_focus}

Instrument-specific priorities:

{category_guidance}

Available measurements: {measurements_available}

Measurement discipline: {measurable_claims_rule}"""


def build_enrichment_prompt(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    score_map: dict,
    mechanical_comments: list[dict],
    evidence_digest_md: str | None = None,
    tool_catalog_text: str | None = None,
    instrument_profile: str | None = None,
) -> str:
    """Build a self-contained Markdown prompt for non-agentic comment enrichment.

    The prompt mirrors the PoC's `comments_enrichment_prompt.md` format while
    using v2 storage-scoped artifacts and v2 tool endpoint names.
    """

    profile = load_instrument_profile(instrument_profile or manifest.instrument_profile)
    voice_block = system_instruction_for_profile(profile, VOICE_GUIDANCE_TEMPLATE)
    score_key = score_map.get("key") if isinstance(score_map, dict) else None
    evidence = evidence_digest_md
    if evidence is None:
        evidence = build_evidence_digest(storage=storage, store=store, manifest=manifest, score_key=score_key)
    tools = tool_catalog_text if tool_catalog_text is not None else build_tool_catalog_text(profile)
    comments = _normalize_mechanical_comments(mechanical_comments)
    enriched_source = {
        "schema_version": 1,
        "session": manifest.session.session_id,
        "video_path": manifest.source_filename,
        "movement": manifest.movement,
        "repertoire": manifest.repertoire,
        "played_measures": _played_measures(manifest),
        "comments": attach_candidate_notes(comments, score_map),
    }

    out: list[str] = []
    out.append("# Comment Enrichment Prompt")
    out.append("")
    out.append(
        f"You are a world-class {profile.instrument} masterclass instructor "
        f"(think: {profile.teacher_examples}) reviewing a student's recording. You have:"
    )
    out.append("")
    out.append("- A list of mechanical comments anchored to specific moments (deterministic measurements from the audio).")
    out.append("- The score map (reference notes per bar with note_ids).")
    out.append("- The evidence digest summarizing the recording.")
    out.append("- Optionally, a sample of extracted frames you may comment on (technique/posture/setup) when something is genuinely visible.")
    out.append("")
    out.append("Your job: produce a SHORT, HIGH-VALUE masterclass critique. The mechanical pass is your **floor**, not your ceiling. Add musical/technical/teaching insight that the harness cannot measure.")
    out.append("")
    out.append("## Voice and quality bar")
    out.append("")
    out.append(voice_block)
    out.append("")
    out.append("## Investigation tools (agentic mode)")
    out.append("")
    out.append("You are NOT limited to the static evidence in this prompt. The harness exposes a small set of investigation tools that an API-capable client can call to dig deeper before writing the final critique. **Use them when available.** A masterclass-quality critique requires looking at specific moments closely, not just summarizing one batch of measurements.")
    out.append("")
    out.append("### Available tools")
    out.append("")
    out.extend(_format_tool_catalog(tools))
    out.append("")
    out.append("### Protocol")
    out.append("")
    out.append("Before writing `comments_enriched.json`, issue tool calls to investigate anything suspicious or interesting. In v2 each call is an HTTP request shaped like:")
    out.append("")
    out.append("```http")
    out.append(f"POST /sessions/{manifest.session.session_id}/tools/<tool_name>")
    out.append("Content-Type: application/json")
    out.append("")
    out.append('{"args": { ... }}')
    out.append("```")
    out.append("")
    out.append("For example, to verify the voicing of the opening chord at t=4.83s:")
    out.append("")
    out.append("```http")
    out.append(f"POST /sessions/{manifest.session.session_id}/tools/inspect_chord")
    out.append("Content-Type: application/json")
    out.append("")
    out.append('{"args": {"time_sec": 4.83}}')
    out.append("```")
    out.append("")
    out.append("Each call returns JSON. Use the result to inform the comment you're writing. **Aim for 5-15 tool calls** before finalizing — fewer than 3 means you're guessing, more than ~20 means you're stalling. If your environment cannot make HTTP/tool calls, produce the best manual fallback from the embedded evidence and mark uncertain claims as hypotheses.")
    out.append("")
    out.append("### Suggested investigations (not exhaustive)")
    out.append("")
    out.append("- For every chord on a downbeat: `inspect_chord` or `inspect_voicing` to see actual voicing (which voice is loudest in the recording vs. which the score implies should be).")
    out.append("- For every long note (>1s): `measure_vibrato` to characterize the vibrato when available for the instrument.")
    out.append("- For any bar flagged in the mechanical comments: `inspect_bar` to get the full picture.")
    out.append("- For a suspicious specific moment: `get_frames` with a 3-5s window to see what the player was doing physically.")
    out.append("- For a single odd note: `inspect_note` with `midi_measure` + `pitch`.")
    out.append("- For phrase shape or balance: `measure_dynamics` to support dynamics/voicing claims.")
    out.append("- For tempo/rubato claims: `measure_tempo` to support structural timing statements.")
    out.append("- **For any intonation claim (this note was sharp/flat by N cents): you MUST call `inspect_intonation` at the cited note's performed time with its expected pitch BEFORE writing the comment. The polyphonic_intonation summary is a pitch-class aggregate and can be skewed by a single buzzy sustained note — it is NOT evidence for a per-note claim.** Call `inspect_intonation` for each note you name in an intonation comment; if the returned `cents_off_score` is within +/- 15c, the note is in tune and you must not claim otherwise.")
    out.append("")
    out.append("### Important: do not invent measurements")
    out.append("")
    out.append('Tool results are facts. The static evidence below is also fact. Anything else is inference — mark it as such ("This sounds like X, hypothesizing because Y"). Never claim a numerical measurement that is not backed by a tool call or the evidence digest.')
    out.append("")
    out.append("## Required deliverables")
    out.append("")
    out.append("Beyond rewriting the mechanical comments you should ALSO ADD comments in these categories where warranted (do not invent measurements, but DO offer informed musical/technical hypotheses based on the score, evidence digest, and frames):")
    out.append("")
    out.append("- **musical**: phrasing, voice-leading, harmonic shape, what each section is about. AT LEAST 2-3 musical comments per take.")
    out.append("- **technique**: sound production, bow/touch/contact-point/distribution, left-hand pacing, articulation, ornament execution. Frame as hypothesis when you only have audio.")
    out.append("- **technique (visible)**: comment on posture/bow angle/instrument hold/hand position ONLY if extracted frames clearly show something. Mark these as 'visual observation; verify with side-angle camera.'")
    out.append("- **camera/setup**: if the frames suggest a recording problem (angle hides technique, low light, mic clipping), say so. Suggest improvements for next take.")
    out.append("- **practice prescription**: 1 final comment near the end of the take that gives 2-3 concrete things to work on this week.")
    out.append("")
    out.append("## Rules")
    out.append("")
    out.append("1. Keep `id`, `start_sec`, `end_sec`, `category`, `severity`, `measure`, `beat`, `evidence_ref` for each comment you preserve. For NEW comments you add, generate fresh ids like `e_new_001`, pick reasonable timestamps anchored to bars, and set `evidence_ref` to e.g. 'inferred_from_score', 'visual_inspection', 'practice_summary'.")
    out.append("2. Title <= 60 chars, musician-readable.")
    out.append("3. Message: 1-3 sentences. Distinguish observation vs. inference vs. hypothesis. Always end with a concrete try-this OR a musical question to audit.")
    out.append("4. Group adjacent mechanical rhythm comments in the same bar into a single observation about that bar's character.")
    out.append("5. For per-bar intonation/voicing, name the harmonic function or textural role inferred from the score.")
    out.append("6. For per-bar rubato, frame the question: structural seam (intentional) or mid-phrase (unconscious)?")
    out.append("7. **DROP** comments that are below the threshold of useful musician-facing feedback (e.g. a single 'beat 2.7 was 60ms late' with no musical consequence). Add their ids to a `dropped` array with one-line reasons.")
    out.append("8. Keep ONE summary card at t=0 as a one-paragraph welcome that orients the student to the piece and what you'll focus on.")
    out.append("9. **Severity discipline**: use `info` sparingly — only for purely descriptive observations. Most comments should be `warn` (worth working on) or `alert` (urgent). The web player hides `info` by default.")
    out.append("10. Do not invent measurements. You may infer musical/technical character without numerical claims.")
    out.append("")
    out.append("## Score-note citations (REQUIRED)")
    out.append("")
    out.append("Every comment MUST include a `note_refs` array. The web player highlights cited notes when the user hovers a comment.")
    out.append("")
    out.append("- For each source comment, the `candidate_notes` field lists score notes in the comment's time window with their `note_id`. Pick the subset your message actually discusses and copy their `note_id` values into `note_refs`.")
    out.append("- For NEW comments you add, look at the score notes in the relevant bar(s) and pick the note_ids that anchor your point.")
    out.append("- Specific notes (e.g. 'the D5 on beat 3 leans flat'): include those notes' `note_id`s.")
    out.append("- Whole-bar character (e.g. 'this bar stretches'): use `note_refs: []`. Player falls back to bar-level highlighting.")
    out.append("- Welcome card and practice-summary card at the start/end: `note_refs: []`.")
    out.append("- Prefer 1-4 note_refs per comment. Avoid more than ~6.")
    out.append('- Each entry can be a string (`note_id`) or an object `{"note_id": "..."}`. Strings are simpler.')
    out.append("")
    out.append("## Output schema")
    out.append("")
    out.append("Write the file to `comments_enriched.json` in the session directory (or persist it as `lesson/comments_enriched.json` in v2 storage):")
    out.append("")
    out.append("```json")
    out.append("{")
    out.append('  "session": "...",')
    out.append('  "video_path": "...",')
    out.append('  "movement": "...",')
    out.append('  "repertoire": "...",')
    out.append('  "played_measures": [first, last],')
    out.append('  "enrichment_notes": "one paragraph describing what changed from the mechanical pass",')
    out.append('  "comments": [')
    out.append('    {')
    out.append('      "id": "c001",')
    out.append('      "start_sec": 0.0,')
    out.append('      "end_sec": 5.0,')
    out.append('      "category": "musical|intonation|rhythm|technique|voicing|dynamics",')
    out.append('      "severity": "info|warn|alert",')
    out.append('      "title": "Welcome / overview",')
    out.append('      "message": "rewritten artist-voice text",')
    out.append('      "measure": 1,')
    out.append('      "beat": null,')
    out.append('      "evidence_ref": "summary",')
    out.append('      "source_comment_ids": ["c001"],')
    out.append('      "note_refs": ["m4_b1.00_A4", "m4_b2.50_C#5"]')
    out.append("    }")
    out.append("  ],")
    out.append('  "dropped": []')
    out.append("}")
    out.append("```")
    out.append("")
    out.append("## Source: mechanical comments (each with candidate score notes for citation)")
    out.append("")
    out.append("```json")
    out.append(json.dumps(enriched_source, indent=2, ensure_ascii=False))
    out.append("```")
    out.append("")
    out.append("## Source: evidence digest")
    out.append("")
    out.append(evidence or "")
    out.extend(_frames_section(storage, store, manifest))
    critique = _optional_text_artifact(storage, manifest, (
        "masterclass_critique.md",
        "analysis/masterclass_critique.md",
        "lesson/masterclass_critique.md",
    ))
    if critique:
        out.append("")
        out.append("## Source: existing artist-voice critique (masterclass_critique.md)")
        out.append("")
        out.append("Use the same voice and prescription style as this document.")
        out.append("")
        out.append(critique)
    return "\n".join(out)


def persist_enrichment_prompt(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    prompt_md: str,
) -> str:
    """Persist the fallback enrichment prompt and stamp the session manifest."""

    key = store.artifact_key(manifest.session, "analysis/enrichment_prompt.md")
    storage.write_bytes(key, prompt_md.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/enrichment_prompt.md"] = key
    manifest.metadata["enrichment_prompt_state"] = "ready"
    manifest.metadata["enrichment_prompt_generated_at"] = datetime.now(UTC).isoformat()
    manifest.metadata["enrichment_prompt_line_count"] = len(prompt_md.splitlines())
    store.save(manifest)
    return key


def _normalize_mechanical_comments(mechanical_comments: Any) -> list[dict]:
    if isinstance(mechanical_comments, dict):
        mechanical_comments = mechanical_comments.get("comments", [])
    if not isinstance(mechanical_comments, list):
        raise ValueError("mechanical_comments must be a list or an object with a comments list")
    return [dict(c) for c in mechanical_comments if isinstance(c, dict)]


def _played_measures(manifest: SessionManifest) -> list[Any] | None:
    first = manifest.metadata.get("first_measure")
    last = manifest.metadata.get("last_measure")
    if first is None or last is None:
        return None
    return [first, last]


def _format_tool_catalog(text: str) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.lower().rstrip(":") == "tool catalog":
            continue
        if line.startswith("- **"):
            lines.append(line)
        elif line.startswith("- "):
            body = line[2:]
            if ":" in body and " — " not in body:
                name, desc = body.split(":", 1)
                lines.append(f"- **{name.strip()}** — {desc.strip()}")
            else:
                lines.append(line)
        else:
            lines.append(f"- {line}")
    return lines or ["- (tool catalog unavailable)"]


def _frames_section(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest) -> list[str]:
    prefix = store.artifact_key(manifest.session, "artifacts/frames")
    try:
        frames = [
            key
            for key in storage.list_keys(prefix)
            if key.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    except Exception:
        frames = []
    frames.sort()
    if not frames:
        return []
    out = [
        "",
        "## Source: extracted video frames (for technique/camera observations)",
        "",
        "These stills are extracted at fixed intervals from the recording. Their filenames encode the timestamp. You may comment on visible posture/hand position/bow-arm/instrument-hold/setup ONLY if a frame clearly shows it. Always mark such comments as 'visual observation; verify with a side-angle camera in next take' — face-on shots can be misleading.",
        "",
    ]
    session_prefix = store.artifact_key(manifest.session, "")
    for key in frames[:30]:
        label = key.removeprefix(session_prefix)
        out.append(f"- `{label}`")
    if len(frames) > 30:
        out.append(f"- ... and {len(frames) - 30} more")
    return out


def _optional_text_artifact(storage: ObjectStorage, manifest: SessionManifest, relative_keys: tuple[str, ...]) -> str:
    for relative_key in relative_keys:
        key = manifest.artifacts.get(relative_key)
        if not key or not storage.exists(key):
            continue
        try:
            return storage.read_bytes(key).decode("utf-8")
        except UnicodeDecodeError:
            continue
    return ""
