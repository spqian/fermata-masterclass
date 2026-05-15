from __future__ import annotations

import wave
from io import BytesIO
from typing import Any

from masterclass.agent.llm import LlmUsage, ToolExecutor


class DryRunLlmProvider:
    """No-cost provider for validating the teacher-agent/storage/tool seam.

    For ``generate_json`` we now produce realistic-looking output that matches
    the request shape, so the rest of the pipeline (UI, player, comment
    rendering) can be exercised end-to-end without spending on the LLM.
    """

    provider_name = "dry-run"

    def generate_with_tools(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        tools: list[dict[str, Any]],
        max_tool_calls: int,
        tool_executor: ToolExecutor | None = None,
    ) -> tuple[str, LlmUsage, list[dict[str, Any]]]:
        del max_tool_calls
        tool_calls: list[dict[str, Any]] = []
        if tool_executor and any(t.get("name") == "inspect_voicing" for t in tools):
            result = tool_executor("inspect_voicing", {"midi_measure": 1})
            tool_calls.append({
                "turn": 1,
                "tool": "inspect_voicing",
                "args": {"midi_measure": 1},
                "status": "error" if "error" in result else "ok",
                "events_matched": result.get("events_matched"),
            })
        is_teach = any(
            isinstance(c, str) and ("score note inventory" in c.lower() or "comments_enriched.json" in c.lower())
            for c in contents
        )
        if is_teach:
            page_count = sum(
                1 for c in contents
                if isinstance(c, dict) and (c.get("mime_type") or "").startswith("image/")
            )
            audio_part = next(
                (c for c in contents if isinstance(c, dict) and (c.get("mime_type") or "").startswith("audio/")),
                None,
            )
            duration_sec = _estimate_wav_duration(audio_part["data"]) if audio_part else 60.0
            payload = _mock_teach_response(duration_sec, page_count, contents, system_instruction)
            payload.setdefault("lesson", _mock_lesson())
            payload.setdefault("dropped", [])
            text = "```json\n" + __import__("json").dumps(payload, indent=2) + "\n```"
            usage = LlmUsage(provider=self.provider_name, model=model, input_tokens=1000, output_tokens=500, estimated_cost_usd=0.0)
            return text, usage, tool_calls
        text = (
            "DRY RUN teacher response.\n\n"
            f"System instruction chars: {len(system_instruction)}\n"
            f"Content parts: {len(contents)}\n"
            f"Tools declared: {[t.get('name') for t in tools]}\n"
            f"Tool calls: {tool_calls}\n"
        )
        usage = LlmUsage(provider=self.provider_name, model=model, input_tokens=0, output_tokens=0, estimated_cost_usd=0.0)
        return text, usage, tool_calls

    def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], LlmUsage]:
        del response_schema
        page_count = sum(
            1 for c in contents
            if isinstance(c, dict) and (c.get("mime_type") or "").startswith("image/")
        )
        audio_part = next(
            (c for c in contents
             if isinstance(c, dict) and (c.get("mime_type") or "").startswith("audio/")),
            None,
        )
        is_teach = any(
            isinstance(c, str) and ("evidence packet" in c.lower() or "lesson audio" in c.lower())
            for c in contents
        )
        if is_teach:
            duration_sec = _estimate_wav_duration(audio_part["data"]) if audio_part else 60.0
            result = _mock_teach_response(duration_sec, page_count, contents, system_instruction)
        else:
            first_music = min(3, max(1, page_count))
            result = {
                "_dry_run": True,
                "system_instruction_chars": len(system_instruction),
                "text_parts": sum(1 for c in contents if isinstance(c, str)),
                "image_parts": page_count,
                "first_music_page": first_music,
                "page_count": page_count,
                "movements": [],
                "pages": [
                    {
                        "page": p,
                        "kind": "music" if p >= first_music else "front",
                        "system_count": 5 if p >= first_music else 0,
                        "first_measure": (p - first_music) * 5 + 1 if p >= first_music else None,
                        "last_measure": (p - first_music + 1) * 5 if p >= first_music else None,
                        "systems": [
                            {
                                "system_index": s,
                                "first_measure": (p - first_music) * 5 + s,
                                "last_measure": (p - first_music) * 5 + s,
                                "bbox": {"x": 0.05, "y": 0.05 + (s - 1) * 0.18, "w": 0.9, "h": 0.16},
                            }
                            for s in range(1, 6)
                        ] if p >= first_music else [],
                    }
                    for p in range(1, page_count + 1)
                ],
                "notes": "Dry-run provider — no real score analysis performed.",
            }
        usage = LlmUsage(
            provider=self.provider_name,
            model=model,
            input_tokens=1000,
            output_tokens=500,
            estimated_cost_usd=0.0,
        )
        return result, usage

    def search_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: list[Any],
    ) -> tuple[dict[str, Any], LlmUsage]:
        del system_instruction, contents
        result = {
            "_dry_run": True,
            "midi_url": "",
            "midi_source": "(dry-run)",
            "midi_attribution": "Dry-run provider — no real search performed.",
            "candidates": [],
        }
        usage = LlmUsage(provider=self.provider_name, model=model, input_tokens=0, output_tokens=0, estimated_cost_usd=0.0)
        return result, usage


def _estimate_wav_duration(data: bytes) -> float:
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except Exception:
        return 60.0


def _mock_teach_response(
    duration_sec: float,
    page_count: int,
    contents: list[Any],
    system_instruction: str,
) -> dict[str, Any]:
    duration_sec = max(8.0, duration_sec)
    has_prior = any(
        isinstance(c, str) and "prior lessons" in c.lower() and "no prior" not in c.lower()
        for c in contents
    )
    fractions = (0.05, 0.18, 0.32, 0.46, 0.60, 0.74, 0.88)
    timestamps = [round(duration_sec * f, 2) for f in fractions]
    templates = [
        ("intonation", "Opening intonation — third finger sharp",
         "The first held note above the staff arrives a touch above pitch; you correct downward "
         "within about a half-second. Try to land it with the bow already in motion so the arrival "
         "is in tune from the first instant rather than corrected after the fact."),
        ("rhythm", "Tempo dragging into the first phrase ending",
         "There is a small but consistent slowing into the cadential figure here; the long note "
         "at the end of the phrase is also slightly held. If the slowing is intentional rubato it "
         "needs to be a clearer gesture; if not, watch that the bow speed does not collapse on the long note."),
        ("dynamics", "Crescendo arrives early",
         "The dynamic peak lands a beat or two before the harmonic arrival. Try anchoring the peak "
         "to the bass note and letting the upper voice come up to meet it, instead of leading it."),
        ("articulation", "String crossing onsets are uneven",
         "Across this passage the upper-string onsets speak slightly later than the lower ones, "
         "which thickens the texture. Practice this very slowly with separate bows, checking that "
         "the new string is already vibrating before the bow leaves the previous one."),
        ("phrasing", "Breath at the phrase boundary",
         "There is a clear musical comma here, but the bow does not lift or slow to mark it. A small "
         "bow lift, even just a release of weight, would let the next phrase begin from silence rather "
         "than from continuing sound."),
        ("interpretation", "Character of the dance is present but understated",
         "The basic pulse and gesture are right, but the upbeats can lean forward more — the dance "
         "character lives in those gestures more than in the downbeats."),
        ("technique", "Vibrato narrows under pressure",
         "On the longer notes in the most exposed register, vibrato narrows and slows. Long, slow "
         "scales with a metronome focusing on a wider, even oscillation will help; you have the "
         "speed, you need the width."),
    ]
    comments = []
    for i, (cat, summary, text) in enumerate(templates, start=1):
        start = timestamps[(i - 1) % len(timestamps)]
        measure = i * 2
        comments.append({
            "id": f"g_{i:03d}",
            "start": start,
            "end": round(start + 4.0, 2),
            "measure": measure,
            "category": cat,
            "summary": summary,
            "text": text,
            "confidence": "medium",
            "references": [
                {
                    "measure": measure,
                    "beat": 1,
                    "note_name": "Bb5" if i % 2 else "G4",
                    "page": min(2 + (i // 4), max(2, page_count)),
                    "system_index": ((measure - 1) % 5) + 1,
                    "hand": "right" if i % 2 else "left",
                }
            ],
        })
    measure_timestamps = []
    for measure in range(1, 17):
        measure_timestamps.append({
            "measure": measure,
            "start": round(duration_sec * (measure - 1) / 16.0, 2),
        })
    return {
        "summary": (
            "A clear performance with strong technical command. The biggest opportunities for the next "
            "lesson are intonation on entries above the staff, marking phrase boundaries with the bow, "
            "and letting the dance character live in the upbeats."
        ),
        "progress_notes": (
            "Compared with the prior lesson the tempo of the second phrase is steadier and the cadence "
            "no longer drags. Intonation on the highest entries is still work-in-progress."
        ) if has_prior else "",
        "measure_timestamps": measure_timestamps,
        "comments": comments,
        "_mock_meta": {
            "score_pages_attached": page_count,
            "audio_seconds": round(duration_sec, 2),
            "system_chars": len(system_instruction),
        },
    }


def _mock_lesson() -> dict[str, Any]:
    return {
        "artistic_summary": "This dry-run lesson treats the piece as a singing line supported by disciplined pulse and color.",
        "what_works": ["Clear commitment to the phrase", "Audible attention to cadence points"],
        "areas_to_develop": [
            {"focus": "Pulse foundation", "priority": "high", "exercise": "Practice left-hand/accompaniment alone with a metronome."},
            {"focus": "Melody projection", "priority": "medium", "exercise": "Play melody alone, then add accompaniment at half dynamic."},
        ],
        "this_week_practice": ["Slow metronome work", "Melody-only shaping", "Record one focused retake"],
        "next_take": "Use the same camera angle and capture one complete take at a comfortable tempo.",
    }
