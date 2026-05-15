from __future__ import annotations

import os
from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import run_ffmpeg_from_storage, session_key

WATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "start_sec": {"type": "number", "description": "Clip start time in seconds"},
        "end_sec": {"type": "number", "description": "Clip end time in seconds (max ~10s of clip)"},
        "question": {"type": "string", "description": "Specific motion/technique question, e.g. 'how fast is the bow moving on the down-bow at this measure?'"},
        "max_height": {"type": "integer", "description": "Optional: downscale to this max height (default 480) to keep upload small"},
        "model": {"type": "string"},
    },
    "required": ["start_sec", "end_sec", "question"],
}
DESCRIPTION = (
    "PERCEPTUAL VIDEO: extract a short MP4 clip (≤10s) from the lesson video, hand it to Gemini, "
    "and ask a motion/technique question. Use this for things a single frame can't show: bow speed, "
    "vibrato motion, pedal-change timing, finger transitions, breath pacing, posture changes. "
    "args: {start_sec, end_sec, question, max_height?, model?}"
)


def watch(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "hint": "Set GEMINI_API_KEY to enable perceptual watching."}
    start = float(args["start_sec"]); end = float(args["end_sec"])
    if end <= start:
        return {"error": "end_sec must be > start_sec"}
    max_clip = float(os.environ.get("WATCH_MAX_CLIP_SEC", "10"))
    if end - start > max_clip:
        return {"error": f"clip too long ({end-start:.1f}s > WATCH_MAX_CLIP_SEC={max_clip}s). Tighten the window — short bursts read motion better."}
    question = str(args.get("question", "")).strip()
    if not question:
        return {"error": "question is required"}
    max_height = int(args.get("max_height") or 480)
    clip_name = f"clip_{int(start*1000):07d}_{int(end*1000):07d}_h{max_height}.mp4"
    clip_rel = f"artifacts/watch_clips/{clip_name}"
    clip_key = session_key(session, clip_rel)
    if storage.exists(clip_key):
        video_bytes = storage.read_bytes(clip_key)
    else:
        # Re-encode to a small h.264 mp4 with no audio for fast upload + Gemini compatibility.
        ff_args = [
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "--",
            "-an",
            "-vf", f"scale=-2:'min({max_height},ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        video_bytes, err = run_ffmpeg_from_storage(storage, session, "input/source_video", clip_name, ff_args)
        if err:
            return {"error": err}
        storage.write_bytes(clip_key, video_bytes or b"", content_type="video/mp4")
    if not video_bytes:
        return {"error": "ffmpeg produced empty clip"}
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return {"error": f"google-genai not installed: {exc}"}
    model_name = args.get("model") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        "You are a music masterclass instructor reviewing a short video clip of a student playing. "
        "Watch carefully for MOTION cues that a single still frame cannot show: bow speed/pressure, "
        "vibrato rate and width, pedal-change timing, finger transitions, hand/wrist motion, posture shifts, breath pacing. "
        "Answer with specific, perceptually grounded observations. Distinguish what you SEE from what you INFER. "
        "Be concise (2-4 sentences). If the camera angle hides what would answer the question, say so plainly.\n\n"
        f"Question: {question}"
    )
    inline_limit = int(os.environ.get("WATCH_INLINE_BYTES", str(15 * 1024 * 1024)))
    parts: list[Any]
    uploaded_file = None
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        return {"error": f"genai client init failed: {type(exc).__name__}: {exc}"}
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            client = genai.Client(api_key=api_key)
            if len(video_bytes) <= inline_limit:
                parts = [types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"), prompt]
            else:
                # Files API path for larger clips.
                import io as _io
                uploaded_file = client.files.upload(file=_io.BytesIO(video_bytes), config={"mime_type": "video/mp4"})
                parts = [uploaded_file, prompt]
            response = client.models.generate_content(model=model_name, contents=parts)
            break
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            if "client has been closed" in msg or "client is closed" in msg:
                continue
            return {"error": f"gemini call failed: {type(exc).__name__}: {exc}"}
    else:
        return {"error": f"gemini call failed after retries: {type(last_error).__name__}: {last_error}"}
    return {
        "window_sec": [start, end],
        "duration_sec": round(end - start, 2),
        "model": model_name,
        "question": question,
        "answer": (response.text or "").strip(),
        "clip_path": clip_rel,
        "clip_bytes": len(video_bytes),
        "transport": "inline" if uploaded_file is None else "files_api",
    }
