from __future__ import annotations

import os
from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import run_ffmpeg_from_storage, session_key

LISTEN_SCHEMA = {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}, "question": {"type": "string"}, "model": {"type": "string"}}, "required": ["start_sec", "end_sec", "question"]}
DESCRIPTION = "PERCEPTUAL: hand an audio clip to Gemini and ask a question. args: {start_sec, end_sec, question, model?}"


def listen(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "hint": "Set GEMINI_API_KEY to enable perceptual listening."}
    start = float(args["start_sec"]); end = float(args["end_sec"])
    if end <= start:
        return {"error": "end_sec must be > start_sec"}
    max_clip = float(os.environ.get("LISTEN_MAX_CLIP_SEC", "60"))
    if end - start > max_clip:
        return {"error": f"clip too long ({end-start:.1f}s > LISTEN_MAX_CLIP_SEC={max_clip}s). Tighten the window."}
    question = str(args.get("question", "")).strip()
    if not question:
        return {"error": "question is required"}
    clip_name = f"clip_{int(start*1000):07d}_{int(end*1000):07d}.wav"
    clip_rel = f"artifacts/listen_clips/{clip_name}"
    clip_key = session_key(session, clip_rel)
    if storage.exists(clip_key):
        audio_bytes = storage.read_bytes(clip_key)
    else:
        audio_bytes, err = run_ffmpeg_from_storage(storage, session, "artifacts/audio.wav", clip_name, ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "--", "-ar", "22050", "-ac", "1"])
        if err:
            return {"error": err}
        storage.write_bytes(clip_key, audio_bytes or b"", content_type="audio/wav")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return {"error": f"google-genai not installed: {exc}. Run: tools\\python\\python.exe -m pip install google-genai"}
    model_name = args.get("model") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = "You are a music masterclass instructor listening carefully to a short audio clip. Answer with specific, perceptually grounded observations. Distinguish what you hear from what you infer. Be concise (2-4 sentences). Do not invent measurements.\n\nQuestion: " + question
    parts = [types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"), prompt]
    last_error: Exception | None = None
    # Retry up to 2 times on the SDK's "client has been closed" quirk.
    for attempt in range(2):
        try:
            client = genai.Client(api_key=api_key)
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
    return {"window_sec": [start, end], "duration_sec": round(end - start, 2), "model": model_name, "question": question, "answer": (response.text or "").strip(), "clip_path": clip_rel}

