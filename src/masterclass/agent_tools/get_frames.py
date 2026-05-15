from __future__ import annotations

import base64
from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import extract_video_frames, list_session_keys, read_json

GET_FRAMES_SCHEMA = {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}, "fps": {"type": "number"}}, "required": ["start_sec", "end_sec"]}
DESCRIPTION = "Extract and return base64 JPEG frames in a time window. args: {start_sec, end_sec, fps?}"


def get_frames(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    start = float(args["start_sec"]); end = float(args["end_sec"]); fps = float(args.get("fps", 4.0))
    if end <= start:
        return {"error": "end_sec must be > start_sec"}
    if end - start > 10.0:
        return {"error": "window too large (max 10s) — narrow your inquiry"}
    frames, err = extract_video_frames(storage, session, start, end, fps)
    if err:
        fallback = _existing_frames_in_window(storage, session, start, end)
        if fallback:
            return {"frames": fallback, "window_sec": [start, end], "fps": fps, "count": len(fallback), "note": f"source video unavailable for fresh extraction; returned existing frames ({err})"}
        return {"error": err}
    return {"frames": frames, "window_sec": [start, end], "fps": fps, "count": len(frames)}


def _existing_frames_in_window(storage: ObjectStorage, session: SessionRef, start: float, end: float) -> list[dict[str, Any]]:
    keys = [k for k in list_session_keys(storage, session, "artifacts/frames") if k.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not keys:
        return []
    manifest = read_json(storage, session, "session.json", {}) or {}
    interval = float((manifest.get("metadata") or {}).get("frame_interval_sec") or 10.0)
    out = []
    for i, key in enumerate(keys):
        t = (i + 1) * interval
        if start <= t <= end:
            data = storage.read_bytes(key)
            out.append({
                "path": key.split(f"/sessions/{session.session_id}/", 1)[-1],
                "name": key.rsplit("/", 1)[-1],
                "estimated_time_sec": round(t, 3),
                "mime_type": "image/jpeg",
                "base64_jpeg": base64.b64encode(data).decode("ascii"),
            })
    if not out:
        # Return the nearest available frame rather than failing when ingest frames are sparse.
        nearest = min(enumerate(keys), key=lambda item: abs(((item[0] + 1) * interval) - ((start + end) / 2)))
        i, key = nearest
        data = storage.read_bytes(key)
        out.append({
            "path": key.split(f"/sessions/{session.session_id}/", 1)[-1],
            "name": key.rsplit("/", 1)[-1],
            "estimated_time_sec": round((i + 1) * interval, 3),
            "mime_type": "image/jpeg",
            "base64_jpeg": base64.b64encode(data).decode("ascii"),
            "note": "nearest existing frame outside requested window",
        })
    return out
