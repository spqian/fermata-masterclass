from __future__ import annotations

from typing import Any

from masterclass.core.models import SessionRef
from masterclass.storage.base import ObjectStorage
from ._common import list_session_keys, read_json, session_prefix

LIST_FRAMES_SCHEMA = {"type": "object", "properties": {}}
DESCRIPTION = "List all currently-extracted video frames."


def list_frames(storage: ObjectStorage, session: SessionRef, args: dict[str, Any]) -> dict[str, Any]:
    del args
    prefix = f"{session_prefix(session)}/artifacts/frames"
    keys = [k for k in list_session_keys(storage, session, "artifacts/frames") if k.lower().endswith((".jpg", ".jpeg", ".png"))]
    manifest = read_json(storage, session, "session.json", {}) or {}
    interval = float((manifest.get("metadata") or {}).get("frame_interval_sec") or 10.0)
    frames = []
    for i, key in enumerate(keys):
        rel = key[len(prefix.rstrip('/') + '/'):] if key.startswith(prefix.rstrip('/') + '/') else key
        frames.append({"path": f"artifacts/frames/{rel}", "name": key.rsplit('/', 1)[-1], "estimated_time_sec": round((i + 1) * interval, 2)})
    return {"frames": frames, "frame_interval_sec": interval, "count": len(frames)} if frames else {"frames": [], "count": 0, "note": "no frames directory yet"}
