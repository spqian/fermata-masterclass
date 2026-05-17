"""Defense-in-depth guard for score-anchored tools.

The drill tool registry already excludes ``inspect_bar``, ``inspect_note``,
``inspect_chord``, etc., but if a caller manually invokes one of these
tools on a drill session, we want a clear error rather than a confusing
"no score_map.json" message. This helper inspects the on-disk manifest
and returns ``True`` if the session is a drill.
"""
from __future__ import annotations

from masterclass.core.models import SESSION_KIND_DRILL, SessionManifest, SessionRef
from masterclass.storage.base import ObjectStorage


def session_is_drill(storage: ObjectStorage, session: SessionRef) -> bool:
    """Return True iff the on-disk manifest declares ``kind = "drill"``.

    Tolerant of missing manifests and malformed payloads: returns False
    when in doubt so existing lessons (which may not have a kind field
    yet on disk) keep working.
    """
    key = f"tenant/{session.tenant_id}/users/{session.user_id}/sessions/{session.session_id}/session.json"
    try:
        if not storage.exists(key):
            return False
        data = storage.read_json(key)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    try:
        return SessionManifest.from_json(data).kind == SESSION_KIND_DRILL
    except Exception:
        return False


_DRILL_REJECTION = {
    "error": (
        "this is a drill session — score-anchored tools (inspect_bar / "
        "inspect_note / inspect_chord / inspect_voicing) only work on "
        "lesson sessions that have been score-matched"
    ),
    "kind": "drill",
}


def reject_if_drill(storage: ObjectStorage, session: SessionRef) -> dict | None:
    """Return a rejection payload (to be returned from a tool) if drill, else None."""
    if session_is_drill(storage, session):
        return dict(_DRILL_REJECTION)
    return None
