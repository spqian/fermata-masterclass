"""End-to-end smoke tests for the drill REST endpoints.

These tests stand up the real FastAPI app via TestClient, override the
auth dependency, and stub the background drill worker so the upload
endpoints can be exercised without invoking ffmpeg or basic-pitch.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTERCLASS_STORAGE_BACKEND", "local")
    monkeypatch.setenv("MASTERCLASS_LOCAL_ADLS_ROOT", str(tmp_path / "adls"))
    monkeypatch.setenv("MASTERCLASS_LLM_PROVIDER", "dry-run")
    monkeypatch.setenv("MASTERCLASS_KEY_ENCRYPTION_KEY", "x" * 64)
    monkeypatch.setenv("ALLOW_SERVER_DEFAULT_KEY", "true")

    # Replace the drill pipeline with a noop so the background thread
    # is harmless. The thread itself still runs but does nothing.
    from masterclass.engine import drill_pipeline as dp
    spawned: list[str] = []

    def noop_pipeline(*, storage, store, manifest, provider, config=None, **kwargs):
        spawned.append(manifest.session.session_id)
        # Do NOT save the manifest from this background thread — the
        # request handler may still be writing to the same file. Tests
        # that need to assert on drill state should call store.save
        # themselves under their own control.
        return manifest

    monkeypatch.setattr(dp, "run_drill_pipeline", noop_pipeline)

    from fastapi.testclient import TestClient
    from masterclass.apps.api import main as api_main

    app = api_main.create_app()

    from masterclass.core.models import TenantContext

    def _stub_tenant():
        return TenantContext(tenant_id="u-test", user_id="u-test")

    for route in app.routes:
        deps = getattr(getattr(route, "dependant", None), "dependencies", None) or []
        for dep in deps:
            call = getattr(dep, "call", None)
            if call is not None and getattr(call, "__name__", "") == "tenant_from_header":
                app.dependency_overrides[call] = _stub_tenant

    client = TestClient(app)
    client.spawned = spawned  # type: ignore[attr-defined]
    yield client


def _create_masterclass(client) -> str:
    # Use a MIDI file instead of a PDF so the score_prep background
    # thread doesn't kick in and race our test writes against the
    # masterclass.json file.
    files = {
        "midi_file": ("ref.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00x", "audio/midi"),
    }
    data = {"piece_name": "Test Drill Piece"}
    resp = client.post("/masterclasses", data=data, files=files)
    assert resp.status_code == 200, resp.text
    return resp.json()["masterclass"]["masterclass_id"]


def _create_lesson_session_manifest(masterclass_id: str):
    """Hand-build a READY lesson with a comment, sidestepping the full upload."""
    from masterclass.apps.api import main as api_main
    storage = api_main._build_storage()
    from masterclass.core.sessions import SessionStore
    from masterclass.core.masterclasses import MasterclassStore
    from masterclass.core.models import JobState, TenantContext

    ctx = TenantContext("u-test", "u-test")
    store = SessionStore(storage)
    mcs = MasterclassStore(storage)
    mc = mcs.load_by_id(ctx, masterclass_id)
    lesson = store.create(ctx, source_filename="lesson.mp4", repertoire="Test", instrument="violin")
    lesson.state = JobState.READY
    lesson.metadata["masterclass_id"] = masterclass_id
    enriched = {
        "comments": [
            {"id": "g_007", "text": "Play the trill from m.7 slowly with a metronome.",
             "summary": "Trill drill", "measure": 7, "severity": "warn"},
        ],
    }
    key = store.artifact_key(lesson.session, "lesson/comments_enriched.json")
    storage.write_json(key, enriched)
    lesson.artifacts["lesson/comments_enriched.json"] = key
    store.save(lesson)
    mc.lessons.append(lesson.session.session_id)
    mcs.save(mc)
    return lesson.session.session_id


def test_practice_clip_for_comment_creates_drill_and_appends_bubble(api_client):
    mc_id = _create_masterclass(api_client)
    lesson_id = _create_lesson_session_manifest(mc_id)

    resp = api_client.post(
        f"/lessons/{lesson_id}/comments/g_007/practice-clip",
        files={"file": ("clip.mp4", b"fakempegbytes", "video/mp4")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    drill_id = body["drill_session_id"]
    assert body["conversation_id"] == "cmt_g_007"
    assert body["parent_session_id"] == lesson_id

    resp2 = api_client.get(f"/drills/{drill_id}")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["drill_instruction"].startswith("Play the trill")
    assert body2["session"]["kind"] == "drill"

    resp3 = api_client.get(f"/drills/{drill_id}/status")
    assert resp3.status_code == 200
    assert "state" in resp3.json()

    history = api_client.get(f"/lessons/{lesson_id}/chat/comment/g_007").json()
    msgs = history.get("messages") or []
    assert any(m.get("metadata", {}).get("type") == "drill_upload" for m in msgs)
    assert api_client.spawned


def test_practice_clip_rejects_unknown_comment_id(api_client):
    mc_id = _create_masterclass(api_client)
    lesson_id = _create_lesson_session_manifest(mc_id)
    resp = api_client.post(
        f"/lessons/{lesson_id}/comments/no_such_comment/practice-clip",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
    )
    assert resp.status_code == 404


def test_masterclass_practice_clip_requires_instruction_or_parent(api_client):
    mc_id = _create_masterclass(api_client)
    resp = api_client.post(
        f"/masterclasses/{mc_id}/practice-clips",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={},
    )
    assert resp.status_code == 400


def test_masterclass_practice_clip_with_explicit_instruction(api_client):
    mc_id = _create_masterclass(api_client)
    resp = api_client.post(
        f"/masterclasses/{mc_id}/practice-clips",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"drill_instruction": "Slow scales in thirds, two octaves, 80bpm."},
    )
    assert resp.status_code == 200, resp.text
    drill_id = resp.json()["drill_session_id"]

    listed = api_client.get(f"/masterclasses/{mc_id}/practice-clips").json()
    assert any(c["drill_session_id"] == drill_id for c in listed)


def test_lesson_endpoints_reject_drill_session(api_client):
    mc_id = _create_masterclass(api_client)
    resp = api_client.post(
        f"/masterclasses/{mc_id}/practice-clips",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"drill_instruction": "scale practice"},
    )
    drill_id = resp.json()["drill_session_id"]
    resp2 = api_client.get(f"/lessons/{drill_id}/manifest")
    assert resp2.status_code == 409, resp2.text


def test_drill_retry_clears_failed_stages(api_client):
    mc_id = _create_masterclass(api_client)
    resp = api_client.post(
        f"/masterclasses/{mc_id}/practice-clips",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"drill_instruction": "x"},
    )
    drill_id = resp.json()["drill_session_id"]

    from masterclass.apps.api import main as api_main
    storage = api_main._build_storage()
    from masterclass.core.sessions import SessionStore
    from masterclass.core.models import TenantContext
    store = SessionStore(storage)
    ctx = TenantContext("u-test", "u-test")
    drill = store.load_by_id(ctx, drill_id)
    drill.metadata["drill_state"] = "failed"
    drill.metadata["drill_feedback_state"] = "failed"
    drill.metadata["drill_feedback_error"] = "fake"
    store.save(drill)

    resp2 = api_client.post(f"/drills/{drill_id}/retry")
    assert resp2.status_code == 200
    body = resp2.json()
    assert "drill_feedback" in body["retried_stages"]
    assert body["state"] == "processing"


def test_drill_delete_tombstones_and_removes_from_masterclass(api_client):
    mc_id = _create_masterclass(api_client)
    resp = api_client.post(
        f"/masterclasses/{mc_id}/practice-clips",
        files={"file": ("clip.mp4", b"x", "video/mp4")},
        data={"drill_instruction": "x"},
    )
    drill_id = resp.json()["drill_session_id"]

    resp2 = api_client.delete(f"/drills/{drill_id}")
    assert resp2.status_code == 200
    listed = api_client.get(f"/masterclasses/{mc_id}/practice-clips").json()
    assert not any(c["drill_session_id"] == drill_id for c in listed)
