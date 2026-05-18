"""Storage layer delete + read-json retry behavior."""
from __future__ import annotations

import json


def test_local_delete_key_returns_true_when_existed_false_when_missing(local_storage):
    local_storage.write_bytes("a/b/c.txt", b"hello")
    assert local_storage.delete_key("a/b/c.txt") is True
    assert local_storage.delete_key("a/b/c.txt") is False


def test_local_delete_prefix_recursive(local_storage):
    local_storage.write_bytes("foo/a.txt", b"1")
    local_storage.write_bytes("foo/sub/b.txt", b"2")
    local_storage.write_bytes("foo/sub/deep/c.txt", b"3")
    local_storage.write_bytes("bar/x.txt", b"unrelated")
    n = local_storage.delete_prefix("foo")
    assert n == 3
    # foo/ gone, bar/ untouched
    assert not local_storage.exists("foo/a.txt")
    assert not local_storage.exists("foo/sub/b.txt")
    assert local_storage.exists("bar/x.txt")


def test_local_delete_prefix_idempotent(local_storage):
    assert local_storage.delete_prefix("never-existed") == 0


def test_session_delete_by_id_removes_artifacts(local_storage, session_store, tenant_ctx):
    from tests.conftest import make_session_manifest
    manifest = make_session_manifest(
        local_storage, session_store, tenant_ctx,
        artifacts={
            "analysis/foo.json": {"x": 1},
            "lesson/bar.md": b"markdown",
        },
    )
    sid = manifest.session.session_id
    # Verify session exists
    session_store.load_by_id(tenant_ctx, sid)
    # Delete
    count = session_store.delete_by_id(tenant_ctx, sid)
    assert count >= 1  # manifest + 2 artifacts
    # Subsequent load should raise FileNotFoundError
    import pytest
    with pytest.raises(FileNotFoundError):
        session_store.load_by_id(tenant_ctx, sid)


def test_read_json_retries_on_empty_then_succeeds(local_storage, monkeypatch):
    # Make read_bytes return empty once, then real data.
    calls = {"n": 0}
    real_payload = json.dumps({"hello": "world"}).encode("utf-8")

    real_read_bytes = local_storage.read_bytes

    def flaky_read_bytes(key):
        calls["n"] += 1
        if calls["n"] == 1:
            return b""
        return real_payload

    monkeypatch.setattr(local_storage, "read_bytes", flaky_read_bytes)
    out = local_storage.read_json("any/key")
    assert out == {"hello": "world"}
    assert calls["n"] == 2  # one empty, one good


def test_read_json_raises_after_all_empty(local_storage, monkeypatch):
    import pytest
    monkeypatch.setattr(local_storage, "read_bytes", lambda key: b"")
    with pytest.raises(json.JSONDecodeError):
        local_storage.read_json("any/key")
