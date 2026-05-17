"""Tests for the drill-feature schema additions: SessionManifest.kind,
ChatMessage.metadata, and conversation_lock.
"""
from __future__ import annotations

import threading
import time

import pytest

from masterclass.core.chat_models import ChatConversation, ChatMessage
from masterclass.core.conversation_lock import conversation_lock
from masterclass.core.models import (
    SESSION_KIND_DRILL,
    SESSION_KIND_LESSON,
    SessionManifest,
    SessionRef,
)


def test_session_manifest_default_kind_is_lesson():
    ref = SessionRef("t", "u", "abc")
    m = SessionManifest(schema_version=1, session=ref)
    assert m.kind == SESSION_KIND_LESSON
    assert m.to_json()["kind"] == SESSION_KIND_LESSON


def test_session_manifest_round_trips_drill_kind():
    ref = SessionRef("t", "u", "abc")
    m = SessionManifest(schema_version=1, session=ref, kind=SESSION_KIND_DRILL)
    payload = m.to_json()
    assert payload["kind"] == SESSION_KIND_DRILL
    restored = SessionManifest.from_json(payload)
    assert restored.kind == SESSION_KIND_DRILL


def test_session_manifest_back_compat_missing_kind_field():
    ref = SessionRef("t", "u", "abc")
    payload = SessionManifest(schema_version=1, session=ref).to_json()
    payload.pop("kind")
    restored = SessionManifest.from_json(payload)
    assert restored.kind == SESSION_KIND_LESSON


def test_session_manifest_rejects_unknown_kind_via_fallback():
    ref = SessionRef("t", "u", "abc")
    payload = SessionManifest(schema_version=1, session=ref).to_json()
    payload["kind"] = "garbage"
    restored = SessionManifest.from_json(payload)
    assert restored.kind == SESSION_KIND_LESSON


def test_chat_message_metadata_default_is_empty_dict_and_omitted_in_json():
    msg = ChatMessage(role="user", content="hi")
    assert msg.metadata == {}
    assert "metadata" not in msg.to_json()


def test_chat_message_metadata_round_trips():
    meta = {"type": "drill_upload", "drill_session_id": "abc", "state": "processing"}
    msg = ChatMessage(role="system", content="uploaded", metadata=meta)
    payload = msg.to_json()
    assert payload["metadata"] == meta
    restored = ChatMessage.from_json(payload)
    assert restored.metadata == meta


def test_chat_message_from_json_missing_metadata_defaults_empty():
    payload = {"role": "user", "content": "hi", "ts": "2025-01-01T00:00:00+00:00"}
    msg = ChatMessage.from_json(payload)
    assert msg.metadata == {}


def test_chat_conversation_round_trips_messages_with_metadata():
    conv = ChatConversation(conversation_id="cmt_g_001", session_id="s", user_id="u")
    conv.append(ChatMessage(role="user", content="hello"))
    conv.append(ChatMessage(
        role="system",
        content="Practice clip uploaded",
        metadata={"type": "drill_upload", "drill_session_id": "drill1"},
    ))
    restored = ChatConversation.from_json(conv.to_json())
    assert len(restored.messages) == 2
    assert restored.messages[1].metadata["drill_session_id"] == "drill1"


def test_conversation_lock_serialises_concurrent_writers():
    key = "test/conv/cmt_g_007.json"
    events: list[str] = []

    def worker(name: str, hold_ms: int) -> None:
        with conversation_lock(key):
            events.append(f"{name}-enter")
            time.sleep(hold_ms / 1000.0)
            events.append(f"{name}-exit")

    threads = [
        threading.Thread(target=worker, args=("a", 60)),
        threading.Thread(target=worker, args=("b", 10)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Whichever ran first, the second's enter must come *after* the first's
    # exit, never interleaved.
    assert events.count("a-enter") == 1
    assert events.count("b-enter") == 1
    first_enter = events[0]
    first_letter = first_enter.split("-")[0]
    other_letter = "b" if first_letter == "a" else "a"
    assert events == [
        f"{first_letter}-enter",
        f"{first_letter}-exit",
        f"{other_letter}-enter",
        f"{other_letter}-exit",
    ]


def test_conversation_lock_releases_on_exception():
    key = "test/conv/release.json"
    with pytest.raises(RuntimeError):
        with conversation_lock(key):
            raise RuntimeError("boom")
    # The lock should now be acquirable again.
    with conversation_lock(key):
        pass


def test_conversation_lock_rejects_empty_key():
    with pytest.raises(ValueError):
        with conversation_lock(""):
            pass
