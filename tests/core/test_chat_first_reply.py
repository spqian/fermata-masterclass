"""First-reply on a per-comment thread must create the conversation, not 404."""
from __future__ import annotations


def test_get_or_create_conversation_creates_with_explicit_id(
    local_storage, session_store, tenant_ctx
):
    from masterclass.core.chat_models import (
        ChatMessage,
        get_or_create_conversation,
        save_conversation,
    )
    from tests.conftest import make_session_manifest

    manifest = make_session_manifest(local_storage, session_store, tenant_ctx)
    # First-reply on a comment thread: deterministic id, file does not exist.
    conv = get_or_create_conversation(
        local_storage, session_store, manifest, "cmt_g_007"
    )
    assert conv.conversation_id == "cmt_g_007"
    assert conv.messages == []

    # Append a turn and save; second call should load the existing file.
    conv.append(ChatMessage(role="user", content="hi"))
    save_conversation(local_storage, session_store, manifest, conv)
    loaded = get_or_create_conversation(
        local_storage, session_store, manifest, "cmt_g_007"
    )
    assert loaded.conversation_id == "cmt_g_007"
    assert [m.content for m in loaded.messages] == ["hi"]


def test_get_or_create_conversation_no_id_creates_random(
    local_storage, session_store, tenant_ctx
):
    from masterclass.core.chat_models import get_or_create_conversation
    from tests.conftest import make_session_manifest

    manifest = make_session_manifest(local_storage, session_store, tenant_ctx)
    conv = get_or_create_conversation(local_storage, session_store, manifest, None)
    assert conv.conversation_id  # non-empty uuid
    assert not conv.conversation_id.startswith("cmt_")


def test_get_or_create_conversation_rejects_unsafe_id(
    local_storage, session_store, tenant_ctx
):
    import pytest
    from masterclass.core.chat_models import get_or_create_conversation
    from tests.conftest import make_session_manifest

    manifest = make_session_manifest(local_storage, session_store, tenant_ctx)
    with pytest.raises(ValueError):
        get_or_create_conversation(local_storage, session_store, manifest, "../escape")
