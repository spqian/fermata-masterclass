from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from masterclass.core.models import SessionManifest, SessionRef
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


_ROLE_VALUES = {"user", "teacher", "system"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class ChatMessage:
    role: str
    content: str
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    # Free-form bag for heterogeneous events that share one conversation
    # file. Drill uploads tag entries with ``{"type": "drill_upload",
    # "drill_session_id": ..., "state": "processing"}``; drill results tag
    # with ``{"type": "drill_result", "drill_session_id": ...}``. Plain
    # chat messages leave it empty.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.tool_calls:
            data["tool_calls"] = list(self.tool_calls)
        if self.usage is not None:
            data["usage"] = dict(self.usage)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ChatMessage":
        role = str(data.get("role") or "").strip().lower()
        if role not in _ROLE_VALUES:
            role = "system"
        meta = data.get("metadata")
        return ChatMessage(
            role=role,
            content=str(data.get("content") or ""),
            ts=str(data.get("ts") or datetime.now(UTC).isoformat()),
            tool_calls=list(data.get("tool_calls") or []),
            usage=dict(data["usage"]) if isinstance(data.get("usage"), dict) else None,
            metadata=dict(meta) if isinstance(meta, dict) else {},
        )


@dataclass
class ChatConversation:
    conversation_id: str
    session_id: str
    user_id: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    messages: list[ChatMessage] = field(default_factory=list)
    schema_version: int = 1

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(UTC).isoformat()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_json() for m in self.messages],
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ChatConversation":
        return ChatConversation(
            schema_version=int(data.get("schema_version") or 1),
            conversation_id=str(data.get("conversation_id") or ""),
            session_id=str(data.get("session_id") or ""),
            user_id=str(data.get("user_id") or ""),
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or datetime.now(UTC).isoformat()),
            messages=[ChatMessage.from_json(m) for m in (data.get("messages") or []) if isinstance(m, dict)],
        )


def new_conversation(manifest: SessionManifest) -> ChatConversation:
    return ChatConversation(
        conversation_id=uuid4().hex,
        session_id=manifest.session.session_id,
        user_id=manifest.session.user_id,
    )


def safe_conversation_id(conversation_id: str) -> str:
    conversation_id = (conversation_id or "").strip()
    if not conversation_id or not _SAFE_ID.match(conversation_id) or ".." in conversation_id:
        raise ValueError("conversation_id must be path-safe")
    return conversation_id


def conversation_key(store: SessionStore, ref: SessionRef, conversation_id: str) -> str:
    return store.artifact_key(ref, f"chat/{safe_conversation_id(conversation_id)}.json")


def load_conversation(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, conversation_id: str) -> ChatConversation:
    key = conversation_key(store, manifest.session, conversation_id)
    data = storage.read_json(key)
    if data.get("deleted"):
        raise FileNotFoundError(conversation_id)
    return ChatConversation.from_json(data)


def save_conversation(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, conversation: ChatConversation) -> None:
    key = conversation_key(store, manifest.session, conversation.conversation_id)
    storage.write_json(key, conversation.to_json())


def get_or_create_conversation(
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    conversation_id: str | None,
) -> ChatConversation:
    """Load an existing conversation or create a new one.

    Behaviour:
    * ``conversation_id`` is None → always create a fresh conversation with
      a random uuid id.
    * ``conversation_id`` provided and the conversation file exists → load
      and return it.
    * ``conversation_id`` provided but no file exists → create a new
      conversation USING that explicit id, so callers (e.g. per-comment
      reply threads with deterministic ``cmt_<id>`` keys) can route the
      first reply into the right file. Previously this branch raised
      FileNotFoundError, which broke first-replies on comment threads.
    """
    if not conversation_id:
        return new_conversation(manifest)
    try:
        return load_conversation(storage, store, manifest, conversation_id)
    except FileNotFoundError:
        # Validate the id is path-safe; we're about to use it as a file
        # name. ``safe_conversation_id`` raises ValueError on bad input.
        safe_conversation_id(conversation_id)
        return ChatConversation(
            conversation_id=conversation_id,
            session_id=manifest.session.session_id,
            user_id=manifest.session.user_id,
        )


def list_conversations(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest) -> list[dict[str, Any]]:
    prefix = store.artifact_key(manifest.session, "chat")
    rows: list[dict[str, Any]] = []
    for key in storage.list_keys(prefix):
        if not key.endswith(".json"):
            continue
        try:
            data = storage.read_json(key)
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("deleted"):
            continue
        conv = ChatConversation.from_json(data)
        rows.append({
            "id": conv.conversation_id,
            "conversation_id": conv.conversation_id,
            "started_at": conv.created_at,
            "last_message_at": conv.updated_at,
            "message_count": conv.message_count,
        })
    rows.sort(key=lambda r: str(r.get("last_message_at") or ""), reverse=True)
    return rows


def delete_conversation(storage: ObjectStorage, store: SessionStore, manifest: SessionManifest, conversation_id: str) -> None:
    key = conversation_key(store, manifest.session, conversation_id)
    if not storage.exists(key):
        raise FileNotFoundError(conversation_id)
    resolver = getattr(storage, "resolve_local_path", None)
    if callable(resolver):
        path = resolver(key)
        path.unlink(missing_ok=True)
    else:
        storage.write_json(key, {"schema_version": 1, "deleted": True, "conversation_id": conversation_id})
