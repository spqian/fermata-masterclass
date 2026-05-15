from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class JobState(StrEnum):
    CREATED = "created"
    UPLOADED = "uploaded"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    ANALYZING = "analyzing"
    ALIGNING = "aligning"
    GENERATING_EVIDENCE = "generating_evidence"
    AWAITING_LLM = "awaiting_llm"
    TEACHING = "teaching"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TenantContext:
    """MVP tenant model: one tenant per authenticated individual user."""

    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id or "/" in self.tenant_id or "\\" in self.tenant_id:
            raise ValueError("tenant_id must be a non-empty path-safe id")
        if not self.user_id or "/" in self.user_id or "\\" in self.user_id:
            raise ValueError("user_id must be a non-empty path-safe id")


@dataclass(frozen=True)
class SessionRef:
    tenant_id: str
    user_id: str
    session_id: str

    @staticmethod
    def new(ctx: TenantContext) -> "SessionRef":
        return SessionRef(ctx.tenant_id, ctx.user_id, uuid4().hex)


@dataclass(frozen=True)
class MasterclassRef:
    tenant_id: str
    user_id: str
    masterclass_id: str

    @staticmethod
    def new(ctx: TenantContext) -> "MasterclassRef":
        return MasterclassRef(ctx.tenant_id, ctx.user_id, uuid4().hex)


@dataclass
class MasterclassManifest:
    schema_version: int
    masterclass: MasterclassRef
    piece_name: str
    movement: str | None = None
    instrument: str | None = None
    instrument_profile: str | None = None
    work_id: str | None = None
    score_url: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC).isoformat()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "masterclass": {
                "tenant_id": self.masterclass.tenant_id,
                "user_id": self.masterclass.user_id,
                "masterclass_id": self.masterclass.masterclass_id,
            },
            "piece_name": self.piece_name,
            "movement": self.movement,
            "instrument": self.instrument,
            "instrument_profile": self.instrument_profile,
            "work_id": self.work_id,
            "score_url": self.score_url,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
            "lessons": list(self.lessons),
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "MasterclassManifest":
        masterclass = data["masterclass"]
        return MasterclassManifest(
            schema_version=int(data["schema_version"]),
            masterclass=MasterclassRef(
                masterclass["tenant_id"],
                masterclass["user_id"],
                masterclass["masterclass_id"],
            ),
            piece_name=data["piece_name"],
            movement=data.get("movement"),
            instrument=data.get("instrument"),
            instrument_profile=data.get("instrument_profile"),
            work_id=data.get("work_id"),
            score_url=data.get("score_url"),
            notes=data.get("notes"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            artifacts=dict(data.get("artifacts", {})),
            metadata=dict(data.get("metadata", {})),
            lessons=list(data.get("lessons", [])),
        )


@dataclass
class SessionManifest:
    schema_version: int
    session: SessionRef
    state: JobState = JobState.CREATED
    source_filename: str | None = None
    repertoire: str | None = None
    movement: str | None = None
    instrument: str | None = None
    instrument_profile: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_usage: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC).isoformat()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session": {
                "tenant_id": self.session.tenant_id,
                "user_id": self.session.user_id,
                "session_id": self.session.session_id,
            },
            "state": self.state.value,
            "source_filename": self.source_filename,
            "repertoire": self.repertoire,
            "movement": self.movement,
            "instrument": self.instrument,
            "instrument_profile": self.instrument_profile,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
            "llm_usage": list(self.llm_usage),
            "errors": list(self.errors),
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "SessionManifest":
        session = data["session"]
        return SessionManifest(
            schema_version=int(data["schema_version"]),
            session=SessionRef(session["tenant_id"], session["user_id"], session["session_id"]),
            state=JobState(data.get("state", JobState.CREATED)),
            source_filename=data.get("source_filename"),
            repertoire=data.get("repertoire"),
            movement=data.get("movement"),
            instrument=data.get("instrument"),
            instrument_profile=data.get("instrument_profile"),
            notes=data.get("notes"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            artifacts=dict(data.get("artifacts", {})),
            metadata=dict(data.get("metadata", {})),
            llm_usage=list(data.get("llm_usage", [])),
            errors=list(data.get("errors", [])),
        )


def session_prefix(ref: SessionRef) -> str:
    """ADLS-compatible logical prefix for all session-owned data."""

    return f"tenant/{ref.tenant_id}/users/{ref.user_id}/sessions/{ref.session_id}"


def masterclass_prefix(ref: MasterclassRef) -> str:
    """ADLS-compatible logical prefix for all masterclass-owned reference data."""

    return f"tenant/{ref.tenant_id}/users/{ref.user_id}/masterclasses/{ref.masterclass_id}"
