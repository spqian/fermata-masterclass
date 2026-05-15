from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from masterclass.core.models import SessionRef, session_prefix
from masterclass.storage.base import ObjectStorage


class QueuedJobType(StrEnum):
    EXTRACT_MEDIA = "extract_media"
    ANALYZE = "analyze"
    EVIDENCE_PACKET = "evidence_packet"
    TEACH = "teach"


class QueuedJobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedJob:
    job_id: str
    session: SessionRef
    job_type: QueuedJobType
    state: QueuedJobState = QueuedJobState.QUEUED
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    errors: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def new(session: SessionRef, job_type: QueuedJobType, payload: dict[str, Any] | None = None) -> "QueuedJob":
        return QueuedJob(job_id=uuid4().hex, session=session, job_type=job_type, payload=payload or {})

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "job_id": self.job_id,
            "session": {
                "tenant_id": self.session.tenant_id,
                "user_id": self.session.user_id,
                "session_id": self.session.session_id,
            },
            "job_type": self.job_type.value,
            "state": self.state.value,
            "payload": self.payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "errors": self.errors,
        }


class JobStore:
    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    def key(self, job: QueuedJob) -> str:
        return f"{session_prefix(job.session)}/jobs/{job.job_id}.json"

    def enqueue(self, session: SessionRef, job_type: QueuedJobType, payload: dict[str, Any] | None = None) -> QueuedJob:
        job = QueuedJob.new(session, job_type, payload)
        self.storage.write_json(self.key(job), job.to_json())
        return job
