from __future__ import annotations

from pathlib import Path

from masterclass.core.models import SessionManifest, SessionRef, TenantContext, session_prefix
from masterclass.storage.base import ObjectStorage


MANIFEST_NAME = "session.json"


class SessionStore:
    """ADLS-first session repository backed by an ObjectStorage implementation."""

    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    def create(
        self,
        ctx: TenantContext,
        *,
        source_filename: str | None = None,
        repertoire: str | None = None,
        movement: str | None = None,
        instrument: str | None = None,
        instrument_profile: str | None = None,
        notes: str | None = None,
    ) -> SessionManifest:
        ref = SessionRef.new(ctx)
        manifest = SessionManifest(
            schema_version=1,
            session=ref,
            source_filename=source_filename,
            repertoire=repertoire,
            movement=movement,
            instrument=instrument,
            instrument_profile=instrument_profile,
            notes=notes,
        )
        self.save(manifest)
        return manifest

    def manifest_key(self, ref: SessionRef) -> str:
        return f"{session_prefix(ref)}/{MANIFEST_NAME}"

    def artifact_key(self, ref: SessionRef, relative_key: str) -> str:
        if relative_key.startswith("/") or "\\" in relative_key or ".." in relative_key.split("/"):
            raise ValueError(f"unsafe artifact key: {relative_key}")
        return f"{session_prefix(ref)}/{relative_key}"

    def save(self, manifest: SessionManifest) -> None:
        manifest.touch()
        self.storage.write_json(self.manifest_key(manifest.session), manifest.to_json())

    def load(self, ref: SessionRef) -> SessionManifest:
        return SessionManifest.from_json(self.storage.read_json(self.manifest_key(ref)))

    def load_by_id(self, ctx: TenantContext, session_id: str) -> SessionManifest:
        if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError("session_id must be a path-safe id")
        return self.load(SessionRef(ctx.tenant_id, ctx.user_id, session_id))

    def list_for_user(self, ctx: TenantContext) -> list[SessionManifest]:
        prefix = f"tenant/{ctx.tenant_id}/users/{ctx.user_id}/sessions"
        manifests: list[SessionManifest] = []
        for key in self.storage.list_keys(prefix):
            if key.endswith(f"/{MANIFEST_NAME}"):
                manifests.append(SessionManifest.from_json(self.storage.read_json(key)))
        manifests.sort(key=lambda m: m.updated_at, reverse=True)
        return manifests

    def attach_local_file(self, manifest: SessionManifest, local_path: Path, artifact_key: str) -> str:
        key = self.artifact_key(manifest.session, artifact_key)
        self.storage.write_bytes(key, local_path.read_bytes())
        manifest.artifacts[artifact_key] = key
        self.save(manifest)
        return key

    def delete(self, ref: SessionRef) -> int:
        """Recursively delete every artifact under a session prefix and return
        the count of objects removed. Idempotent — deleting a non-existent
        session returns 0 instead of raising."""
        return self.storage.delete_prefix(session_prefix(ref))

    def delete_by_id(self, ctx: TenantContext, session_id: str) -> int:
        if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError("session_id must be a path-safe id")
        return self.delete(SessionRef(ctx.tenant_id, ctx.user_id, session_id))
