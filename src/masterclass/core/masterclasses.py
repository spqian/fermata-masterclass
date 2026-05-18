from __future__ import annotations

from masterclass.core.models import MasterclassManifest, MasterclassRef, TenantContext, masterclass_prefix
from masterclass.storage.base import ObjectStorage


MANIFEST_NAME = "masterclass.json"


class MasterclassStore:
    """ADLS-first repository for a sequence of lessons on one piece."""

    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    def create(
        self,
        ctx: TenantContext,
        *,
        piece_name: str,
        movement: str | None = None,
        instrument: str | None = None,
        instrument_profile: str | None = None,
        score_url: str | None = None,
        notes: str | None = None,
    ) -> MasterclassManifest:
        piece = piece_name.strip()
        if not piece:
            raise ValueError("piece_name is required")
        manifest = MasterclassManifest(
            schema_version=1,
            masterclass=MasterclassRef.new(ctx),
            piece_name=piece,
            movement=movement,
            instrument=instrument,
            instrument_profile=instrument_profile,
            score_url=score_url,
            notes=notes,
        )
        self.save(manifest)
        return manifest

    def manifest_key(self, ref: MasterclassRef) -> str:
        return f"{masterclass_prefix(ref)}/{MANIFEST_NAME}"

    def artifact_key(self, ref: MasterclassRef, relative_key: str) -> str:
        if relative_key.startswith("/") or "\\" in relative_key or ".." in relative_key.split("/"):
            raise ValueError(f"unsafe masterclass artifact key: {relative_key}")
        return f"{masterclass_prefix(ref)}/{relative_key}"

    def save(self, manifest: MasterclassManifest) -> None:
        manifest.touch()
        self.storage.write_json(self.manifest_key(manifest.masterclass), manifest.to_json())

    def load(self, ref: MasterclassRef) -> MasterclassManifest:
        return MasterclassManifest.from_json(self.storage.read_json(self.manifest_key(ref)))

    def load_by_id(self, ctx: TenantContext, masterclass_id: str) -> MasterclassManifest:
        if not masterclass_id or "/" in masterclass_id or "\\" in masterclass_id or ".." in masterclass_id:
            raise ValueError("masterclass_id must be a path-safe id")
        return self.load(MasterclassRef(ctx.tenant_id, ctx.user_id, masterclass_id))

    def list_for_user(self, ctx: TenantContext) -> list[MasterclassManifest]:
        prefix = f"tenant/{ctx.tenant_id}/users/{ctx.user_id}/masterclasses"
        manifests: list[MasterclassManifest] = []
        for key in self.storage.list_keys(prefix):
            if key.endswith(f"/{MANIFEST_NAME}"):
                manifests.append(MasterclassManifest.from_json(self.storage.read_json(key)))
        manifests.sort(key=lambda m: m.updated_at, reverse=True)
        return manifests

    def add_lesson(self, manifest: MasterclassManifest, session_id: str) -> None:
        if session_id not in manifest.lessons:
            manifest.lessons.append(session_id)
            self.save(manifest)

    def delete(self, ref: MasterclassRef) -> int:
        """Recursively delete every artifact under a masterclass prefix and
        return the count of objects removed. NOTE: the caller is responsible
        for first removing child sessions (lessons + drills) — this only
        removes the masterclass's OWN data (manifest, score files, etc.)."""
        return self.storage.delete_prefix(masterclass_prefix(ref))

    def delete_by_id(self, ctx: TenantContext, masterclass_id: str) -> int:
        if not masterclass_id or "/" in masterclass_id or "\\" in masterclass_id or ".." in masterclass_id:
            raise ValueError("masterclass_id must be a path-safe id")
        return self.delete(MasterclassRef(ctx.tenant_id, ctx.user_id, masterclass_id))
