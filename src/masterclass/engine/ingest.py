from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from masterclass.core.models import JobState, SessionManifest, TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage
from masterclass.toolchain.ffmpeg import FfmpegToolchain


@dataclass(frozen=True)
class IngestRequest:
    tenant: TenantContext
    video_path: Path
    repertoire: str | None = None
    movement: str | None = None
    instrument: str | None = None
    instrument_profile: str | None = None
    notes: str | None = None
    frame_interval_sec: float = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_video(
    *,
    store: SessionStore,
    storage: ObjectStorage,
    ffmpeg: FfmpegToolchain,
    request: IngestRequest,
) -> SessionManifest:
    """Create a session and persist source/audio/metadata/frames through storage.

    The ffmpeg work happens in a temporary local workspace because ADLS-like
    object stores are not POSIX filesystems. Only the storage keys are recorded
    in the manifest.
    """

    video = request.video_path.expanduser().resolve()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")

    manifest = create_session_for_upload(
        store=store,
        request=request,
        source_filename=video.name,
        source_sha256=sha256_file(video),
        source_size_bytes=video.stat().st_size,
    )
    upload_source_file(store=store, storage=storage, manifest=manifest, video=video)
    return extract_media_artifacts(
        store=store,
        storage=storage,
        ffmpeg=ffmpeg,
        manifest=manifest,
        frame_interval_sec=request.frame_interval_sec,
    )


def create_session_for_upload(
    *,
    store: SessionStore,
    request: IngestRequest,
    source_filename: str,
    source_sha256: str | None = None,
    source_size_bytes: int | None = None,
) -> SessionManifest:
    manifest = store.create(
        request.tenant,
        source_filename=source_filename,
        repertoire=request.repertoire,
        movement=request.movement,
        instrument=request.instrument,
        instrument_profile=request.instrument_profile,
        notes=request.notes,
    )
    if source_sha256:
        manifest.metadata["source_sha256"] = source_sha256
    if source_size_bytes is not None:
        manifest.metadata["source_size_bytes"] = source_size_bytes
    store.save(manifest)
    return manifest


def upload_source_file(*, store: SessionStore, storage: ObjectStorage, manifest: SessionManifest, video: Path) -> str:
    source_key = store.artifact_key(manifest.session, f"input/{video.name}")
    storage.write_file(source_key, video, content_type="video/mp4")
    manifest.artifacts["input/source_video"] = source_key
    manifest.state = JobState.UPLOADED
    if "source_sha256" not in manifest.metadata:
        manifest.metadata["source_sha256"] = sha256_file(video)
    if "source_size_bytes" not in manifest.metadata:
        manifest.metadata["source_size_bytes"] = video.stat().st_size
    store.save(manifest)
    return source_key


def extract_media_artifacts(
    *,
    store: SessionStore,
    storage: ObjectStorage,
    ffmpeg: FfmpegToolchain,
    manifest: SessionManifest,
    frame_interval_sec: float,
) -> SessionManifest:
    source_key = manifest.artifacts.get("input/source_video")
    if not source_key:
        raise ValueError("manifest is missing input/source_video artifact")

    manifest.state = JobState.INGESTING
    store.save(manifest)

    with tempfile.TemporaryDirectory(prefix="masterclass-ingest-") as tmp_raw:
        tmp = Path(tmp_raw)
        source = tmp / (manifest.source_filename or "source.mp4")
        storage.read_to_file(source_key, source)
        metadata = ffmpeg.probe(source)
        audio = tmp / "audio.wav"
        frames = tmp / "frames"
        ffmpeg.extract_audio(source, audio)
        frame_paths = ffmpeg.extract_frames(source, frames, every_seconds=frame_interval_sec)

        metadata_key = store.artifact_key(manifest.session, "artifacts/metadata.json")
        audio_key = store.artifact_key(manifest.session, "artifacts/audio.wav")
        storage.write_json(metadata_key, metadata)
        storage.write_file(audio_key, audio, content_type="audio/wav")
        manifest.artifacts["artifacts/metadata.json"] = metadata_key
        manifest.artifacts["artifacts/audio.wav"] = audio_key
        manifest.metadata["frame_interval_sec"] = frame_interval_sec
        manifest.metadata["frame_count"] = len(frame_paths)

        frame_keys: list[str] = []
        for frame in frame_paths:
            key = store.artifact_key(manifest.session, f"artifacts/frames/{frame.name}")
            storage.write_file(key, frame, content_type="image/jpeg")
            frame_keys.append(key)
        manifest.artifacts["artifacts/frames"] = store.artifact_key(manifest.session, "artifacts/frames")
        manifest.metadata["frames"] = frame_keys

    manifest.state = JobState.INGESTED
    store.save(manifest)
    return manifest
