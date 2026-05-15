from masterclass.core.jobs import QueuedJobType
from masterclass.core.models import SessionManifest, SessionRef
from masterclass.core.sessions import SessionStore
from masterclass.engine.analysis import analyze_session, build_evidence_packet
from masterclass.engine.ingest import extract_media_artifacts
from masterclass.storage.base import ObjectStorage
from masterclass.toolchain.ffmpeg import FfmpegToolchain


def run_extract_media_job(
    *,
    store: SessionStore,
    storage: ObjectStorage,
    ffmpeg: FfmpegToolchain,
    manifest: SessionManifest,
    frame_interval_sec: float = 10.0,
) -> SessionManifest:
    return extract_media_artifacts(
        store=store,
        storage=storage,
        ffmpeg=ffmpeg,
        manifest=manifest,
        frame_interval_sec=frame_interval_sec,
    )


def run_job(
    *,
    store: SessionStore,
    storage: ObjectStorage,
    ffmpeg: FfmpegToolchain,
    job: dict,
) -> SessionManifest:
    session_data = job["session"]
    ref = SessionRef(
        tenant_id=session_data["tenant_id"],
        user_id=session_data["user_id"],
        session_id=session_data["session_id"],
    )
    manifest = store.load(ref)
    job_type = QueuedJobType(job["job_type"])
    if job_type == QueuedJobType.EXTRACT_MEDIA:
        return extract_media_artifacts(
            store=store,
            storage=storage,
            ffmpeg=ffmpeg,
            manifest=manifest,
            frame_interval_sec=float(job.get("payload", {}).get("frame_interval_sec", 10.0)),
        )
    if job_type == QueuedJobType.ANALYZE:
        return analyze_session(store=store, storage=storage, manifest=manifest)
    if job_type == QueuedJobType.EVIDENCE_PACKET:
        return build_evidence_packet(store=store, storage=storage, manifest=manifest)
    raise NotImplementedError(f"worker job type not implemented yet: {job_type}")


def run_worker_once() -> None:
    """Future queue worker entrypoint.

    The worker will claim one ADLS-backed job, run deterministic engine steps,
    and persist state transitions back into the session manifest.
    """

    raise NotImplementedError("queue worker not implemented yet")
