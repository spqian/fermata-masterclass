from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from masterclass.auth.encryption import ensure_key_encryption_key
from masterclass.auth.google_oauth import (
    clear_session_cookie,
    complete_google_callback,
    get_session_user_id,
    build_login_redirect,
    set_session_cookie,
    OAUTH_COOKIE_NAME,
)
from masterclass.core.artifact_catalog import ArtifactCatalog
from masterclass.core.jobs import JobStore, QueuedJobType
from masterclass.core.chat_models import delete_conversation, list_conversations, load_conversation
from masterclass.core.conversation_lock import conversation_lock
from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import JobState, SESSION_KIND_DRILL, SESSION_KIND_LESSON
from masterclass.core.models import TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.core.user_profiles import DEFAULT_MODEL, UserProfileStore
from masterclass.engine.score_prep import ScorePrepConfig, prepare_score, select_score_pages_for_lesson
from masterclass.engine.analysis import analyze_session, build_evidence_packet
from masterclass.engine.audio_truth import run_audio_truth_pipeline
from masterclass.engine.ingest import extract_media_artifacts
from masterclass.engine.instruments import intonation_enabled_for_profile, load_instrument_profile
from masterclass.engine.intonation import analyze_intonation
from masterclass.engine.mechanical_comments import generate_mechanical_comments, persist_mechanical_comments
from masterclass.engine.onsets import detect_rich_onsets
from masterclass.engine.rhythm import analyze_rhythm, persist_rhythm
from masterclass.engine.debug_spectrogram import render_window as render_spectrogram_window
from masterclass.engine.score_map import build_score_map, persist_score_map
from masterclass.engine.chat_guardrails import (
    ChatGuardrailError,
    check_conversation_turn_cap,
    check_message_size,
    check_user_quota,
    topic_guard,
)
from masterclass.engine.teach_chat import ChatConfig, chat_usage_dict, run_chat_turn
from masterclass.engine.teach_lesson import TeachConfig, teach_lesson
from masterclass.engine.voicing import analyze_voicing, persist_voicing
from masterclass.storage.base import ObjectStorage
from masterclass.storage.local import LocalObjectStorage
from masterclass.toolchain.ffmpeg import FfmpegToolchain

try:
    from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
    from fastapi.responses import HTMLResponse, RedirectResponse, Response
    from pydantic import BaseModel
except ImportError:
    BackgroundTasks = Body = Depends = FastAPI = File = Form = Header = HTTPException = Request = HTMLResponse = RedirectResponse = Response = UploadFile = BaseModel = None


if BaseModel is not None:
    class ChatRequest(BaseModel):
        message: str
        conversation_id: str | None = None
        comment_id: str | None = None

    class ProfilePatch(BaseModel):
        gemini_api_key: str | None = None
        clear_gemini_key: bool = False
        preferred_model: str | None = None
        pro_mode: bool | None = None


_COMMENT_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _comment_conversation_id(comment_id: str) -> str:
    """Map a teacher-comment id (e.g. ``g_007``) to its reply-thread file id.

    Path-sanitises the user-supplied id then prefixes ``cmt_`` so the chat
    listing endpoint can later distinguish per-comment reply threads from the
    main lesson conversation.
    """
    safe = _COMMENT_ID_SAFE.sub("_", (comment_id or "").strip())
    if not safe:
        raise ValueError("comment_id is required")
    return f"cmt_{safe[:64]}"


def _require_lesson_kind(manifest) -> None:
    """Reject access if the loaded session is a drill, not a lesson.

    Every /lessons/{id}/* endpoint runs this guard so drill sessions
    (uploaded via /lessons/{id}/comments/{cid}/practice-clip and the
    masterclass-level practice-clip endpoint) can't be mistaken for
    lessons and pushed through the full teach pipeline, which would
    fail score-matching against a recording of a different passage.
    """
    kind = getattr(manifest, "kind", SESSION_KIND_LESSON) or SESSION_KIND_LESSON
    if kind != SESSION_KIND_LESSON:
        from fastapi import HTTPException  # local import: tests stub FastAPI
        raise HTTPException(
            status_code=409,
            detail=f"session is a {kind}, not a lesson; use the /drills/* endpoints",
        )


def _require_drill_kind(manifest) -> None:
    kind = getattr(manifest, "kind", SESSION_KIND_LESSON) or SESSION_KIND_LESSON
    if kind != SESSION_KIND_DRILL:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=409,
            detail=f"session is a {kind}, not a drill",
        )


def _load_lesson_manifest(store, ctx, session_id):
    """Load a session, 404 if missing, 409 if it's a drill not a lesson.

    Used by every /lessons/{id}/* endpoint so the drill-vs-lesson split
    has one chokepoint.
    """
    from fastapi import HTTPException
    try:
        manifest = store.load_by_id(ctx, session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="lesson not found") from exc
    _require_lesson_kind(manifest)
    return manifest


def _build_storage() -> ObjectStorage:
    backend = os.environ.get("MASTERCLASS_STORAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalObjectStorage(Path(os.environ.get("MASTERCLASS_LOCAL_ADLS_ROOT", "local_adls")))
    if backend == "adls":
        from masterclass.storage.adls import AdlsObjectStorage

        account_url = os.environ["MASTERCLASS_ADLS_ACCOUNT_URL"]
        file_system = os.environ["MASTERCLASS_ADLS_FILE_SYSTEM"]
        return AdlsObjectStorage(account_url=account_url, file_system=file_system)
    raise ValueError(f"unknown storage backend: {backend}")


def _load_dotenv() -> None:
    """Load ``.env`` from the v2 project root if present.

    Standalone replacement for python-dotenv: keeps the package light and
    avoids one more dependency for a one-line config file.
    """

    root = Path(__file__).resolve().parents[4]
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def create_app():
    """Create the future FastAPI app.

    Kept intentionally thin: authz, request validation, and job scheduling live
    here; music processing stays in engine/worker packages.
    """

    if FastAPI is None:
        raise RuntimeError("Install API dependencies with: pip install -e .[api]")

    _load_dotenv()
    ensure_key_encryption_key()
    app = FastAPI(title="Music Masterclass API")
    storage = _build_storage()
    masterclasses = MasterclassStore(storage)
    store = SessionStore(storage)
    jobs = JobStore(storage)
    user_profiles = UserProfileStore(storage)
    static_dir = Path(__file__).resolve().parent / "static"
    score_prep_model = os.environ.get("MASTERCLASS_SCORE_PREP_MODEL", "gemini-2.5-pro")
    midi_find_model = os.environ.get("MASTERCLASS_MIDI_FIND_MODEL", "gemini-2.5-flash")

    def _truthy(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _server_default_key_allowed() -> bool:
        return _truthy(os.environ.get("ALLOW_SERVER_DEFAULT_KEY"))

    def _preferred_model_for_user(user_id: str) -> str:
        try:
            return user_profiles.load(user_id).preferred_model
        except FileNotFoundError:
            return DEFAULT_MODEL

    def _gemini_api_key_for_user(user_id: str) -> str:
        try:
            user_key = user_profiles.get_gemini_key_plain(user_id)
        except FileNotFoundError:
            user_key = None
        if user_key:
            return user_key
        if _server_default_key_allowed():
            server_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
            if server_key:
                return server_key
        raise HTTPException(status_code=402, detail="Add your Gemini API key in Settings to run a lesson")

    def _build_llm_provider_for_user(user_id: str, config=None):
        if os.environ.get("MASTERCLASS_LLM_PROVIDER", "gemini").lower() == "dry-run":
            from masterclass.agent.dry_run import DryRunLlmProvider
            return DryRunLlmProvider()
        from masterclass.agent.gemini import SharedKeyGeminiProvider
        return SharedKeyGeminiProvider(api_key=_gemini_api_key_for_user(user_id), config=config)

    def _downsample_audio_for_teach(ffmpeg: FfmpegToolchain, source_audio_key: str) -> bytes:
        """Compress lesson audio to 16 kHz mono PCM so the inline multimodal call fits.

        16 kHz × 2 bytes/sample ≈ 32 KB/sec; 28 MB cap → ~14 minutes of audio,
        which covers a normal masterclass lesson without hitting Gemini's
        inline limits.
        """

        import tempfile
        with tempfile.TemporaryDirectory(prefix="masterclass-teach-audio-") as tmp_raw:
            src = Path(tmp_raw) / "source.wav"
            dst = Path(tmp_raw) / "compact.wav"
            storage.read_to_file(source_audio_key, src)
            from masterclass.toolchain.process import run_process
            run_process([
                str(ffmpeg.ffmpeg), "-y", "-i", str(src),
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(dst),
            ], timeout_sec=900)
            return dst.read_bytes()

    def _run_lesson_jobs(session_id: str, tenant_id: str, user_id: str, masterclass_id: str | None, *, resume: bool = False) -> None:
        """Drain a lesson's queued jobs sequentially inside this API process.

        MVP equivalent of an out-of-process worker: keeps the API responsive
        for upload, runs deterministic engine steps in the background, then
        runs the multimodal teacher, and updates session state so the UI can
        poll progress.

        When ``resume=True``, every stage whose ``{stage}_state`` is already
        ``ready`` (or a ``skipped*`` variant) is skipped — the pipeline only
        re-runs failed / pending stages. This is what /retry-failed uses so
        a transient Gemini API error doesn't force a 4-minute re-ingest.
        """

        import logging
        try:
            ctx = TenantContext(tenant_id=tenant_id, user_id=user_id)
            ffmpeg = FfmpegToolchain.discover()
            manifest = store.load_by_id(ctx, session_id)
            try:
                def mark_stage(stage: str, state: str, error: str | None = None) -> None:
                    manifest.metadata[f"{stage}_state"] = state
                    manifest.metadata[f"{stage}_error"] = error
                    manifest.metadata[f"{stage}_updated_at"] = datetime.now(UTC).isoformat()
                    store.save(manifest)

                def _stage_done(stage: str) -> bool:
                    """In resume mode, a stage is done if it's ready or skipped*."""
                    if not resume:
                        return False
                    cur = manifest.metadata.get(f"{stage}_state") or ""
                    return cur == "ready" or cur.startswith("skipped")

                def run_required(stage: str, func) -> None:
                    """Strict stage: failure aborts the whole pipeline. Skip if
                    resuming and already done."""
                    if _stage_done(stage):
                        return
                    mark_stage(stage, "running")
                    result = func()
                    if result is not None and hasattr(result, "metadata"):
                        # extract_media/analyze/build_evidence_packet return a
                        # fresh manifest. Update our local binding so subsequent
                        # mark_stage / run_required calls see the latest state.
                        nonlocal manifest
                        manifest = result
                    mark_stage(stage, "ready")

                run_required("extract_media", lambda: extract_media_artifacts(store=store, storage=storage, ffmpeg=ffmpeg, manifest=manifest, frame_interval_sec=10.0))
                run_required("analyze", lambda: analyze_session(store=store, storage=storage, manifest=manifest))
                run_required("evidence_packet", lambda: build_evidence_packet(store=store, storage=storage, manifest=manifest))

                def warn_stage(stage: str, exc: Exception) -> None:
                    manifest.metadata[f"{stage}_state"] = "failed"
                    manifest.metadata[f"{stage}_error"] = f"{type(exc).__name__}: {exc}"
                    manifest.errors.append({
                        "stage": stage,
                        "warning": True,
                        "error": manifest.metadata[f"{stage}_error"],
                        "at": datetime.now(UTC).isoformat(),
                    })
                    store.save(manifest)

                def run_best_effort(stage: str, func) -> None:
                    if _stage_done(stage):
                        return
                    manifest.metadata[f"{stage}_state"] = "running"
                    store.save(manifest)
                    try:
                        func()
                        manifest.metadata[f"{stage}_state"] = "ready"
                        manifest.metadata[f"{stage}_error"] = None
                        manifest.metadata[f"{stage}_updated_at"] = datetime.now(UTC).isoformat()
                        store.save(manifest)
                    except Exception as exc:
                        warn_stage(stage, exc)

                class_manifest = None
                if masterclass_id:
                    try:
                        class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
                    except FileNotFoundError:
                        class_manifest = None

                run_best_effort("onsets", lambda: detect_rich_onsets(storage=storage, store=store, manifest=manifest))

                # Audio-truth alignment: primary timing source. Uses ByteDance
                # PTI for piano lessons and Spotify basic-pitch for everything
                # else, then matches detected notes against the reference score
                # (MusicXML preferred, MIDI fallback) for per-note staff/voice/
                # measure tagging. Output artifacts:
                #   analysis/audio_truth_notes.json          (raw transcriber)
                #   analysis/audio_truth_matched_notes.json  (+ score tagging)
                #   analysis/aligned_notes.json              (legacy shim, new name)
                #   analysis/hmm_aligned_notes.json          (legacy shim, deprecated name)
                #   analysis/hmm_alignment.json              (legacy shim, deprecated name)
                # The legacy shim artifacts are synthesised from audio_truth
                # data so the old consumers (voicing/rhythm/intonation/
                # score_map/inspect_*) keep working unchanged. They will be
                # removed once those consumers are migrated to read
                # audio_truth_matched_notes directly.
                run_best_effort(
                    "audio_truth",
                    lambda: run_audio_truth_pipeline(storage=storage, store=store, manifest=manifest),
                )
                # Dual-write the audio-truth status under the vocabulary-clean
                # key ``aligned_notes_state`` so newer UI/code can read either
                # name. We keep writing ``audio_truth_state`` (above) until
                # every consumer has migrated; eventually that key, the
                # legacy hmm-named shim artifacts, and this dual-write all
                # go away in the same release.
                _at_state = manifest.metadata.get("audio_truth_state")
                if _at_state is not None:
                    manifest.metadata["aligned_notes_state"] = _at_state
                    if "audio_truth_error" in manifest.metadata:
                        manifest.metadata["aligned_notes_error"] = manifest.metadata.get("audio_truth_error")
                    if "audio_truth_updated_at" in manifest.metadata:
                        manifest.metadata["aligned_notes_updated_at"] = manifest.metadata["audio_truth_updated_at"]
                    store.save(manifest)

                run_best_effort(
                    "score_map",
                    lambda: persist_score_map(
                        storage=storage,
                        store=store,
                        manifest=manifest,
                        result=build_score_map(storage=storage, masterclass_store=masterclasses, store=store, manifest=manifest),
                    ),
                )

                profile = load_instrument_profile(manifest.instrument_profile)
                # Gate downstream analyses on the audio-truth stage status.
                # We read ``audio_truth_state`` here for back-compat; the
                # equivalent vocabulary-clean key ``aligned_notes_state``
                # is mirrored immediately after the audio-truth run above
                # so newer code can read either name. Both will live for
                # one release; once every consumer has migrated to
                # ``aligned_notes_state``, remove the dual-write and read
                # only the new name.
                if intonation_enabled_for_profile(profile) and manifest.metadata.get("audio_truth_state") == "ready":
                    run_best_effort("intonation", lambda: analyze_intonation(storage=storage, store=store, manifest=manifest))
                else:
                    manifest.metadata["intonation_state"] = "skipped"
                    store.save(manifest)

                if manifest.metadata.get("audio_truth_state") == "ready":
                    run_best_effort(
                        "rhythm",
                        lambda: persist_rhythm(
                            storage=storage,
                            store=store,
                            manifest=manifest,
                            result=analyze_rhythm(storage=storage, store=store, manifest=manifest),
                        ),
                    )
                else:
                    manifest.metadata["rhythm_state"] = "skipped"
                    store.save(manifest)

                if profile.family == "keyboard" and manifest.metadata.get("audio_truth_state") == "ready":
                    run_best_effort(
                        "voicing",
                        lambda: persist_voicing(
                            storage=storage,
                            store=store,
                            manifest=manifest,
                            result=analyze_voicing(storage=storage, store=store, manifest=manifest),
                        ),
                    )
                else:
                    manifest.metadata["voicing_state"] = "skipped"
                    store.save(manifest)

                if manifest.metadata.get("rhythm_state") == "ready":
                    run_best_effort(
                        "mechanical_comments",
                        lambda: persist_mechanical_comments(
                            storage=storage,
                            store=store,
                            manifest=manifest,
                            result=generate_mechanical_comments(storage=storage, store=store, manifest=manifest),
                        ),
                    )
                else:
                    manifest.metadata["mechanical_comments_state"] = "skipped"
                    store.save(manifest)

                try:
                    provider = _build_llm_provider_for_user(user_id)
                except HTTPException as exc:
                    manifest.metadata["teach_state"] = "failed"
                    manifest.metadata["teach_error"] = str(exc.detail)
                    manifest.state = JobState.READY
                    store.save(manifest)
                    return

                score_pages: list[bytes] = []
                score_layout: list[dict[str, Any]] = []
                if class_manifest is not None:
                    try:
                        score_pages, score_layout = select_score_pages_for_lesson(
                            storage=storage,
                            masterclass=class_manifest,
                            first_measure=manifest.metadata.get("first_measure"),
                            last_measure=manifest.metadata.get("last_measure"),
                        )
                    except FileNotFoundError:
                        score_pages, score_layout = [], []

                manifest.metadata["teach_state"] = "running"
                store.save(manifest)
                audio_key = ArtifactCatalog(manifest).audio_wav()
                if not audio_key:
                    raise RuntimeError("audio missing after extract_media")
                compact_audio = _downsample_audio_for_teach(ffmpeg, audio_key)
                compact_key = store.artifact_key(manifest.session, "artifacts/audio_16k.wav")
                storage.write_bytes(compact_key, compact_audio, content_type="audio/wav")
                manifest.artifacts["artifacts/audio_16k.wav"] = compact_key
                store.save(manifest)
                try:
                    manifest = teach_lesson(
                        storage=storage,
                        store=store,
                        manifest=manifest,
                        provider=provider,
                        score_pages=score_pages,
                        score_layout=score_layout,
                        config=TeachConfig(model=os.environ.get("MASTERCLASS_TEACH_MODEL", _preferred_model_for_user(user_id))),
                    )
                except Exception as exc:
                    manifest.metadata["teach_state"] = "failed"
                    manifest.metadata["teach_error"] = f"{type(exc).__name__}: {exc}"
                    manifest.errors.append({
                        "stage": "teach",
                        "warning": True,
                        "error": manifest.metadata["teach_error"],
                        "at": datetime.now(UTC).isoformat(),
                    })
                    manifest.state = JobState.READY
                    store.save(manifest)
            except Exception as exc:
                manifest.state = JobState.FAILED
                manifest.errors.append({
                    "stage": "lesson_jobs",
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": datetime.now(UTC).isoformat(),
                })
                store.save(manifest)
                raise
        except Exception:
            logging.exception("lesson background jobs failed for %s", session_id)

    def _spawn(target, *args, **kwargs) -> None:
        """Run a background job in a real OS thread so multiple jobs run in parallel.
        FastAPI's BackgroundTasks runs handlers sequentially, which can starve
        independent tasks (e.g. score_prep should not block the lesson upload
        response)."""
        import threading
        import functools
        fn = functools.partial(target, *args, **kwargs) if kwargs else target
        threading.Thread(target=fn if kwargs else target, args=() if kwargs else args, daemon=True).start()

    def _run_score_prep(masterclass_id: str, tenant_id: str, user_id: str) -> None:
        try:
            ctx = TenantContext(tenant_id=tenant_id, user_id=user_id)
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
            try:
                provider = _build_llm_provider_for_user(user_id)
            except HTTPException as exc:
                manifest.metadata["score_prep_state"] = "skipped"
                manifest.metadata["score_prep_error"] = str(exc.detail)
                manifest.metadata["score_prep_updated_at"] = datetime.now(UTC).isoformat()
                masterclasses.save(manifest)
                return
            # score_prep used to wait up to 90s for the Gemini midi_finder to
            # land so it could cross-check measure counts. midi_finder is
            # removed (audio-truth reads MusicXML directly from Audiveris),
            # so the wait would always time out; just proceed.
            prepare_score(
                storage=storage,
                masterclass_store=masterclasses,
                manifest=manifest,
                provider=provider,
                config=ScorePrepConfig(model=score_prep_model),
            )
        except Exception:  # pragma: no cover - background errors land on the manifest
            import logging
            logging.exception("score prep failed for masterclass %s", masterclass_id)

    class CreateSessionRequest(BaseModel):
        source_filename: str | None = None
        repertoire: str | None = None
        movement: str | None = None
        instrument: str | None = None
        instrument_profile: str | None = None
        notes: str | None = None

    def tenant_from_header(request: Request) -> TenantContext:
        """Resolve the caller's tenant strictly from the signed session cookie.

        The previous X-User-Id header and ?user_id= query fallbacks were a full
        IDOR primitive — any client could impersonate any account by setting the
        header. Now identity comes solely from the HttpOnly session cookie that
        ``set_session_cookie`` produces after a verified Google sign-in. Forging
        the cookie requires MASTERCLASS_KEY_ENCRYPTION_KEY.
        """
        resolved = (get_session_user_id(request) or "").strip()
        if not resolved:
            raise HTTPException(status_code=401, detail="Sign in with Google to continue")
        try:
            return TenantContext(tenant_id=resolved, user_id=resolved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.get("/auth/login")
    def auth_login(request: Request, next: str | None = "/"):
        return build_login_redirect(request, next)

    @app.get("/auth/callback", name="auth_callback")
    async def auth_callback(request: Request):
        identity, next_url = await complete_google_callback(request)
        profile = user_profiles.upsert_oauth_user(
            google_sub=identity.google_sub,
            email=identity.email,
            display_name=identity.display_name,
        )
        response = RedirectResponse(next_url or "/", status_code=302)
        set_session_cookie(response, request, profile.google_sub)
        response.delete_cookie(OAUTH_COOKIE_NAME, path="/")
        return response

    @app.post("/auth/logout")
    def auth_logout(request: Request):
        # CSRF defense: only honor logout requests from a same-origin context.
        # All modern browsers send Origin on POST; fall back to Referer if absent.
        origin = (request.headers.get("origin") or "").rstrip("/")
        referer = request.headers.get("referer") or ""
        expected_root = f"{request.url.scheme}://{request.url.netloc}"
        same_origin = (origin == expected_root) or (referer and referer.startswith(expected_root + "/"))
        if not same_origin:
            raise HTTPException(status_code=403, detail="Cross-origin logout blocked")
        response = RedirectResponse("/", status_code=303)
        clear_session_cookie(response, request)
        return response

    @app.get("/auth/dev-login")
    def auth_dev_login() -> None:
        """Removed. Use real Google OAuth via /auth/login."""
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/auth/me")
    def auth_me(request: Request) -> dict:
        user_id = get_session_user_id(request)
        if not user_id:
            raise HTTPException(status_code=401, detail="not signed in")
        try:
            return user_profiles.load(user_id).public_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=401, detail="profile not found") from exc

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page() -> HTMLResponse:
        return HTMLResponse(
            (static_dir / "settings.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.get("/api/me/profile")
    def get_my_profile(ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            return user_profiles.load(ctx.user_id).public_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=401, detail="Sign in with Google to manage your profile") from exc

    @app.patch("/api/me/profile")
    def patch_my_profile(
        body: ProfilePatch = Body(...),
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        try:
            profile = user_profiles.load(ctx.user_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=401, detail="Sign in with Google to manage your profile") from exc
        try:
            if body.preferred_model is not None:
                profile = user_profiles.set_preferred_model(ctx.user_id, body.preferred_model)
            if body.clear_gemini_key:
                profile = user_profiles.clear_gemini_key(ctx.user_id)
            if body.gemini_api_key is not None and body.gemini_api_key.strip():
                profile = user_profiles.set_gemini_key(ctx.user_id, body.gemini_api_key)
            if body.pro_mode is not None:
                profile = user_profiles.set_pro_mode(ctx.user_id, body.pro_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return profile.public_json()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def ingest_page() -> HTMLResponse:
        return HTMLResponse(
            (static_dir / "ingest.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    def _clean_optional(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    async def _attach_reference_files(
        *,
        manifest,
        score_file: UploadFile | None,
        midi_file: UploadFile | None,
    ) -> None:
        if score_file and score_file.filename:
            score_name = Path(score_file.filename).name
            if Path(score_name).suffix.lower() != ".pdf":
                raise HTTPException(status_code=400, detail="score_file must be a PDF")
            score_content = await score_file.read()
            score_key = masterclasses.artifact_key(manifest.masterclass, f"reference/score/{score_name}")
            storage.write_bytes(score_key, score_content, content_type=score_file.content_type or "application/pdf")
            manifest.artifacts["reference/score_pdf"] = score_key
            manifest.metadata["reference_score_filename"] = score_name
            manifest.metadata["reference_score_size_bytes"] = len(score_content)
            manifest.metadata["reference_score_uploaded_at"] = datetime.now(UTC).isoformat()
        if midi_file and midi_file.filename:
            midi_name = Path(midi_file.filename).name
            if Path(midi_name).suffix.lower() not in {".mid", ".midi"}:
                raise HTTPException(status_code=400, detail="midi_file must be .mid or .midi")
            midi_content = await midi_file.read()
            midi_key = masterclasses.artifact_key(manifest.masterclass, f"reference/midi/{midi_name}")
            storage.write_bytes(midi_key, midi_content, content_type=midi_file.content_type or "audio/midi")
            manifest.artifacts["reference/midi"] = midi_key
            manifest.metadata["reference_midi_filename"] = midi_name
            manifest.metadata["reference_midi_size_bytes"] = len(midi_content)
            manifest.metadata["reference_midi_uploaded_at"] = datetime.now(UTC).isoformat()

    def _read_past_lesson_full(lesson_manifest) -> dict[str, Any]:
        """Read the prior lesson's enriched output: comments + lesson block + dropped."""
        for artifact_name in ("lesson/comments_enriched.json", "lesson/comments.json", "player/comments_enriched.json", "player/comments.json"):
            key = lesson_manifest.artifacts.get(artifact_name)
            if not key:
                # Fallback: try the conventional path even if not in artifacts map.
                key = store.artifact_key(lesson_manifest.session, artifact_name)
            if not key or not storage.exists(key):
                continue
            try:
                payload = storage.read_json(key)
            except (FileNotFoundError, ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            comments_raw = payload.get("comments") or []
            comments: list[dict[str, Any]] = []
            if isinstance(comments_raw, list):
                for comment in comments_raw:
                    if not isinstance(comment, dict):
                        continue
                    comments.append({
                        "id": comment.get("id"),
                        "start": comment.get("start"),
                        "end": comment.get("end"),
                        "measure": comment.get("measure"),
                        "category": comment.get("category"),
                        "severity": comment.get("severity"),
                        "summary": comment.get("summary") or comment.get("title") or comment.get("text"),
                        "text": comment.get("text") or comment.get("comment"),
                    })
            return {
                "summary": payload.get("summary"),
                "progress_notes": payload.get("progress_notes"),
                "lesson": payload.get("lesson") or {},
                "comments": comments,
            }
        return {"summary": None, "progress_notes": None, "lesson": {}, "comments": []}

    def _build_prior_lesson_context(class_manifest, current_session_id: str) -> dict[str, Any]:
        lessons: list[dict[str, Any]] = []
        for session_id in class_manifest.lessons:
            if session_id == current_session_id:
                continue
            try:
                lesson = store.load_by_id(
                    TenantContext(class_manifest.masterclass.tenant_id, class_manifest.masterclass.user_id),
                    session_id,
                )
            except FileNotFoundError:
                continue
            full = _read_past_lesson_full(lesson)
            lessons.append({
                "session_id": session_id,
                "created_at": lesson.created_at,
                "updated_at": lesson.updated_at,
                "state": lesson.state.value,
                "movement": lesson.movement,
                "notes": lesson.notes,
                "first_measure": lesson.metadata.get("first_measure") or lesson.metadata.get("auto_detected_first_measure"),
                "last_measure": lesson.metadata.get("last_measure") or lesson.metadata.get("auto_detected_last_measure"),
                "summary": full["summary"],
                "progress_notes": full["progress_notes"],
                "lesson": full["lesson"],
                "teacher_comments": full["comments"],
            })
        # Sort prior lessons oldest-first so the teacher reads chronologically.
        lessons.sort(key=lambda l: (l.get("created_at") or ""))
        return {
            "masterclass_id": class_manifest.masterclass.masterclass_id,
            "piece_name": class_manifest.piece_name,
            "movement": class_manifest.movement,
            "work_id": class_manifest.work_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "lesson_count": len(lessons),
            "lessons": lessons,
        }

    @app.get("/masterclasses")
    def list_masterclasses(ctx: TenantContext = Depends(tenant_from_header)) -> list[dict]:
        return [manifest.to_json() for manifest in masterclasses.list_for_user(ctx)]

    @app.get("/masterclasses/{masterclass_id}")
    def get_masterclass(masterclass_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            return masterclasses.load_by_id(ctx, masterclass_id).to_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc

    @app.post("/masterclasses")
    async def create_masterclass(
        background: BackgroundTasks,
        piece_name: str = Form(...),
        movement: str | None = Form(default=None),
        instrument: str | None = Form(default=None),
        instrument_profile: str | None = Form(default=None),
        score_url: str | None = Form(default=None),
        score_file: UploadFile | None = File(default=None),
        midi_file: UploadFile | None = File(default=None),
        notes: str | None = Form(default=None),
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        has_score_pdf = bool(score_file and score_file.filename)
        has_midi = bool(midi_file and midi_file.filename)
        if not has_score_pdf and not has_midi:
            # Score reference is still required: PDF is the most useful, MIDI works
            # as a fallback. We will *also* try to auto-find MIDI from the web, but
            # we still need at least one user-supplied reference to anchor the piece.
            raise HTTPException(
                status_code=400,
                detail="Attach a score PDF (recommended) or a MIDI file as a starting reference.",
            )
        try:
            manifest = masterclasses.create(
                ctx,
                piece_name=piece_name,
                movement=_clean_optional(movement),
                instrument=_clean_optional(instrument),
                instrument_profile=_clean_optional(instrument_profile),
                score_url=_clean_optional(score_url),
                notes=_clean_optional(notes),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _attach_reference_files(manifest=manifest, score_file=score_file, midi_file=midi_file)
        if has_score_pdf:
            manifest.metadata["score_prep_state"] = "queued"
            manifest.metadata["score_prep_updated_at"] = datetime.now(UTC).isoformat()
        masterclasses.save(manifest)
        if has_score_pdf:
            _spawn(
                _run_score_prep,
                manifest.masterclass.masterclass_id,
                manifest.masterclass.tenant_id,
                manifest.masterclass.user_id,
            )
        return manifest.to_json()

    @app.post("/masterclasses/{masterclass_id}/find-midi")
    def rerun_midi_find_deprecated(
        masterclass_id: str,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        """Deprecated. MIDI auto-find was removed when the audio-truth
        pipeline migrated to MusicXML for score correlation. The endpoint
        is kept as a 410 Gone for backward compatibility with old clients."""
        raise HTTPException(
            status_code=410,
            detail="MIDI auto-find has been removed. Audio alignment now uses the "
                   "MusicXML score directly via the audio-truth pipeline.",
        )

    @app.get("/masterclasses/{masterclass_id}/midi-find")
    def get_midi_find_deprecated(masterclass_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        """Deprecated alias matching the old midi-find shape, with empty
        results so any UI still polling this endpoint sees a stable response."""
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        return {
            "state": "deprecated",
            "error": None,
            "updated_at": manifest.metadata.get("score_prep_updated_at"),
            "midi_attached": "reference/midi" in manifest.artifacts,
            "midi_url": None,
            "midi_source": None,
            "midi_attribution": None,
            "midi_confidence": None,
            "audit": None,
        }

    @app.post("/masterclasses/{masterclass_id}/prepare-score")
    def rerun_score_prep(
        masterclass_id: str,
        background: BackgroundTasks,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        if "reference/score_pdf" not in manifest.artifacts:
            raise HTTPException(status_code=400, detail="masterclass has no PDF score attached")
        manifest.metadata["score_prep_state"] = "queued"
        manifest.metadata["score_prep_updated_at"] = datetime.now(UTC).isoformat()
        masterclasses.save(manifest)
        _spawn(
            _run_score_prep,
            manifest.masterclass.masterclass_id,
            manifest.masterclass.tenant_id,
            manifest.masterclass.user_id,
        )
        return manifest.to_json()

    @app.get("/masterclasses/{masterclass_id}/score-prep")
    def get_score_prep(masterclass_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        prep_key = manifest.artifacts.get("reference/score_prep.json")
        prep: dict[str, Any] | None = None
        if prep_key:
            try:
                prep = storage.read_json(prep_key)
            except FileNotFoundError:
                prep = None
        page_keys = [
            (rel, key)
            for rel, key in manifest.artifacts.items()
            if rel.startswith("reference/score_pages/") and rel.endswith(".png")
        ]
        page_keys.sort(key=lambda item: item[0])
        return {
            "state": manifest.metadata.get("score_prep_state", "not_run"),
            "error": manifest.metadata.get("score_prep_error"),
            "updated_at": manifest.metadata.get("score_prep_updated_at"),
            "started_at": manifest.metadata.get("score_prep_started_at"),
            "substage": manifest.metadata.get("score_prep_substage"),
            "elapsed_sec": manifest.metadata.get("score_prep_elapsed_sec"),
            "first_music_page": manifest.metadata.get("score_prep_first_music_page"),
            "movement_count": manifest.metadata.get("score_prep_movement_count"),
            "page_count": len(page_keys),
            "prep": prep,
        }

    class ScorePrepOverride(BaseModel):
        first_music_page: int | None = None
        movements: list[dict[str, Any]] | None = None
        pages: list[dict[str, Any]] | None = None
        notes: str | None = None

    @app.patch("/masterclasses/{masterclass_id}/score-prep")
    def patch_score_prep(
        masterclass_id: str,
        body: ScorePrepOverride,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        prep_key = manifest.artifacts.get("reference/score_prep.json")
        prep: dict[str, Any] = {}
        if prep_key:
            try:
                prep = storage.read_json(prep_key) or {}
            except FileNotFoundError:
                prep = {}
        update = body.model_dump(exclude_none=True)
        prep.update(update)
        prep.setdefault("_meta", {})
        prep["_meta"]["last_user_edit_at"] = datetime.now(UTC).isoformat()
        if not prep_key:
            prep_key = masterclasses.artifact_key(manifest.masterclass, "reference/score_prep.json")
        storage.write_json(prep_key, prep)
        manifest.artifacts["reference/score_prep.json"] = prep_key
        manifest.metadata["score_prep_state"] = "ready"
        if "first_music_page" in update:
            manifest.metadata["score_prep_first_music_page"] = update["first_music_page"]
        if "movements" in update:
            manifest.metadata["score_prep_movement_count"] = len(update["movements"])
        manifest.metadata["score_prep_updated_at"] = datetime.now(UTC).isoformat()
        masterclasses.save(manifest)
        return prep

    @app.get("/masterclasses/{masterclass_id}/score-page/{page}")
    def get_score_page(
        masterclass_id: str,
        page: int,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> Response:
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        rel = f"reference/score_pages/page-{page:03d}.png"
        key = manifest.artifacts.get(rel)
        if not key or not storage.exists(key):
            raise HTTPException(status_code=404, detail="page image not found")
        return Response(content=storage.read_bytes(key), media_type="image/png")

    @app.get("/masterclasses/{masterclass_id}/lessons")
    def list_lessons(masterclass_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> list[dict]:
        class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
        lessons = []
        for session_id in class_manifest.lessons:
            try:
                lessons.append(store.load_by_id(ctx, session_id).to_json())
            except FileNotFoundError:
                continue
        return lessons

    @app.post("/masterclasses/{masterclass_id}/lessons/run")
    async def create_lesson_upload_and_enqueue(
        masterclass_id: str,
        background: BackgroundTasks,
        file: UploadFile = File(...),
        movement: str | None = Form(default=None),
        first_measure: int | None = Form(default=None),
        last_measure: int | None = Form(default=None),
        notes: str | None = Form(default=None),
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        try:
            class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc

        lesson_movement = _clean_optional(movement) or class_manifest.movement
        manifest = store.create(
            ctx,
            source_filename=Path(file.filename or "source.mp4").name,
            repertoire=class_manifest.piece_name,
            movement=lesson_movement,
            instrument=class_manifest.instrument,
            instrument_profile=class_manifest.instrument_profile,
            notes=_clean_optional(notes),
        )
        filename = Path(file.filename or manifest.source_filename or "source.mp4").name
        key = store.artifact_key(manifest.session, f"input/{filename}")
        content = await file.read()
        storage.write_bytes(key, content, content_type=file.content_type or "video/mp4")
        manifest.artifacts["input/source_video"] = key
        manifest.metadata["source_size_bytes"] = len(content)
        manifest.metadata["source_uploaded_at"] = datetime.now(UTC).isoformat()
        manifest.metadata["masterclass_id"] = class_manifest.masterclass.masterclass_id
        manifest.metadata["lesson_number"] = len(class_manifest.lessons) + 1
        manifest.metadata["piece_name"] = class_manifest.piece_name
        if first_measure is not None:
            manifest.metadata["first_measure"] = first_measure
        if last_measure is not None:
            manifest.metadata["last_measure"] = last_measure
        for artifact_name, artifact_key in class_manifest.artifacts.items():
            manifest.artifacts[f"masterclass/{artifact_name}"] = artifact_key

        prior_context = _build_prior_lesson_context(class_manifest, manifest.session.session_id)
        prior_context_key = store.artifact_key(manifest.session, "context/prior_lessons.json")
        storage.write_json(prior_context_key, prior_context)
        manifest.artifacts["context/prior_lessons.json"] = prior_context_key
        manifest.metadata["prior_lesson_count"] = len(prior_context["lessons"])
        manifest.state = JobState.UPLOADED
        store.save(manifest)
        masterclasses.add_lesson(class_manifest, manifest.session.session_id)

        queued = [
            jobs.enqueue(manifest.session, QueuedJobType.EXTRACT_MEDIA).to_json(),
            jobs.enqueue(manifest.session, QueuedJobType.ANALYZE).to_json(),
            jobs.enqueue(manifest.session, QueuedJobType.EVIDENCE_PACKET).to_json(),
        ]
        _spawn(
            _run_lesson_jobs,
            manifest.session.session_id,
            manifest.session.tenant_id,
            manifest.session.user_id,
            class_manifest.masterclass.masterclass_id,
        )
        return {"masterclass": class_manifest.to_json(), "session": manifest.to_json(), "jobs": queued}

    @app.post("/sessions")
    def create_session(body: CreateSessionRequest, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        manifest = store.create(
            ctx,
            source_filename=body.source_filename,
            repertoire=body.repertoire,
            movement=body.movement,
            instrument=body.instrument,
            instrument_profile=body.instrument_profile,
            notes=body.notes,
        )
        return manifest.to_json()

    @app.get("/sessions")
    def list_sessions(ctx: TenantContext = Depends(tenant_from_header)) -> list[dict]:
        return [manifest.to_json() for manifest in store.list_for_user(ctx)]

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            return store.load_by_id(ctx, session_id).to_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @app.get("/sessions/{session_id}/artifact")
    def get_session_artifact(
        session_id: str,
        key: str,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> Response:
        manifest = store.load_by_id(ctx, session_id)
        artifact_key = manifest.artifacts.get(key)
        if not artifact_key or not storage.exists(artifact_key):
            raise HTTPException(status_code=404, detail="artifact not found")
        media_type = "application/octet-stream"
        lower = key.lower()
        if lower.endswith(".json"):
            media_type = "application/json"
        elif lower.endswith(".md") or lower.endswith(".txt"):
            media_type = "text/markdown" if lower.endswith(".md") else "text/plain"
        elif lower.endswith(".png"):
            media_type = "image/png"
        elif lower.endswith(".wav"):
            media_type = "audio/wav"
        elif lower.endswith(".mp4"):
            media_type = "video/mp4"
        return Response(content=storage.read_bytes(artifact_key), media_type=media_type)

    _SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")

    _MXL_DOCTYPE_RE = re.compile(rb"<!DOCTYPE[^>]*>")

    def _strip_external_dtd(xml_bytes: bytes) -> bytes:
        """Remove the <!DOCTYPE ...> declaration so browser XML parsers
        (used by OpenSheetMusicDisplay) don't try to fetch the external
        MusicXML DTD and reject the document.
        """
        return _MXL_DOCTYPE_RE.sub(b"", xml_bytes, count=1)

    def _player_html_response(session_id: str) -> HTMLResponse:
        # Strict allowlist: session IDs are uuid4().hex (32 lowercase hex chars).
        # Without this, a crafted path like /lessons/";alert(1);//abc/player would
        # be injected verbatim into the JS string `const SESSION_ID = "..."`,
        # giving a reflected XSS (caught by CodeQL py/reflective-xss).
        if not _SESSION_ID_RE.match(session_id or ""):
            raise HTTPException(status_code=400, detail="invalid session id")
        html = (static_dir / "player.html").read_text(encoding="utf-8")
        # Defense in depth: JSON-encode the literal so even if the regex above
        # is ever loosened, the substitution stays a safe JS string literal.
        return HTMLResponse(
            html.replace('"__SESSION_ID__"', json.dumps(session_id)),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.get("/lessons/{session_id}/player", response_class=HTMLResponse)
    def lesson_player(session_id: str) -> HTMLResponse:
        return _player_html_response(session_id)

    @app.get("/static/player.html", response_class=HTMLResponse)
    def static_player(session: str | None = None) -> HTMLResponse:
        if not session:
            raise HTTPException(status_code=400, detail="missing session")
        return _player_html_response(session)

    @app.get("/lessons/{session_id}/manifest")
    def lesson_manifest(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        masterclass_id = manifest.metadata.get("masterclass_id")
        masterclass_doc = None
        if masterclass_id:
            try:
                masterclass_doc = masterclasses.load_by_id(ctx, masterclass_id).to_json()
            except FileNotFoundError:
                masterclass_doc = None
        return {"session": manifest.to_json(), "masterclass": masterclass_doc}

    @app.get("/lessons/{session_id}/comments")
    def lesson_comments(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        key = manifest.artifacts.get("lesson/comments_enriched.json")
        if not key or not storage.exists(key):
            payload: dict[str, Any] = {"comments": [], "summary": "", "progress_notes": "", "_meta": {}}
        else:
            payload = storage.read_json(key)

        # Prefer the audio-truth-derived per-measure alignment over the
        # chroma DTW fallback over the teacher's listening estimates. The
        # ``analysis/hmm_alignment.json`` artifact is the legacy-shim file
        # written by ``audio_truth._build_legacy_hmm_artifacts`` and will
        # be renamed in a follow-up pass. Re-anchor each comment's start
        # time to the alignment-derived measure start.
        align_doc: dict[str, Any] | None = None
        align_source = "llm_estimate"
        for candidate_key, candidate_source in (
            ("analysis/hmm_alignment.json", "audio_truth_shim"),
            ("analysis/alignment.json", "chroma_dtw"),
        ):
            key2 = manifest.artifacts.get(candidate_key)
            if key2 and storage.exists(key2):
                try:
                    doc = storage.read_json(key2)
                except (FileNotFoundError, ValueError):
                    continue
                if doc and doc.get("measure_timestamps"):
                    align_doc = doc
                    align_source = candidate_source
                    break
        if align_doc and align_doc.get("measure_timestamps"):
            payload["measure_timestamps"] = align_doc["measure_timestamps"]
            payload.setdefault("_meta", {})["measure_timestamps_source"] = align_source
            mt_index = {int(e["measure"]): float(e["start"]) for e in align_doc["measure_timestamps"] if "measure" in e and "start" in e}
            for c in payload.get("comments") or []:
                m = c.get("measure")
                if isinstance(m, int) and m in mt_index:
                    aligned_start = mt_index[m]
                    # Optionally nudge by beat within the measure if both this
                    # measure and the next one are in the alignment table.
                    next_start = mt_index.get(m + 1)
                    beat = (c.get("references") or [{}])[0].get("beat") if c.get("references") else None
                    if isinstance(beat, (int, float)) and next_start is not None and beat > 1:
                        beat_frac = max(0.0, min((beat - 1) / 4.0, 0.95))
                        aligned_start = aligned_start + (next_start - aligned_start) * beat_frac
                    c["original_llm_start"] = c.get("start")
                    c["start"] = round(aligned_start, 3)
                    if c.get("end") and c.get("original_llm_start") is not None:
                        # Preserve the original duration if we have one.
                        try:
                            duration = max(1.0, float(c["end"]) - float(c["original_llm_start"]))
                            c["end"] = round(aligned_start + duration, 3)
                        except (TypeError, ValueError):
                            c["end"] = round(aligned_start + 3.0, 3)
                    else:
                        c["end"] = round(aligned_start + 3.0, 3)
            # Re-sort by start
            payload["comments"] = sorted(payload.get("comments") or [], key=lambda c: c.get("start") or 0)
        else:
            payload.setdefault("_meta", {})["measure_timestamps_source"] = "llm_estimate"

        # Augment references that are missing page/system with score-prep layout.
        # Do not overwrite explicit LLM references: in multi-movement works measure
        # numbers can restart, so a measure-only lookup against the full score can
        # incorrectly move Adagio comments to a later movement with the same bar.
        masterclass_id = manifest.metadata.get("masterclass_id")
        layout_by_measure: dict[int, dict[str, Any]] = {}
        if masterclass_id:
            try:
                class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
                prep_key = class_manifest.artifacts.get("reference/score_prep.json")
                if prep_key and storage.exists(prep_key):
                    prep = storage.read_json(prep_key) or {}
                    for page in (prep.get("pages") or []):
                        if not isinstance(page, dict) or page.get("kind") != "music":
                            continue
                        for sys in (page.get("systems") or []):
                            if not isinstance(sys, dict):
                                continue
                            f = sys.get("first_measure")
                            l = sys.get("last_measure")
                            if isinstance(f, int) and isinstance(l, int):
                                for m in range(f, l + 1):
                                    layout_by_measure[m] = {
                                        "page": page.get("page"),
                                        "system_index": sys.get("system_index"),
                                        "bbox": sys.get("bbox"),
                                    }
            except FileNotFoundError:
                pass
        if layout_by_measure:
            for c in payload.get("comments") or []:
                for ref in c.get("references") or []:
                    if ref.get("page") is not None and ref.get("system_index") is not None:
                        continue
                    m = ref.get("measure")
                    if isinstance(m, int) and m in layout_by_measure:
                        ref["page"] = layout_by_measure[m]["page"]
                        ref["system_index"] = layout_by_measure[m]["system_index"]
        return payload

    @app.get("/lessons/{session_id}/video")
    def lesson_video(session_id: str, ctx: TenantContext = Depends(tenant_from_header)):
        from fastapi.responses import FileResponse
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        video_key = manifest.artifacts.get("input/source_video")
        if not video_key:
            raise HTTPException(status_code=404, detail="lesson has no source video")
        if not isinstance(storage, LocalObjectStorage):
            raise HTTPException(status_code=501, detail="streaming video requires local storage backend in this build")
        path = storage.resolve_local_path(video_key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="video file missing")
        return FileResponse(str(path), media_type="video/mp4")

    @app.get("/lessons/{session_id}/score-page/{page}")
    def lesson_score_page(
        session_id: str,
        page: int,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> Response:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        masterclass_id = manifest.metadata.get("masterclass_id")
        if not masterclass_id:
            raise HTTPException(status_code=404, detail="lesson is not linked to a masterclass")
        try:
            class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        rel = f"reference/score_pages/page-{page:03d}.png"
        key = class_manifest.artifacts.get(rel)
        if not key or not storage.exists(key):
            raise HTTPException(status_code=404, detail="page not found")
        return Response(content=storage.read_bytes(key), media_type="image/png")

    @app.get("/lessons/{session_id}/score-prep")
    def lesson_score_prep(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        masterclass_id = manifest.metadata.get("masterclass_id")
        if not masterclass_id:
            return {}
        try:
            class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError:
            return {}
        prep_key = class_manifest.artifacts.get("reference/score_prep.json")
        if not prep_key or not storage.exists(prep_key):
            return {}
        return storage.read_json(prep_key)

    @app.get("/lessons/{session_id}/score-map")
    def lesson_score_map(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        key = manifest.artifacts.get("score/score_map.json")
        if key and storage.exists(key):
            return storage.read_json(key)
        fallback_key = store.artifact_key(manifest.session, "score/score_map.json")
        if storage.exists(fallback_key):
            return storage.read_json(fallback_key)
        return {}

    @app.get("/lessons/{session_id}/tool-calls")
    def lesson_tool_calls(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        key = manifest.artifacts.get("analysis/teach_tool_calls.json")
        if key and storage.exists(key):
            payload = storage.read_json(key)
            return payload if isinstance(payload, dict) else {"tool_calls": payload}
        fallback_key = store.artifact_key(manifest.session, "analysis/teach_tool_calls.json")
        if storage.exists(fallback_key):
            payload = storage.read_json(fallback_key)
            return payload if isinstance(payload, dict) else {"tool_calls": payload}
        calls: list[dict[str, Any]] = []
        for usage in manifest.llm_usage or []:
            if isinstance(usage, dict) and isinstance(usage.get("tool_calls"), list):
                calls.extend(usage["tool_calls"])
        return {"tool_calls": calls}

    # ------------------------------------------------------------------
    # Technical Viewer (Pro mode) - raw evidence inspection for power users
    # ------------------------------------------------------------------
    def _pro_ctx(ctx: TenantContext = Depends(tenant_from_header)) -> TenantContext:
        """Gate that requires the caller to have pro_mode=true.

        This is the single chokepoint that future billing checks plug into:
        flip pro_mode based on entitlement and every /debug/* route + the
        viewer page light up automatically.
        """
        try:
            profile = user_profiles.load(ctx.user_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=401, detail="Sign in with Google") from exc
        if not getattr(profile, "pro_mode", False):
            raise HTTPException(
                status_code=403,
                detail="Technical viewer requires Pro mode (enable in Settings).",
            )
        return ctx

    def _load_lesson_artifact(ctx: TenantContext, session_id: str, rel: str):
        """Read a per-lesson artifact via the same dual-path lookup other routes use."""
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        key = manifest.artifacts.get(rel)
        if key and storage.exists(key):
            return manifest, key
        fallback_key = store.artifact_key(manifest.session, rel)
        if storage.exists(fallback_key):
            return manifest, fallback_key
        return manifest, None

    @app.get("/lessons/{session_id}/technical-viewer", response_class=HTMLResponse)
    def lesson_technical_viewer(
        session_id: str,
        ctx: TenantContext = Depends(_pro_ctx),
    ) -> HTMLResponse:
        # Reuse the same uuid4-hex allowlist that protects the player route from
        # reflected XSS via the session_id path segment.
        if not _SESSION_ID_RE.match(session_id or ""):
            raise HTTPException(status_code=400, detail="invalid session id")
        # Confirm the caller actually owns the lesson; load_by_id raises 404 if not.
        store.load_by_id(ctx, session_id)
        html = (static_dir / "technical_viewer.html").read_text(encoding="utf-8")
        return HTMLResponse(
            html.replace('"__SESSION_ID__"', json.dumps(session_id)),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.get("/lessons/{session_id}/debug/musicxml")
    def debug_musicxml(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> Response:
        """Serve the masterclass's reference MusicXML for this lesson.

        Used by the technical viewer's MusicXML panel (OpenSheetMusicDisplay).
        Always returns uncompressed XML (text/xml) — if the artifact is
        .mxl (zipped), we extract the inner score.xml on the fly so the
        browser-side renderer doesn't have to deal with two formats.
        """
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        masterclass_id = manifest.metadata.get("masterclass_id")
        if not masterclass_id:
            raise HTTPException(status_code=404, detail="lesson has no masterclass binding")
        try:
            mc = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc

        for rel in ("reference/musicxml.musicxml", "reference/musicxml", "reference/musicxml.mxl"):
            key = mc.artifacts.get(rel)
            if not key or not storage.exists(key):
                continue
            raw = storage.read_bytes(key)
            # .mxl is a ZIP with a META-INF/container.xml pointing at the
            # main score file (usually score.xml or musicxml.xml).
            if rel.endswith(".mxl") or raw[:2] == b"PK":
                import zipfile, io as _io
                from xml.etree import ElementTree as _ET
                try:
                    with zipfile.ZipFile(_io.BytesIO(raw)) as zf:
                        rootfile = None
                        if "META-INF/container.xml" in zf.namelist():
                            container = _ET.fromstring(zf.read("META-INF/container.xml"))
                            ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                            node = container.find(".//c:rootfile", ns) or container.find(".//rootfile")
                            if node is not None:
                                rootfile = node.get("full-path")
                        if not rootfile:
                            # Heuristic fallback: pick the first .xml that isn't in META-INF/
                            rootfile = next(
                                (n for n in zf.namelist() if n.lower().endswith(".xml") and not n.startswith("META-INF/")),
                                None,
                            )
                        if not rootfile:
                            raise HTTPException(status_code=500, detail=".mxl has no readable score xml")
                        xml_bytes = zf.read(rootfile)
                except zipfile.BadZipFile as exc:
                    raise HTTPException(status_code=500, detail=f"malformed .mxl: {exc}") from exc
                xml_bytes = _strip_external_dtd(xml_bytes)
                return Response(
                    content=xml_bytes,
                    media_type="application/xml; charset=utf-8",
                    headers={"Cache-Control": "private, max-age=300"},
                )
            return Response(
                content=_strip_external_dtd(raw),
                media_type="application/xml; charset=utf-8",
                headers={"Cache-Control": "private, max-age=300"},
            )
        raise HTTPException(status_code=404, detail="no MusicXML artifact for this masterclass")

    @app.get("/lessons/{session_id}/debug/comments")
    def debug_comments(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        for rel in ("lesson/comments_enriched.json", "lesson/comments.json"):
            _, key = _load_lesson_artifact(ctx, session_id, rel)
            if key:
                payload = storage.read_json(key)
                if isinstance(payload, dict):
                    return payload
        raise HTTPException(status_code=404, detail="no comments artifact for this lesson")

    @app.get("/lessons/{session_id}/debug/evidence-packet")
    def debug_evidence_packet(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> Response:
        _, key = _load_lesson_artifact(ctx, session_id, "analysis/evidence_packet.md")
        if not key:
            raise HTTPException(status_code=404, detail="no evidence packet for this lesson")
        return Response(content=storage.read_bytes(key), media_type="text/markdown; charset=utf-8")

    @app.get("/lessons/{session_id}/debug/teach-context")
    def debug_teach_context(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        """Return the complete text context that gets sent to the teacher LLM.

        Mirrors ``teach_lesson._build_user_contents`` but omits binary parts
        (audio bytes, score PNGs, video frame JPEGs). Used by the technical
        viewer to debug hallucinations: if the LLM cited a pitch that isn't
        in the score, or made a wrong measure claim, this view shows the
        exact prose the model was looking at.
        """
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc

        from masterclass.agent_tools.catalog import tool_catalog_text
        from masterclass.engine.instruments import (
            load_instrument_profile,
            system_instruction_for_profile,
        )
        from masterclass.engine.prompt_evidence import build_evidence_digest
        from masterclass.engine.prompt_inventory import build_score_note_inventory
        from masterclass.engine.teach_lesson import (
            _frame_keys,
            _read_prior_context,
            _read_score_map,
            _score_image_keys,
            TeachConfig,
        )

        profile = load_instrument_profile(manifest.instrument_profile)
        system_instruction = system_instruction_for_profile(profile, tool_catalog=tool_catalog_text(profile))

        score_map = _read_score_map(storage, store, manifest)
        score_key = score_map.get("key") if score_map else None
        first_measure = manifest.metadata.get("first_measure") if manifest.metadata else None
        last_measure = manifest.metadata.get("last_measure") if manifest.metadata else None
        try:
            evidence_digest = build_evidence_digest(storage=storage, store=store, manifest=manifest, score_key=score_key)
        except Exception as exc:
            evidence_digest = f"(evidence digest unavailable: {exc})"
        if score_key:
            evidence_digest = (
                f"Key: {score_key}. Use this key's spelling in all comments; copy note spellings from the inventory.\n\n"
                + evidence_digest
            )
        try:
            inventory = build_score_note_inventory(score_map, first_measure=first_measure, last_measure=last_measure) if score_map else ""
        except Exception as exc:
            inventory = f"(inventory unavailable: {exc})"
        try:
            prior_context = _read_prior_context(storage, manifest)
        except Exception as exc:
            prior_context = f"(prior context unavailable: {exc})"

        # Audio key — surfaced as a label/placeholder only, never the bytes.
        audio_key = manifest.artifacts.get("artifacts/audio_16k.wav") or manifest.artifacts.get("artifacts/audio.wav")
        cfg = TeachConfig()
        try:
            score_image_keys = _score_image_keys(storage, store, manifest, score_map)
        except Exception:
            score_image_keys = []
        try:
            frame_keys_list = _frame_keys(storage, manifest, cfg.max_video_frames)
        except Exception:
            frame_keys_list = []

        # Mirror the order of _build_user_contents exactly so what the viewer
        # shows = what the model sees (modulo the binary blobs).
        parts: list[dict[str, Any]] = []
        parts.append({"kind": "system", "label": "System instruction", "text": system_instruction})
        parts.append({"kind": "text", "label": "Evidence digest", "text": evidence_digest or "(empty)"})
        parts.append({
            "kind": "text",
            "label": "Recording briefing",
            "text": (
                f"Repertoire: {manifest.repertoire or '(unknown)'}\n"
                f"Movement: {manifest.movement or '(unknown)'}\n"
                f"Instrument: {manifest.instrument or manifest.instrument_profile or '(unspecified)'}\n"
                f"Measures: {first_measure}\u2013{last_measure}\n"
                f"Student notes: {(manifest.notes or '(none)').strip()}\n"
            ),
        })
        parts.append({"kind": "text", "label": "Prior takes of this piece", "text": prior_context or "(no prior lessons)"})
        if audio_key:
            parts.append({
                "kind": "binary-omitted",
                "label": "Audio (audio/wav)",
                "text": f"[binary omitted] artifact_key={audio_key}",
            })
        if score_image_keys:
            parts.append({
                "kind": "binary-omitted",
                "label": f"Score images ({min(len(score_image_keys), cfg.max_score_pages)} of {len(score_image_keys)})",
                "text": "\n".join(
                    f"[binary omitted] score image {i + 1}: {k}"
                    for i, k in enumerate(score_image_keys[: cfg.max_score_pages])
                ),
            })
        else:
            parts.append({"kind": "binary-omitted", "label": "Score images (0)", "text": "(no score images available)"})
        if frame_keys_list:
            parts.append({
                "kind": "binary-omitted",
                "label": f"Video frames ({len(frame_keys_list)})",
                "text": "\n".join(
                    f"[binary omitted] video frame {i + 1}: {k}"
                    for i, k in enumerate(frame_keys_list)
                ),
            })
        else:
            parts.append({"kind": "binary-omitted", "label": "Video frames (0)", "text": "(no frames available)"})
        parts.append({"kind": "text", "label": "Score note inventory", "text": inventory or "(no score note inventory available)"})
        parts.append({
            "kind": "text",
            "label": "Task prompt",
            "text": (
                "Listen to the recording. Examine the score. Examine the video frames above. "
                "Use investigation tools to fact-check measurable claims. "
                "Produce the final v2 comments_enriched.json as a single JSON code block."
            ),
        })

        total_chars = sum(len(p["text"]) for p in parts)
        return {
            "session_id": session_id,
            "model": os.environ.get("MASTERCLASS_TEACH_MODEL", "gemini-2.5-pro"),
            "total_chars": total_chars,
            "parts": parts,
        }

    @app.get("/lessons/{session_id}/debug/raw-llm-response")
    def debug_raw_llm_response(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        _, key = _load_lesson_artifact(ctx, session_id, "llm/raw_teacher_response.json")
        if not key:
            raise HTTPException(status_code=404, detail="no raw LLM response for this lesson")
        payload = storage.read_json(key)
        if not isinstance(payload, dict):
            return {"raw": payload}
        return payload

    @app.get("/lessons/{session_id}/debug/analysis")
    def debug_analysis(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        _, key = _load_lesson_artifact(ctx, session_id, "analysis/analysis.json")
        if not key:
            raise HTTPException(status_code=404, detail="no analysis artifact for this lesson")
        payload = storage.read_json(key)
        return payload if isinstance(payload, dict) else {"analysis": payload}

    @app.get("/lessons/{session_id}/debug/aligned-notes")
    def debug_aligned_notes(
        session_id: str,
        source: str = "audio_truth_matched",
        start: float | None = None,
        end: float | None = None,
        ctx: TenantContext = Depends(_pro_ctx),
    ) -> dict:
        """Unified alignment endpoint: serve audio-truth / DTW / basic-pitch notes.

        ``source`` chooses the alignment artifact:
          - ``audio_truth_matched`` -> analysis/audio_truth_matched_notes.json
                              (canonical: transcriber + score matcher; default)
          - ``audio_truth`` -> analysis/audio_truth_notes.json (raw transcriber)
          - ``aligned``     -> analysis/aligned_notes.json    (legacy shim, new name)
          - ``hmm``         -> analysis/hmm_aligned_notes.json (legacy shim, deprecated name)
          - ``dtw``         -> analysis/dtw_aligned_notes.json (chroma DTW; tooling)
          - ``basic_pitch`` -> analysis/basic_pitch_notes.json (audio-only fallback)

        Notes are also filtered to the lesson's PlayedRange: rows whose
        ``measure`` falls outside ``[first_measure, last_measure]`` are
        dropped, with a ``played_range`` block echoed on the response.
        """
        src = (source or "audio_truth_matched").lower().strip()
        artifact_by_src = {
            "audio_truth_matched": "analysis/audio_truth_matched_notes.json",
            "audio_truth": "analysis/audio_truth_notes.json",
            "aligned": "analysis/aligned_notes.json",
            "hmm": "analysis/hmm_aligned_notes.json",
            "dtw": "analysis/dtw_aligned_notes.json",
            "basic_pitch": "analysis/basic_pitch_notes.json",
            "basic_pitch_matched": "analysis/basic_pitch_matched_notes.json",
            "piano_transcription": "analysis/piano_transcription_notes.json",
            "piano_transcription_matched": "analysis/piano_transcription_matched_notes.json",
        }
        if src not in artifact_by_src:
            raise HTTPException(status_code=400, detail=f"unknown alignment source: {source}")
        manifest, key = _load_lesson_artifact(ctx, session_id, artifact_by_src[src])
        if not key:
            raise HTTPException(status_code=404, detail=f"no {src} aligned notes for this lesson")
        payload = storage.read_json(key)
        notes = payload.get("notes") if isinstance(payload, dict) else None
        if not isinstance(notes, list):
            return payload if isinstance(payload, dict) else {"notes": [], "source": src}

        # Scope by played-range: rows from measures outside the lesson's
        # played window are noise (they only exist because the matcher
        # snapped detected onsets to nearby score events). Drop them here
        # so the UI never has to filter again.
        from masterclass.core.played_range import derive_played_range
        masterclass_id = manifest.metadata.get("masterclass_id")
        mc_manifest = None
        if masterclass_id:
            try:
                mc_manifest = masterclasses.load_by_id(ctx, masterclass_id)
            except FileNotFoundError:
                mc_manifest = None
        played_range = derive_played_range(manifest, mc_manifest)
        in_range_notes: list[dict[str, Any]] = []
        dropped_out_of_range = 0
        for n in notes:
            if not isinstance(n, dict):
                continue
            m = n.get("measure")
            if m is None or played_range.contains(m):
                in_range_notes.append(n)
            else:
                dropped_out_of_range += 1
        notes = in_range_notes

        if start is not None or end is not None:
            lo = start if start is not None else float("-inf")
            hi = end if end is not None else float("inf")
            notes = [
                n for n in notes
                if isinstance(n, dict)
                and lo <= float(n.get("performed_time_sec", n.get("perf_time", 0.0)) or 0.0) <= hi
            ]
        return {
            "notes": notes,
            "source": src,
            "method": payload.get("method") if isinstance(payload, dict) else None,
            "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
            "played_range": {
                "first_measure": played_range.first_measure,
                "last_measure": played_range.last_measure,
                "source": played_range.source,
            },
            "dropped_out_of_range": dropped_out_of_range,
        }

    @app.get("/lessons/{session_id}/debug/hmm-aligned-notes")
    def debug_hmm_notes(
        session_id: str,
        start: float | None = None,
        end: float | None = None,
        ctx: TenantContext = Depends(_pro_ctx),
    ) -> dict:
        # Deprecated alias: forwards to the unified endpoint with
        # ``source="hmm"`` (the legacy-shim artifact). Kept so external
        # tooling that still hits the old URL keeps working until the
        # shim itself is deleted.
        return debug_aligned_notes(
            session_id=session_id, source="hmm", start=start, end=end, ctx=ctx,
        )

    @app.get("/lessons/{session_id}/debug/pitch-events")
    def debug_pitch_events(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        _, key = _load_lesson_artifact(ctx, session_id, "analysis/pitch_events.json")
        if not key:
            raise HTTPException(status_code=404, detail="no pitch events for this lesson")
        payload = storage.read_json(key)
        return payload if isinstance(payload, dict) else {"events": payload}

    @app.get("/lessons/{session_id}/debug/mechanical-comments")
    def debug_mechanical_comments(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        _, key = _load_lesson_artifact(ctx, session_id, "analysis/mechanical_comments.json")
        if not key:
            raise HTTPException(status_code=404, detail="no mechanical comments for this lesson")
        payload = storage.read_json(key)
        return payload if isinstance(payload, dict) else {"comments": payload}

    @app.get("/lessons/{session_id}/debug/spectrogram")
    def debug_spectrogram(
        session_id: str,
        start: float,
        end: float,
        w: int = 1100,
        h: int = 520,
        ctx: TenantContext = Depends(_pro_ctx),
    ) -> Response:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        audio_key = manifest.artifacts.get("artifacts/audio_16k.wav") or manifest.artifacts.get("artifacts/audio.wav")
        if not audio_key:
            audio_key = store.artifact_key(manifest.session, "artifacts/audio_16k.wav")
            if not storage.exists(audio_key):
                audio_key = store.artifact_key(manifest.session, "artifacts/audio.wav")
                if not storage.exists(audio_key):
                    raise HTTPException(status_code=404, detail="lesson has no decoded audio")
        try:
            png, meta = render_spectrogram_window(
                storage=storage,
                audio_key=audio_key,
                start_sec=float(start),
                end_sec=float(end),
                width=int(w),
                height=int(h),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Headers let the frontend translate mouse pixel coords -> data coords
        # without guessing matplotlib's layout. Exposed via Access-Control so
        # any future cross-origin proxy can still read them.
        plot_bbox = meta["plot_bbox_px"]
        midi_range = meta["midi_range"]
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-Spec-Plot-Bbox": f"{plot_bbox[0]:.2f},{plot_bbox[1]:.2f},{plot_bbox[2]:.2f},{plot_bbox[3]:.2f}",
                "X-Spec-Image-Size": f"{meta['image_width']},{meta['image_height']}",
                "X-Spec-Time-Range": f"{meta['time_range_sec'][0]:.6f},{meta['time_range_sec'][1]:.6f}",
                "X-Spec-Midi-Range": f"{midi_range[0]:.4f},{midi_range[1]:.4f}",
                "Access-Control-Expose-Headers": "X-Spec-Plot-Bbox,X-Spec-Image-Size,X-Spec-Time-Range,X-Spec-Midi-Range",
            },
        )

    _WATCH_CLIP_RE = re.compile(r"^clip_\d+_\d+_h\d+\.mp4$")

    @app.get("/lessons/{session_id}/debug/watch-clip/{clip_name}")
    def debug_watch_clip(
        session_id: str,
        clip_name: str,
        ctx: TenantContext = Depends(_pro_ctx),
    ) -> Response:
        # Strict filename allowlist: prevents path-traversal via the clip_name segment.
        if not _WATCH_CLIP_RE.match(clip_name or ""):
            raise HTTPException(status_code=400, detail="invalid clip filename")
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        key = store.artifact_key(manifest.session, f"watch_clips/{clip_name}")
        if not storage.exists(key):
            raise HTTPException(status_code=404, detail="watch clip not found")
        return Response(
            content=storage.read_bytes(key),
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/lessons/{session_id}/debug/watch-clips")
    def debug_list_watch_clips(session_id: str, ctx: TenantContext = Depends(_pro_ctx)) -> dict:
        """List the watch-clip filenames Gemini was given for this lesson."""
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        if not isinstance(storage, LocalObjectStorage):
            return {"clips": [], "note": "listing only supported on local storage"}
        prefix = store.artifact_key(manifest.session, "watch_clips")
        local_dir = storage.resolve_local_path(prefix)
        if not local_dir.exists() or not local_dir.is_dir():
            return {"clips": []}
        clips = []
        for p in sorted(local_dir.iterdir()):
            if not _WATCH_CLIP_RE.match(p.name):
                continue
            stem = p.stem  # clip_0009000_0010000_h480
            parts = stem.split("_")
            start_ms = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else None
            end_ms = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            clips.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "start_sec": (start_ms / 1000.0) if start_ms is not None else None,
                "end_sec": (end_ms / 1000.0) if end_ms is not None else None,
            })
        return {"clips": clips}

    @app.post("/lessons/{session_id}/debug/rebuild-audio-truth")
    def debug_rebuild_audio_truth(
        session_id: str,
        ctx: TenantContext = Depends(_pro_ctx),
    ) -> dict:
        """Re-run audio-truth transcription + score-matching on an existing lesson.

        Useful for lessons created before the audio-truth pipeline existed
        (everything in the local store right now), or after re-tuning the
        transcriber. Runs synchronously since PTI/basic-pitch model load
        plus inference is on the order of 30-90s and the user is actively
        watching the technical viewer.
        """
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        try:
            summary = run_audio_truth_pipeline(storage=storage, store=store, manifest=manifest)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - surfaces upstream failures
            import logging as _logging
            _logging.getLogger(__name__).exception("rebuild-audio-truth failed for %s", session_id)
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return summary

    @app.post("/lessons/{session_id}/retry-failed")
    def retry_failed_stages(
        session_id: str,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        """Re-run ONLY the failed (or never-started) stages of a lesson.

        Differs from /rerun in that any stage whose ``{stage}_state`` is
        already ``ready`` is left untouched — its artifact is reused. This
        is the right action for transient errors like a Gemini File-API
        upload-terminated where the upstream stages all finished cleanly
        and we just need to retry the teacher. A 4-minute re-ingest is
        wasteful in that case.
        """
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        masterclass_id = manifest.metadata.get("masterclass_id")
        # Wipe failed-stage errors so the UI shows them as running again.
        cleared: list[str] = []
        for stage in (
            "extract_media", "analyze", "evidence_packet",
            "onsets", "audio_truth", "score_map",
            "intonation", "rhythm", "voicing",
            "mechanical_comments", "teach",
        ):
            state = manifest.metadata.get(f"{stage}_state")
            if state in ("failed", "running") or state is None:
                manifest.metadata[f"{stage}_state"] = "pending"
                manifest.metadata[f"{stage}_error"] = None
                cleared.append(stage)
        manifest.state = JobState.UPLOADED
        store.save(manifest)
        _spawn(
            _run_lesson_jobs,
            manifest.session.session_id,
            manifest.session.tenant_id,
            manifest.session.user_id,
            masterclass_id,
            resume=True,
        )
        return {"session_id": session_id, "retried_stages": cleared, "state": manifest.state.value}

    @app.post("/lessons/{session_id}/rerun")
    def rerun_lesson_pipeline(
        session_id: str,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        """Re-run the full lesson background pipeline for an existing session.

        Use after a code update (e.g. the audio-truth refactor) to re-process
        a lesson with the latest engine. Idempotent: each stage rebuilds its
        artifact. Spawns the same _run_lesson_jobs thread the upload flow
        spawns.
        """
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        masterclass_id = manifest.metadata.get("masterclass_id")
        # Reset stale "running" markers so the UI starts polling fresh.
        for stage_state in (
            "extract_media_state", "analyze_state", "evidence_packet_state",
            "onsets_state", "audio_truth_state", "score_map_state",
            "intonation_state", "rhythm_state", "voicing_state",
            "mechanical_comments_state", "teach_state",
        ):
            if manifest.metadata.get(stage_state) == "running":
                manifest.metadata[stage_state] = "requeued"
        manifest.state = JobState.UPLOADED
        store.save(manifest)
        _spawn(
            _run_lesson_jobs,
            manifest.session.session_id,
            manifest.session.tenant_id,
            manifest.session.user_id,
            masterclass_id,
        )
        return {"session_id": session_id, "requeued": True, "state": manifest.state.value}

    @app.post("/lessons/{session_id}/chat")
    def lesson_chat(session_id: str, body: ChatRequest = Body(...), ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            message = (body.message or "").strip()
            if not message:
                raise HTTPException(status_code=400, detail="message is required")
            check_message_size(message)
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
            if manifest.state != JobState.READY:
                raise HTTPException(status_code=409, detail="lesson is not ready for chat yet")
            if body.conversation_id:
                conversation = load_conversation(storage, store, manifest, body.conversation_id)
                check_conversation_turn_cap(conversation.user_message_count)
            elif body.comment_id:
                # Per-comment reply thread: derive a stable conversation id so
                # follow-ups under the same comment go to the same file.
                cmt_conv_id = _comment_conversation_id(body.comment_id)
                try:
                    conversation = load_conversation(storage, store, manifest, cmt_conv_id)
                    check_conversation_turn_cap(conversation.user_message_count)
                except FileNotFoundError:
                    check_conversation_turn_cap(0)
            else:
                check_conversation_turn_cap(0)
            check_user_quota(storage, ctx.tenant_id, ctx.user_id)
            # Use the caller's per-user BYO Gemini key (falls back to the server's
            # shared key only when ALLOW_SERVER_DEFAULT_KEY=true) so chat respects
            # the same billing/quota contract as lesson processing.
            provider = _build_llm_provider_for_user(ctx.user_id)
            topic = topic_guard(storage, ctx.tenant_id, ctx.user_id, message, provider)
            chat_model = os.environ.get(
                "MASTERCLASS_TEACH_MODEL",
                "gemini-2.5-pro",
            )
            effective_conv_id = body.conversation_id or (_comment_conversation_id(body.comment_id) if body.comment_id else None)
            result = run_chat_turn(
                storage=storage,
                store=store,
                manifest=manifest,
                provider=provider,
                message=message,
                conversation_id=effective_conv_id,
                comment_id=body.comment_id,
                masterclasses=masterclasses,
                config=ChatConfig(model=chat_model),
                topic_guard_usage=topic.usage,
            )
            return {
                "conversation_id": result.conversation_id,
                "reply": result.reply,
                "tool_calls": result.tool_calls,
                "usage": result.usage,
            }
        except ChatGuardrailError as exc:
            usage = exc.usage or chat_usage_dict(None)
            raise HTTPException(
                status_code=exc.status_code,
                detail={"reply": exc.detail, "usage": usage},
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson or conversation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            # Gemini retried-and-failed errors surface as RuntimeError from
            # agent/gemini.py:_send. Map peak-demand 503s to a friendly UI
            # message instead of an opaque 500.
            msg = str(exc)
            if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail={
                        "reply": (
                            "Gemini is overloaded right now (peak demand). "
                            "This usually clears within a few minutes — please try again shortly."
                        ),
                        "usage": chat_usage_dict(None),
                    },
                ) from exc
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                raise HTTPException(
                    status_code=429,
                    detail={
                        "reply": (
                            "Your Gemini API quota has been exceeded. "
                            "Wait a minute and try again, or check your billing tier at "
                            "https://aistudio.google.com/apikey."
                        ),
                        "usage": chat_usage_dict(None),
                    },
                ) from exc
            raise

    @app.get("/lessons/{session_id}/chat")
    def lesson_chat_list(session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> list[dict]:
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        return list_conversations(storage, store, manifest)

    @app.get("/lessons/{session_id}/chat/comment/{comment_id}")
    def lesson_chat_comment_history(
        session_id: str,
        comment_id: str,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        """Return the reply thread for one teacher comment, or an empty stub."""
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        conv_id = _comment_conversation_id(comment_id)
        try:
            return load_conversation(storage, store, manifest, conv_id).to_json()
        except FileNotFoundError:
            return {
                "conversation_id": conv_id,
                "comment_id": comment_id,
                "messages": [],
                "exists": False,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/lessons/{session_id}/chat/{conv_id}")
    def lesson_chat_history(session_id: str, conv_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
            return load_conversation(storage, store, manifest, conv_id).to_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson or conversation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/lessons/{session_id}/chat/{conv_id}")
    def lesson_chat_delete(session_id: str, conv_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = store.load_by_id(ctx, session_id)
            _require_lesson_kind(manifest)
            delete_conversation(storage, store, manifest, conv_id)
            return {"deleted": True, "conversation_id": conv_id}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson or conversation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/source")
    async def upload_source(
        session_id: str,
        file: UploadFile = File(...),
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        filename = Path(file.filename or manifest.source_filename or "source.mp4").name
        key = store.artifact_key(manifest.session, f"input/{filename}")
        content = await file.read()
        storage.write_bytes(key, content, content_type=file.content_type or "video/mp4")
        manifest.source_filename = filename
        manifest.artifacts["input/source_video"] = key
        manifest.metadata["source_size_bytes"] = len(content)
        manifest.metadata["source_uploaded_at"] = datetime.now(UTC).isoformat()
        manifest.state = JobState.UPLOADED
        store.save(manifest)
        return manifest.to_json()

    class EnqueueJobRequest(BaseModel):
        payload: dict = {}

    @app.post("/sessions/{session_id}/jobs/{job_type}")
    def enqueue_job(
        session_id: str,
        job_type: QueuedJobType,
        body: EnqueueJobRequest | None = None,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        manifest = store.load_by_id(ctx, session_id)
        _require_lesson_kind(manifest)
        job = jobs.enqueue(manifest.session, job_type, (body.payload if body else {}))
        return job.to_json()

    # ------------------------------------------------------------------
    # Drill (practice-clip) endpoints
    # ------------------------------------------------------------------
    _COMMENT_ID_SAFE_OUTER = _COMMENT_ID_SAFE

    def _read_lesson_comment(manifest, comment_id: str) -> dict | None:
        """Pull a comment dict out of the lesson's enriched comments artifact."""
        for rel in ("lesson/comments_enriched.json", "lesson/comments.json"):
            key = manifest.artifacts.get(rel) or store.artifact_key(manifest.session, rel)
            if not storage.exists(key):
                continue
            try:
                payload = storage.read_json(key)
            except (FileNotFoundError, ValueError):
                continue
            comments = (payload.get("comments") if isinstance(payload, dict) else None) or []
            for comment in comments:
                if isinstance(comment, dict) and str(comment.get("id") or "") == comment_id:
                    return comment
        return None

    def _run_drill_jobs(drill_session_id: str, tenant_id: str, user_id: str) -> None:
        """Background drill worker: build provider + ffmpeg, then run the pipeline."""
        import logging
        try:
            ctx = TenantContext(tenant_id=tenant_id, user_id=user_id)
            try:
                manifest = store.load_by_id(ctx, drill_session_id)
            except FileNotFoundError:
                return
            try:
                provider = _build_llm_provider_for_user(user_id)
            except HTTPException as exc:
                manifest.metadata["drill_state"] = "failed"
                manifest.metadata["drill_error"] = str(exc.detail)
                manifest.metadata["drill_feedback_state"] = "failed"
                manifest.metadata["drill_feedback_error"] = str(exc.detail)
                manifest.state = JobState.FAILED
                store.save(manifest)
                return
            from masterclass.engine.drill_pipeline import DrillConfig, run_drill_pipeline
            run_drill_pipeline(
                storage=storage,
                store=store,
                manifest=manifest,
                provider=provider,
                config=DrillConfig(model=os.environ.get("MASTERCLASS_DRILL_MODEL", "gemini-2.5-flash")),
            )
        except Exception:
            logging.exception("drill background job failed for %s", drill_session_id)

    def _create_drill_manifest(
        ctx: TenantContext,
        *,
        source_filename: str,
        source_bytes: bytes,
        source_content_type: str | None,
        drill_instruction: str,
        parent_session_id: str | None,
        parent_comment_id: str | None,
        parent_comment: dict | None,
        masterclass_id: str | None,
        instrument: str | None,
        instrument_profile: str | None,
    ):
        drill = store.create(
            ctx,
            source_filename=source_filename,
            repertoire=None,
            movement=None,
            instrument=instrument,
            instrument_profile=instrument_profile,
            notes=None,
        )
        drill.kind = SESSION_KIND_DRILL
        # Drill payload always lives under input/source_video so the
        # existing extract_media plumbing can read it without a special
        # case for audio uploads (ffmpeg handles either).
        key = store.artifact_key(drill.session, f"input/{source_filename}")
        storage.write_bytes(key, source_bytes, content_type=source_content_type or "application/octet-stream")
        drill.artifacts["input/source_video"] = key
        drill.metadata["source_size_bytes"] = len(source_bytes)
        drill.metadata["source_uploaded_at"] = datetime.now(UTC).isoformat()
        drill.metadata["drill_instruction"] = drill_instruction
        drill.metadata["drill_state"] = "uploaded"
        if parent_session_id:
            drill.metadata["parent_session_id"] = parent_session_id
        if parent_comment_id:
            drill.metadata["parent_comment_id"] = parent_comment_id
        if parent_comment:
            drill.metadata["parent_comment"] = {
                "id": parent_comment.get("id"),
                "measure": parent_comment.get("measure"),
                "category": parent_comment.get("category"),
                "severity": parent_comment.get("severity"),
                "summary": parent_comment.get("summary"),
                "text": parent_comment.get("text"),
            }
        if masterclass_id:
            drill.metadata["masterclass_id"] = masterclass_id
        drill.state = JobState.UPLOADED
        store.save(drill)
        return drill

    def _post_drill_upload_bubble(parent_manifest, conv_id: str, drill_session_id: str, instruction: str) -> None:
        """Append an optimistic 'uploading…' bubble to a comment thread."""
        from masterclass.core.chat_models import (
            ChatConversation,
            ChatMessage,
            conversation_key as _conv_key,
            load_conversation as _load_conv,
            save_conversation as _save_conv,
        )
        key = _conv_key(store, parent_manifest.session, conv_id)
        with conversation_lock(key):
            try:
                conv = _load_conv(storage, store, parent_manifest, conv_id)
            except FileNotFoundError:
                conv = ChatConversation(
                    conversation_id=conv_id,
                    session_id=parent_manifest.session.session_id,
                    user_id=parent_manifest.session.user_id,
                )
            conv.append(ChatMessage(
                role="system",
                content="📎 Practice clip uploaded — analysing…",
                metadata={
                    "type": "drill_upload",
                    "drill_session_id": drill_session_id,
                    "state": "processing",
                    "drill_instruction_excerpt": (instruction or "")[:200],
                },
            ))
            _save_conv(storage, store, parent_manifest, conv)

    @app.post("/lessons/{session_id}/comments/{comment_id}/practice-clip")
    async def upload_practice_clip_for_comment(
        session_id: str,
        comment_id: str,
        file: UploadFile = File(...),
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        # Validate parent integrity: lesson must exist, kind=lesson,
        # owned by caller, and the comment must exist.
        parent = _load_lesson_manifest(store, ctx, session_id)
        safe_comment_id = _COMMENT_ID_SAFE.sub("_", comment_id or "").strip("_")
        if not safe_comment_id:
            raise HTTPException(status_code=400, detail="comment_id is required")
        parent_comment = _read_lesson_comment(parent, comment_id)
        if parent_comment is None:
            raise HTTPException(status_code=404, detail=f"comment {comment_id} not found on this lesson")
        instruction = (parent_comment.get("text") or parent_comment.get("summary") or "").strip()
        if not instruction:
            raise HTTPException(status_code=400, detail="parent comment has no text to use as drill instruction")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file")
        filename = Path(file.filename or "drill.mp4").name

        drill = _create_drill_manifest(
            ctx,
            source_filename=filename,
            source_bytes=content,
            source_content_type=file.content_type,
            drill_instruction=instruction,
            parent_session_id=session_id,
            parent_comment_id=comment_id,
            parent_comment=parent_comment,
            masterclass_id=parent.metadata.get("masterclass_id"),
            instrument=parent.instrument,
            instrument_profile=parent.instrument_profile,
        )

        conv_id = _comment_conversation_id(comment_id)
        _post_drill_upload_bubble(parent, conv_id, drill.session.session_id, instruction)

        _spawn(
            _run_drill_jobs,
            drill.session.session_id,
            drill.session.tenant_id,
            drill.session.user_id,
        )
        return {
            "drill_session_id": drill.session.session_id,
            "conversation_id": conv_id,
            "parent_session_id": session_id,
            "parent_comment_id": comment_id,
        }

    @app.post("/masterclasses/{masterclass_id}/practice-clips")
    async def upload_practice_clip_for_masterclass(
        masterclass_id: str,
        file: UploadFile = File(...),
        drill_instruction: str | None = Form(default=None),
        parent_session_id: str | None = Form(default=None),
        parent_comment_id: str | None = Form(default=None),
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        try:
            mc = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc

        parent_manifest = None
        parent_comment = None
        if parent_session_id:
            try:
                parent_manifest = store.load_by_id(ctx, parent_session_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="parent lesson not found") from exc
            _require_lesson_kind(parent_manifest)
            if parent_manifest.metadata.get("masterclass_id") and parent_manifest.metadata["masterclass_id"] != masterclass_id:
                raise HTTPException(status_code=400, detail="parent lesson belongs to a different masterclass")
            if parent_comment_id:
                parent_comment = _read_lesson_comment(parent_manifest, parent_comment_id)
                if parent_comment is None:
                    raise HTTPException(status_code=404, detail=f"comment {parent_comment_id} not found on parent lesson")

        # Drill must have either an explicit instruction or a parent comment.
        instruction = (drill_instruction or "").strip()
        if not instruction and parent_comment:
            instruction = (parent_comment.get("text") or parent_comment.get("summary") or "").strip()
        if not instruction:
            raise HTTPException(
                status_code=400,
                detail="provide either drill_instruction text or parent_comment_id linking to an existing comment",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file")
        filename = Path(file.filename or "drill.mp4").name

        drill = _create_drill_manifest(
            ctx,
            source_filename=filename,
            source_bytes=content,
            source_content_type=file.content_type,
            drill_instruction=instruction,
            parent_session_id=parent_session_id,
            parent_comment_id=parent_comment_id,
            parent_comment=parent_comment,
            masterclass_id=masterclass_id,
            instrument=mc.instrument,
            instrument_profile=mc.instrument_profile,
        )

        # Track the drill id on the masterclass.
        drill_ids = list(mc.metadata.get("drill_session_ids") or [])
        drill_ids.append(drill.session.session_id)
        mc.metadata["drill_session_ids"] = drill_ids
        masterclasses.save(mc)

        # If linked to a parent comment thread, post the optimistic bubble too.
        if parent_manifest is not None and parent_comment_id:
            conv_id = _comment_conversation_id(parent_comment_id)
            _post_drill_upload_bubble(parent_manifest, conv_id, drill.session.session_id, instruction)

        _spawn(
            _run_drill_jobs,
            drill.session.session_id,
            drill.session.tenant_id,
            drill.session.user_id,
        )
        return {
            "drill_session_id": drill.session.session_id,
            "masterclass_id": masterclass_id,
            "parent_session_id": parent_session_id,
            "parent_comment_id": parent_comment_id,
        }

    def _load_drill_manifest(ctx: TenantContext, drill_session_id: str):
        try:
            manifest = store.load_by_id(ctx, drill_session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="drill not found") from exc
        _require_drill_kind(manifest)
        return manifest

    @app.get("/drills/{drill_session_id}")
    def get_drill(drill_session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        drill = _load_drill_manifest(ctx, drill_session_id)
        feedback_key = drill.artifacts.get("lesson/drill_feedback.md")
        feedback = None
        if feedback_key and storage.exists(feedback_key):
            try:
                feedback = storage.read_bytes(feedback_key).decode("utf-8")
            except Exception:
                feedback = None
        metrics_key = drill.artifacts.get("analysis/drill_metrics.json")
        metrics = None
        if metrics_key and storage.exists(metrics_key):
            try:
                metrics = storage.read_json(metrics_key)
            except Exception:
                metrics = None
        return {
            "session": drill.to_json(),
            "drill_state": drill.metadata.get("drill_state"),
            "drill_instruction": drill.metadata.get("drill_instruction"),
            "parent_session_id": drill.metadata.get("parent_session_id"),
            "parent_comment_id": drill.metadata.get("parent_comment_id"),
            "parent_comment": drill.metadata.get("parent_comment"),
            "masterclass_id": drill.metadata.get("masterclass_id"),
            "metrics": metrics,
            "feedback": feedback,
        }

    @app.get("/drills/{drill_session_id}/status")
    def get_drill_status(drill_session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        drill = _load_drill_manifest(ctx, drill_session_id)
        return {
            "drill_session_id": drill_session_id,
            "state": drill.metadata.get("drill_state") or "unknown",
            "error": drill.metadata.get("drill_error"),
            "stages": {
                stage: drill.metadata.get(f"{stage}_state")
                for stage in ("extract_media", "transcribe", "drill_metrics", "drill_feedback")
            },
            "updated_at": drill.updated_at,
        }

    @app.post("/drills/{drill_session_id}/retry")
    def retry_drill(drill_session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        drill = _load_drill_manifest(ctx, drill_session_id)
        # Clear failed-stage markers so the pipeline reruns them; ready
        # stages are preserved (resume mode in run_drill_pipeline).
        cleared: list[str] = []
        for stage in ("extract_media", "transcribe", "drill_metrics", "drill_feedback"):
            if (drill.metadata.get(f"{stage}_state") or "") in ("failed", "running"):
                drill.metadata[f"{stage}_state"] = "pending"
                drill.metadata[f"{stage}_error"] = None
                cleared.append(stage)
        drill.metadata["drill_state"] = "processing"
        drill.metadata["drill_error"] = None
        drill.state = JobState.ANALYZING
        store.save(drill)
        _spawn(
            _run_drill_jobs,
            drill.session.session_id,
            drill.session.tenant_id,
            drill.session.user_id,
        )
        return {"drill_session_id": drill_session_id, "retried_stages": cleared, "state": "processing"}

    @app.delete("/drills/{drill_session_id}")
    def delete_drill(drill_session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        drill = _load_drill_manifest(ctx, drill_session_id)
        # Best-effort: remove from masterclass.metadata.drill_session_ids.
        mc_id = drill.metadata.get("masterclass_id")
        if mc_id:
            try:
                mc = masterclasses.load_by_id(ctx, mc_id)
            except FileNotFoundError:
                mc = None
            if mc is not None:
                ids = list(mc.metadata.get("drill_session_ids") or [])
                if drill_session_id in ids:
                    ids.remove(drill_session_id)
                    mc.metadata["drill_session_ids"] = ids
                    masterclasses.save(mc)
        # Tombstone the manifest by writing a deleted marker. We don't
        # have a hard delete in SessionStore; flagging it suffices and
        # keeps audit trails.
        drill.metadata["drill_state"] = "deleted"
        drill.state = JobState.CANCELLED
        store.save(drill)
        return {"deleted": True, "drill_session_id": drill_session_id}

    @app.get("/masterclasses/{masterclass_id}/practice-clips")
    def list_practice_clips(masterclass_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> list[dict]:
        try:
            mc = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        out: list[dict] = []
        for drill_id in mc.metadata.get("drill_session_ids") or []:
            try:
                drill = store.load_by_id(ctx, drill_id)
            except FileNotFoundError:
                continue
            if drill.kind != SESSION_KIND_DRILL:
                continue
            if drill.state == JobState.CANCELLED or drill.metadata.get("drill_state") == "deleted":
                continue
            out.append({
                "drill_session_id": drill_id,
                "state": drill.metadata.get("drill_state"),
                "drill_instruction": drill.metadata.get("drill_instruction"),
                "parent_session_id": drill.metadata.get("parent_session_id"),
                "parent_comment_id": drill.metadata.get("parent_comment_id"),
                "created_at": drill.created_at,
                "updated_at": drill.updated_at,
            })
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    @app.get("/drills/{drill_session_id}/audio")
    def get_drill_audio(drill_session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> Response:
        drill = _load_drill_manifest(ctx, drill_session_id)
        key = drill.artifacts.get("artifacts/audio.wav") or drill.artifacts.get("input/source_video")
        if not key or not storage.exists(key):
            raise HTTPException(status_code=404, detail="no drill audio available")
        media_type = "audio/wav" if str(key).lower().endswith(".wav") else "application/octet-stream"
        return Response(content=storage.read_bytes(key), media_type=media_type)

    @app.get("/drills/{drill_session_id}/frame")
    def get_drill_frame(drill_session_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> Response:
        drill = _load_drill_manifest(ctx, drill_session_id)
        frames = drill.metadata.get("frames") or []
        if not frames:
            raise HTTPException(status_code=404, detail="no frames extracted for this drill")
        key = str(frames[0])
        if not storage.exists(key):
            raise HTTPException(status_code=404, detail="frame file missing")
        return Response(content=storage.read_bytes(key), media_type="image/jpeg")

    _DRILL_ID_RE = re.compile(r"^[0-9a-f]{32}$")

    @app.get("/drills/{drill_session_id}/page", response_class=HTMLResponse)
    def drill_page(drill_session_id: str) -> HTMLResponse:
        if not _DRILL_ID_RE.match(drill_session_id or ""):
            raise HTTPException(status_code=400, detail="invalid drill id")
        path = static_dir / "drill.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="drill page template not installed")
        html = path.read_text(encoding="utf-8")
        return HTMLResponse(
            html.replace('"__DRILL_SESSION_ID__"', json.dumps(drill_session_id)),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    return app
