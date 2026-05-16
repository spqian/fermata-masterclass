from __future__ import annotations

import os
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
from masterclass.core.jobs import JobStore, QueuedJobType
from masterclass.core.chat_models import delete_conversation, list_conversations, load_conversation
from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import JobState
from masterclass.core.models import TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.core.user_profiles import DEFAULT_MODEL, UserProfileStore
from masterclass.engine.score_prep import ScorePrepConfig, prepare_score, select_score_pages_for_lesson
from masterclass.engine.alignment import AlignmentConfig, align_lesson_with_midi, persist_alignment
from masterclass.engine.analysis import analyze_session, build_evidence_packet
from masterclass.engine.hmm_align import align_lesson_with_midi_hmm, persist_hmm_alignment
from masterclass.engine.ingest import extract_media_artifacts
from masterclass.engine.instruments import intonation_enabled_for_profile, load_instrument_profile
from masterclass.engine.intonation import analyze_intonation
from masterclass.engine.mechanical_comments import generate_mechanical_comments, persist_mechanical_comments
from masterclass.engine.midi_finder import MidiFinderConfig, auto_attach_midi_to_masterclass
from masterclass.engine.onsets import detect_rich_onsets
from masterclass.engine.rhythm import analyze_rhythm, persist_rhythm
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

    class ProfilePatch(BaseModel):
        gemini_api_key: str | None = None
        clear_gemini_key: bool = False
        preferred_model: str | None = None


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

    def _run_lesson_jobs(session_id: str, tenant_id: str, user_id: str, masterclass_id: str | None) -> None:
        """Drain a lesson's queued jobs sequentially inside this API process.

        MVP equivalent of an out-of-process worker: keeps the API responsive
        for upload, runs deterministic engine steps in the background, then
        runs the multimodal teacher, and updates session state so the UI can
        poll progress.
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

                mark_stage("extract_media", "running")
                manifest = extract_media_artifacts(store=store, storage=storage, ffmpeg=ffmpeg, manifest=manifest, frame_interval_sec=10.0)
                mark_stage("extract_media", "ready")

                mark_stage("analyze", "running")
                manifest = analyze_session(store=store, storage=storage, manifest=manifest)
                mark_stage("analyze", "ready")

                mark_stage("evidence_packet", "running")
                manifest = build_evidence_packet(store=store, storage=storage, manifest=manifest)
                mark_stage("evidence_packet", "ready")

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
                midi_key = None
                midi_bytes = None
                if masterclass_id:
                    try:
                        class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
                        for _ in range(40):
                            state = class_manifest.metadata.get("midi_find_state")
                            if state in {"ready", "not_found", "skipped", "skipped_user_upload", "failed", None}:
                                break
                            time.sleep(3)
                            class_manifest = masterclasses.load_by_id(ctx, masterclass_id)
                        midi_key = class_manifest.artifacts.get("reference/midi")
                        if midi_key and storage.exists(midi_key):
                            midi_bytes = storage.read_bytes(midi_key)
                    except FileNotFoundError:
                        class_manifest = None

                run_best_effort("onsets", lambda: detect_rich_onsets(storage=storage, store=store, manifest=manifest))

                if midi_bytes:
                    def run_hmm() -> None:
                        result = align_lesson_with_midi_hmm(
                            storage=storage,
                            store=store,
                            manifest=manifest,
                            midi_bytes=midi_bytes,
                        )
                        persist_hmm_alignment(storage=storage, store=store, manifest=manifest, result=result)

                    run_best_effort("hmm_align", run_hmm)
                else:
                    manifest.metadata["hmm_align_state"] = "skipped"
                    manifest.metadata["hmm_align_error"] = "reference MIDI missing"
                    store.save(manifest)

                if midi_bytes and manifest.metadata.get("hmm_align_state") != "ready":
                    def run_chroma_fallback() -> None:
                        result = align_lesson_with_midi(
                            storage=storage,
                            store=store,
                            ffmpeg=ffmpeg,
                            manifest=manifest,
                            midi_bytes=midi_bytes,
                        )
                        persist_alignment(storage=storage, store=store, manifest=manifest, result=result)

                    run_best_effort("alignment", run_chroma_fallback)
                elif manifest.metadata.get("hmm_align_state") == "ready":
                    manifest.metadata["alignment_state"] = "skipped_hmm_ready"
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
                if intonation_enabled_for_profile(profile) and manifest.metadata.get("hmm_align_state") == "ready":
                    run_best_effort("intonation", lambda: analyze_intonation(storage=storage, store=store, manifest=manifest))
                else:
                    manifest.metadata["intonation_state"] = "skipped"
                    store.save(manifest)

                if manifest.metadata.get("hmm_align_state") == "ready":
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

                if profile.family == "keyboard" and manifest.metadata.get("hmm_align_state") == "ready":
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
                audio_key = manifest.artifacts.get("artifacts/audio.wav")
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

    def _spawn(target, *args) -> None:
        """Run a background job in a real OS thread so multiple jobs run in parallel.
        FastAPI's BackgroundTasks runs handlers sequentially, which causes deadlocks
        when one task waits for another (e.g. score_prep waiting for midi_find)."""
        import threading
        threading.Thread(target=target, args=args, daemon=True).start()

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
            # Wait briefly for MIDI find to land so the score-prep MIDI cross-check
            # can validate measure counts. MIDI find usually finishes in <15s; we
            # cap the wait at 90s and proceed regardless after that.
            import time as _time
            wait_deadline = _time.time() + 90
            while _time.time() < wait_deadline:
                refreshed = masterclasses.load_by_id(ctx, masterclass_id)
                state = refreshed.metadata.get("midi_find_state", "queued")
                if "reference/midi" in refreshed.artifacts or state in ("ready", "failed", "not_found", "skipped", "skipped_user_upload"):
                    manifest = refreshed
                    break
                _time.sleep(2)
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

    def _run_midi_find(masterclass_id: str, tenant_id: str, user_id: str) -> None:
        import concurrent.futures
        import logging
        OVERALL_WALL_CLOCK_SEC = 120

        try:
            ctx = TenantContext(tenant_id=tenant_id, user_id=user_id)
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
            if "reference/midi" in manifest.artifacts:
                manifest.metadata["midi_find_state"] = "skipped_user_upload"
                masterclasses.save(manifest)
                return
            try:
                _build_llm_provider_for_user(user_id)
            except HTTPException as exc:
                manifest.metadata["midi_find_state"] = "skipped"
                manifest.metadata["midi_find_error"] = str(exc.detail)
                masterclasses.save(manifest)
                return

            def _work() -> None:
                # Build a provider with a short HTTP timeout for the search calls
                # so a hung Gemini grounded search doesn't pin the manifest in
                # "running" state for minutes. The wall-clock concurrent.futures
                # timeout below is the outer safety net.
                from masterclass.agent.llm import SharedKeyGeminiConfig
                short_provider = _build_llm_provider_for_user(
                    user_id,
                    config=SharedKeyGeminiConfig(request_timeout_sec=30, max_tool_calls=3),
                )
                auto_attach_midi_to_masterclass(
                    storage=storage,
                    masterclass_store=masterclasses,
                    manifest=manifest,
                    provider=short_provider,
                    config=MidiFinderConfig(model=midi_find_model),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_work)
                try:
                    fut.result(timeout=OVERALL_WALL_CLOCK_SEC)
                except concurrent.futures.TimeoutError:
                    # The Gemini search is still running but we'd rather mark the
                    # state ASAP so the user can move on. The thread will finish
                    # in the background; we just abandon its result.
                    fut.cancel()
                    refreshed = masterclasses.load_by_id(ctx, masterclass_id)
                    refreshed.metadata["midi_find_state"] = "not_found"
                    refreshed.metadata["midi_find_error"] = (
                        f"web search timed out after {OVERALL_WALL_CLOCK_SEC}s — "
                        "upload a MIDI manually or re-run find"
                    )
                    refreshed.metadata["midi_find_updated_at"] = datetime.now(UTC).isoformat()
                    masterclasses.save(refreshed)
        except Exception as exc:
            logging.exception("midi auto-find failed for masterclass %s", masterclass_id)
            try:
                ctx = TenantContext(tenant_id=tenant_id, user_id=user_id)
                manifest = masterclasses.load_by_id(ctx, masterclass_id)
                manifest.metadata["midi_find_state"] = "failed"
                manifest.metadata["midi_find_error"] = f"{type(exc).__name__}: {exc}"
                manifest.metadata["midi_find_updated_at"] = datetime.now(UTC).isoformat()
                masterclasses.save(manifest)
            except Exception:
                pass

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
        if not has_midi:
            manifest.metadata["midi_find_state"] = "queued"
            manifest.metadata["midi_find_updated_at"] = datetime.now(UTC).isoformat()
        masterclasses.save(manifest)
        if has_score_pdf:
            _spawn(
                _run_score_prep,
                manifest.masterclass.masterclass_id,
                manifest.masterclass.tenant_id,
                manifest.masterclass.user_id,
            )
        if not has_midi:
            _spawn(
                _run_midi_find,
                manifest.masterclass.masterclass_id,
                manifest.masterclass.tenant_id,
                manifest.masterclass.user_id,
            )
        return manifest.to_json()

    @app.post("/masterclasses/{masterclass_id}/find-midi")
    def rerun_midi_find(
        masterclass_id: str,
        background: BackgroundTasks,
        ctx: TenantContext = Depends(tenant_from_header),
    ) -> dict:
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        manifest.metadata["midi_find_state"] = "queued"
        manifest.metadata["midi_find_updated_at"] = datetime.now(UTC).isoformat()
        masterclasses.save(manifest)
        _spawn(
            _run_midi_find,
            manifest.masterclass.masterclass_id,
            manifest.masterclass.tenant_id,
            manifest.masterclass.user_id,
        )
        return manifest.to_json()

    @app.get("/masterclasses/{masterclass_id}/midi-find")
    def get_midi_find(masterclass_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = masterclasses.load_by_id(ctx, masterclass_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="masterclass not found") from exc
        audit_key = manifest.artifacts.get("reference/midi_find.json")
        audit = None
        if audit_key and storage.exists(audit_key):
            audit = storage.read_json(audit_key)
        return {
            "state": manifest.metadata.get("midi_find_state", "not_run"),
            "error": manifest.metadata.get("midi_find_error"),
            "updated_at": manifest.metadata.get("midi_find_updated_at"),
            "midi_attached": "reference/midi" in manifest.artifacts,
            "midi_url": manifest.metadata.get("reference_midi_url"),
            "midi_source": manifest.metadata.get("reference_midi_source"),
            "midi_attribution": manifest.metadata.get("reference_midi_attribution"),
            "midi_confidence": manifest.metadata.get("reference_midi_confidence"),
            "audit": audit,
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

    def _player_html_response(session_id: str) -> HTMLResponse:
        # The HTML reads other endpoints client-side with ?user_id; serving the
        # shell as anonymous keeps it linkable and avoids header pre-flight.
        html = (static_dir / "player.html").read_text(encoding="utf-8")
        return HTMLResponse(
            html.replace("__SESSION_ID__", session_id),
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
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        key = manifest.artifacts.get("lesson/comments_enriched.json")
        if not key or not storage.exists(key):
            payload: dict[str, Any] = {"comments": [], "summary": "", "progress_notes": "", "_meta": {}}
        else:
            payload = storage.read_json(key)

        # Prefer the HMM alignment (per-note Viterbi) over the chroma DTW fallback
        # over the teacher's listening estimates. Re-anchor each comment's start
        # time to the alignment-derived measure start.
        align_doc: dict[str, Any] | None = None
        align_source = "llm_estimate"
        for candidate_key, candidate_source in (
            ("analysis/hmm_alignment.json", "hmm_viterbi"),
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

    @app.post("/lessons/{session_id}/chat")
    def lesson_chat(session_id: str, body: ChatRequest = Body(...), ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            message = (body.message or "").strip()
            if not message:
                raise HTTPException(status_code=400, detail="message is required")
            check_message_size(message)
            manifest = store.load_by_id(ctx, session_id)
            if manifest.state != JobState.READY:
                raise HTTPException(status_code=409, detail="lesson is not ready for chat yet")
            if body.conversation_id:
                conversation = load_conversation(storage, store, manifest, body.conversation_id)
                check_conversation_turn_cap(conversation.user_message_count)
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
            result = run_chat_turn(
                storage=storage,
                store=store,
                manifest=manifest,
                provider=provider,
                message=message,
                conversation_id=body.conversation_id,
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
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson not found") from exc
        return list_conversations(storage, store, manifest)

    @app.get("/lessons/{session_id}/chat/{conv_id}")
    def lesson_chat_history(session_id: str, conv_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = store.load_by_id(ctx, session_id)
            return load_conversation(storage, store, manifest, conv_id).to_json()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="lesson or conversation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/lessons/{session_id}/chat/{conv_id}")
    def lesson_chat_delete(session_id: str, conv_id: str, ctx: TenantContext = Depends(tenant_from_header)) -> dict:
        try:
            manifest = store.load_by_id(ctx, session_id)
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
        job = jobs.enqueue(manifest.session, job_type, (body.payload if body else {}))
        return job.to_json()

    return app
