"""Structured JSON logging + ASGI request context middleware.

Design goals
------------
* **Zero new runtime deps.** Only stdlib + Starlette (already required by
  FastAPI). We do not pull in OpenTelemetry, structlog, etc. — those are
  options for later if we outgrow this.
* **Every log line carries trace context** (request_id, user_id, job_id,
  stage, masterclass_id, session_id) by propagating through contextvars
  and merging into each ``LogRecord`` via a ``logging.Filter``.
* **JSON-per-line output** so Log Analytics can parse with
  ``parse_json(Log_s)``. Falls back to text format when
  ``MASTERCLASS_LOG_FORMAT=text`` (e.g., local dev).

Context lifecycle
-----------------
* For HTTP requests: ``RequestContextMiddleware`` generates (or honours
  an inbound ``X-Request-ID`` header) one request_id per request and
  binds it for the duration of the request handler. The same id is
  echoed back in the response ``X-Request-ID`` header so a user/curl
  caller can correlate.

* For background pipelines: wrap the entry point with
  ``with_job_context(job_id=..., job_kind=...)``. Any logs emitted from
  inside that block — including from libraries we don't control like
  the Gemini SDK retries — will carry job_id automatically.

* For long pipelines with substages: use ``with_stage("score_prep")``
  to scope a span of logs to a stage name.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Trace context — propagated via contextvars so every log line inherits.
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
job_kind_var: ContextVar[str | None] = ContextVar("job_kind", default=None)
stage_var: ContextVar[str | None] = ContextVar("stage", default=None)
masterclass_id_var: ContextVar[str | None] = ContextVar("masterclass_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)

_ALL_CONTEXT_VARS: tuple[tuple[str, ContextVar[str | None]], ...] = (
    ("request_id", request_id_var),
    ("user_id", user_id_var),
    ("tenant_id", tenant_id_var),
    ("job_id", job_id_var),
    ("job_kind", job_kind_var),
    ("stage", stage_var),
    ("masterclass_id", masterclass_id_var),
    ("session_id", session_id_var),
)

# Standard LogRecord attributes — anything *not* in this set on a record's
# __dict__ is treated as "extra" and emitted as a structured field.
_LOGRECORD_STANDARD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


class _ContextFilter(logging.Filter):
    """Inject contextvars onto every LogRecord so the formatter can see them."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        for name, var in _ALL_CONTEXT_VARS:
            value = var.get()
            if value is not None and not hasattr(record, name):
                setattr(record, name, value)
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record. Includes context + extras + exc."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Trace context (only included when present).
        for name, _var in _ALL_CONTEXT_VARS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        # Caller's extras (any record attribute not in stdlib set + not a context var).
        ctx_names = {name for name, _ in _ALL_CONTEXT_VARS}
        for key, value in record.__dict__.items():
            if key in _LOGRECORD_STANDARD_ATTRS or key in ctx_names:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)  # ensure serialisable
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    """Compact text formatter for local dev. Surfaces context inline."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base = super().format(record)
        ctx_bits: list[str] = []
        for name, _var in _ALL_CONTEXT_VARS:
            value = getattr(record, name, None)
            if value is not None:
                ctx_bits.append(f"{name}={value}")
        if ctx_bits:
            base = f"{base} [{' '.join(ctx_bits)}]"
        return base


def setup_logging() -> str:
    """Configure root logging once at process start.

    Env vars:
      * ``MASTERCLASS_LOG_LEVEL`` (default ``INFO``)
      * ``MASTERCLASS_LOG_FORMAT`` (``json`` (default) | ``text``)

    Returns the effective log-level name so callers can log a startup line.
    """
    level_name = (os.environ.get("MASTERCLASS_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = (os.environ.get("MASTERCLASS_LOG_FORMAT") or "json").lower()

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else _TextFormatter())
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Tame noisy libraries — we still want their warnings, just not their info.
    for noisy in ("azure", "azure.core", "azure.identity", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return level_name


# ---------------------------------------------------------------------------
# Background job context.
# ---------------------------------------------------------------------------

@contextmanager
def with_job_context(
    *,
    job_id: str | None = None,
    job_kind: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    masterclass_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[None]:
    """Bind background-job context for the duration of the with-block.

    All log lines emitted inside the block will carry these fields. Used
    at the top of every background-thread entry point so a single job_id
    threads through dozens of log lines (Gemini retries, stage transitions,
    storage IO, etc.).
    """
    tokens = []
    if job_id is not None:
        tokens.append((job_id_var, job_id_var.set(job_id)))
    if job_kind is not None:
        tokens.append((job_kind_var, job_kind_var.set(job_kind)))
    if tenant_id is not None:
        tokens.append((tenant_id_var, tenant_id_var.set(tenant_id)))
    if user_id is not None:
        tokens.append((user_id_var, user_id_var.set(user_id)))
    if masterclass_id is not None:
        tokens.append((masterclass_id_var, masterclass_id_var.set(masterclass_id)))
    if session_id is not None:
        tokens.append((session_id_var, session_id_var.set(session_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


@contextmanager
def with_stage(stage: str) -> Iterator[None]:
    """Scope ``stage_var`` to a span of code (e.g., a pipeline stage)."""
    token = stage_var.set(stage)
    try:
        yield
    finally:
        stage_var.reset(token)


# ---------------------------------------------------------------------------
# ASGI request middleware — generates request_id, captures user, logs start/end.
# ---------------------------------------------------------------------------

_REQUEST_LOG = logging.getLogger("masterclass.request")

# Paths whose access logs we downgrade to DEBUG. Static files + health checks
# would otherwise dominate the log volume and add no signal.
_QUIET_PREFIXES = ("/static/", "/healthz", "/favicon")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Per-request ``request_id`` + structured access logging.

    Honours an inbound ``X-Request-ID`` header so callers (e.g., curl,
    a debug session, the future React UI) can pin their own correlation
    ID. Echoes the id back on the response so the caller can match.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        inbound = request.headers.get("x-request-id")
        request_id = inbound if inbound and len(inbound) <= 64 else uuid.uuid4().hex
        rid_token = request_id_var.set(request_id)
        # tenant/user is filled in by route handlers once they've resolved
        # auth — we leave it blank here so the request_id is bound early
        # enough to cover auth failures too.
        tenant_token = tenant_id_var.set(None)
        user_token = user_id_var.set(None)

        path = request.url.path
        quiet = any(path.startswith(p) for p in _QUIET_PREFIXES)
        start = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        except Exception:  # noqa: BLE001
            _REQUEST_LOG.exception(
                "request unhandled exception",
                extra={
                    "http_method": request.method,
                    "http_path": path,
                },
            )
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            log = _REQUEST_LOG.debug if quiet else _REQUEST_LOG.info
            log(
                "request",
                extra={
                    "http_method": request.method,
                    "http_path": path,
                    "http_status": status_code,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(rid_token)
            tenant_id_var.reset(tenant_token)
            user_id_var.reset(user_token)


def bind_request_principal(*, tenant_id: str | None, user_id: str | None) -> None:
    """Called from auth dependencies to attach principal to the active request."""
    if tenant_id is not None:
        tenant_id_var.set(tenant_id)
    if user_id is not None:
        user_id_var.set(user_id)
