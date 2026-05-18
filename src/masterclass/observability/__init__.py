"""Production observability: structured JSON logging + trace context.

Exposes a small API that the rest of the app uses to get logs that carry
request/job correlation IDs through to Log Analytics:

    from masterclass.observability import (
        setup_logging,
        RequestContextMiddleware,
        with_job_context,
        request_id_var,
        user_id_var,
        job_id_var,
        stage_var,
    )

The JSON formatter emits one JSON object per line so each Container App
console-log record (``ContainerAppConsoleLogs_CL.Log_s``) can be parsed
in KQL with ``parse_json(Log_s)``.

See ~/fermata-hosting/kql/ for ready-made queries.
"""

from .logging_setup import (
    RequestContextMiddleware,
    bind_request_principal,
    job_id_var,
    job_kind_var,
    masterclass_id_var,
    request_id_var,
    session_id_var,
    setup_logging,
    stage_var,
    tenant_id_var,
    user_id_var,
    with_job_context,
    with_stage,
)

__all__ = [
    "RequestContextMiddleware",
    "bind_request_principal",
    "job_id_var",
    "job_kind_var",
    "masterclass_id_var",
    "request_id_var",
    "session_id_var",
    "setup_logging",
    "stage_var",
    "tenant_id_var",
    "user_id_var",
    "with_job_context",
    "with_stage",
]
