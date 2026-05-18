"""Unit tests for masterclass.observability — JSON formatter + contextvars."""

from __future__ import annotations

import json
import logging

from masterclass.observability import (
    request_id_var,
    setup_logging,
    with_job_context,
    with_stage,
)
from masterclass.observability.logging_setup import JsonFormatter, _ContextFilter


def _make_record(level: int = logging.INFO, msg: str = "hi", **extras) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=None,
        exc_info=None,
    )
    for k, v in extras.items():
        setattr(rec, k, v)
    return rec


def test_json_formatter_emits_one_line_with_core_fields() -> None:
    rec = _make_record()
    out = JsonFormatter().format(rec)
    assert "\n" not in out
    parsed = json.loads(out)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"
    assert parsed["msg"] == "hi"
    assert parsed["ts"].endswith("Z")


def test_context_filter_injects_active_contextvars() -> None:
    flt = _ContextFilter()
    with with_job_context(job_id="job-42", user_id="u1"):
        rec = _make_record()
        assert flt.filter(rec) is True
        out = json.loads(JsonFormatter().format(rec))
        assert out["job_id"] == "job-42"
        assert out["user_id"] == "u1"
    rec = _make_record()
    flt.filter(rec)
    out = json.loads(JsonFormatter().format(rec))
    assert "job_id" not in out
    assert "user_id" not in out


def test_with_stage_scopes_only_within_block() -> None:
    flt = _ContextFilter()
    with with_stage("score_prep"):
        rec = _make_record()
        flt.filter(rec)
        assert json.loads(JsonFormatter().format(rec))["stage"] == "score_prep"
    rec = _make_record()
    flt.filter(rec)
    assert "stage" not in json.loads(JsonFormatter().format(rec))


def test_extras_are_emitted_as_structured_fields() -> None:
    rec = _make_record(http_status=200, duration_ms=42)
    out = json.loads(JsonFormatter().format(rec))
    assert out["http_status"] == 200
    assert out["duration_ms"] == 42


def test_non_serialisable_extras_are_repr_fallback() -> None:
    rec = _make_record(weird=object())
    out = json.loads(JsonFormatter().format(rec))
    assert isinstance(out["weird"], str)
    assert "object" in out["weird"]


def test_setup_logging_installs_single_handler_and_returns_level() -> None:
    name = setup_logging()
    assert name in {"DEBUG", "INFO", "WARNING", "ERROR"}
    root = logging.getLogger()
    assert len(root.handlers) == 1


def test_request_id_var_resets_outside_token_scope() -> None:
    assert request_id_var.get() is None
    token = request_id_var.set("abc")
    try:
        assert request_id_var.get() == "abc"
    finally:
        request_id_var.reset(token)
    assert request_id_var.get() is None
