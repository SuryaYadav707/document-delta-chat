"""Structured JSON logging — structlog, with a correlation id on every record.

correlation_id == the active trace id (see tracing.trace), so logs and the
per-run trace file join. JSON output, not free text.
"""
from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def _add_correlation_id(_, __, event_dict):
    from src.observability.tracing import _current_trace
    tr = _current_trace.get()
    if tr is not None:
        event_dict["correlation_id"] = tr.id
    return event_dict


def configure(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", stream=sys.stderr,
                        level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "delta-chat"):
    if not _configured:
        configure()
    return structlog.get_logger(name)
