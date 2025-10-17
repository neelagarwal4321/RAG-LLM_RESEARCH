from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import structlog


def configure_logging() -> None:
    """Configure structlog for JSON logging."""
    if getattr(configure_logging, "_configured", False):
        return

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName("INFO"),
    )

    configure_logging._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
