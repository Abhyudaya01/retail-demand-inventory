"""Structured logging helpers for the retail demand project."""

from __future__ import annotations

import logging
import sys

import structlog


def get_logger(name: str, level: str = "INFO") -> structlog.stdlib.BoundLogger:
    """Inputs: logger name and level; outputs: bound logger; side effects: configures logging."""
    logging.basicConfig(format="%(message)s", level=level.upper(), stream=sys.stdout)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(name)
