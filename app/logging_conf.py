"""Structured JSON logging. Every line carries run_id when one is in scope, so a
single run can be reconstructed from logs alone (Cloud Logging parses these natively)."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            # Cloud Logging picks up "severity"; stdlib readers still get levelname.
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = run_id_var.get()
        if rid:
            payload["run_id"] = rid
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", stream=None) -> None:
    """Defaults to stdout, which is what Cloud Logging scrapes. The MCP server
    passes stderr instead: its stdio transport reserves stdout exclusively for
    JSON-RPC frames, and a single log line written there kills the session."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "urllib3", "pdfminer", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
