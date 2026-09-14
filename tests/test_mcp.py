"""MCP interface tests.

Skipped when the optional `mcp` package is absent, so the core suite stays
dependency-free. The import-path test exists because the SDK renamed FastMCP to
MCPServer in 2.0 and the old path fails only at server startup -- invisible to
every other test.
"""
from __future__ import annotations

import logging

import pytest

mcp = pytest.importorskip("mcp")


def test_server_builds_and_registers_every_tool():
    from app.mcp_server import build_server
    server = build_server()
    assert server is not None


def test_logging_can_be_routed_off_stdout():
    # The stdio transport reserves stdout for JSON-RPC frames. One log line
    # written there closes the session, and the failure looks like a protocol
    # error rather than a logging bug.
    import io
    from app.logging_conf import configure_logging

    buffer = io.StringIO()
    configure_logging("INFO", stream=buffer)
    logging.getLogger("probe").info("hello")
    assert "hello" in buffer.getvalue()
    assert logging.getLogger().handlers[0].stream is buffer
