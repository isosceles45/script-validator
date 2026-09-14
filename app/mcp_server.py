"""MCP interface.

A thin wrapper over the same scoring pipeline the REST API uses -- deliberately
no business logic of its own, so the two interfaces can never disagree about a
score. Exposes the validator to any MCP client (Claude Desktop, Claude Code, an
internal agent) over stdio.

    python -m app.mcp_server

Client config:
    {"mcpServers": {"tfs-validator": {
        "command": "/path/to/.venv/bin/python", "args": ["-m", "app.mcp_server"],
        "cwd": "/path/to/script-validator"}}}

Requires the optional `mcp` package: pip install "mcp>=1.2"
"""
from __future__ import annotations

import json
import logging

from .config import get_settings
from .evaluation import golden
from .logging_conf import configure_logging
from .scoring.pipeline import score_script
from .service import get_run_store, get_vector_store, health, new_providers

log = logging.getLogger(__name__)


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "the MCP interface needs the `mcp` package: pip install 'mcp>=1.2'"
        ) from exc

    settings = get_settings()
    server = FastMCP("tfs-script-validator")

    @server.tool()
    def validate_script(brief: str, script: str, product_hint: str | None = None,
                        campaign: str | None = None) -> str:
        """Score a creator script against a campaign brief and the TFS product
        manuals. Returns brief-alignment, message-quality and product-claim-validity
        scores, per-claim verdicts with manual citations, overall feedback, and the
        retrieval evaluation for the run.

        Args:
            brief: the campaign brief the script was written against.
            script: the creator-submitted script to validate.
            product_hint: optional product/SKU name to bias retrieval.
            campaign: optional campaign label for run filtering.
        """
        store = get_vector_store()
        if store.stats()["n_chunks"] == 0:
            return json.dumps({"error": "no manuals ingested; run "
                                        "`python -m app.cli ingest` first"})
        llm, embedder = new_providers(settings)
        result = score_script(brief=brief, script=script, settings=settings,
                              llm=llm, embedder=embedder, store=store,
                              run_store=get_run_store(),
                              product_hint=product_hint, campaign=campaign)
        # Drop the bulky retrieval detail: an MCP client wants the verdict and
        # the citations, and the full evidence stays in the run artifact.
        result.pop("retrieval_detail", None)
        result.pop("inputs", None)
        return json.dumps(result, indent=2, default=str)

    @server.tool()
    def list_products() -> str:
        """List the TFS products currently covered by the ingested manuals.
        Use this to check coverage before validating a script, since claims about
        an uncovered product can only ever come back 'unverifiable'."""
        return json.dumps(get_vector_store().stats(), indent=2, default=str)

    @server.tool()
    def retrieval_eval(k: int | None = None) -> str:
        """Run the full golden-set retrieval evaluation and return recall@k and MRR.
        Use this to check retrieval health independently of any one script."""
        _, embedder = new_providers(settings)
        return json.dumps(golden.run_full(settings, embedder, get_vector_store(), k=k),
                          indent=2, default=str)

    @server.tool()
    def service_health() -> str:
        """Report provider, models, corpus size and readiness of the validator."""
        return json.dumps(health(), indent=2, default=str)

    return server


def main() -> None:
    configure_logging(get_settings().log_level)
    build_server().run()


if __name__ == "__main__":
    main()
