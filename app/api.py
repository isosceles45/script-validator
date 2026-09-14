"""HTTP service. Deploys unchanged to Cloud Run (container listens on $PORT)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import PROMPT_VERSION, get_settings
from .evaluation import golden
from .logging_conf import configure_logging
from .providers import ProviderError
from .scoring.pipeline import score_script
from .service import get_run_store, get_vector_store, health, new_providers

settings = get_settings()
configure_logging(settings.log_level)
log = logging.getLogger(__name__)

app = FastAPI(
    title="TFS Creative Script Validator",
    version="1.0.0",
    description="Scores a creator script against a campaign brief and the TFS "
                "product manual corpus (brief alignment, message quality, "
                "product claim validity), with per-run retrieval evaluation.",
)

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


class ScoreRequest(BaseModel):
    brief: str = Field(..., min_length=10,
                       description="The campaign brief the script was written against")
    script: str = Field(..., min_length=10, description="The creator-submitted script")
    product_hint: str | None = Field(
        None, description="Optional product/SKU name to bias retrieval. Soft: if it "
                          "matches no manual the full corpus is searched anyway.")
    campaign: str | None = Field(None, description="Campaign label, for run filtering")
    run_eval: bool = Field(True, description="Run the golden-set retrieval eval")


@app.exception_handler(ProviderError)
async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
    # 502, not 500: the failure is upstream at the model provider, and the
    # distinction matters for alerting and for client retry behaviour.
    log.error("provider error", extra={"error": str(exc)})
    return JSONResponse(status_code=502,
                        content={"error": "provider_error", "detail": str(exc)})


@app.get("/", include_in_schema=False)
def index() -> Any:
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return JSONResponse({"service": "tfs-script-validator", "docs": "/docs"})


@app.get("/health")
def health_check() -> dict[str, Any]:
    return health()


@app.get("/corpus")
def corpus() -> dict[str, Any]:
    """What the validator actually knows about -- the honest answer to 'why was
    my claim unverifiable'."""
    return get_vector_store().stats()


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    store = get_vector_store()
    if store.stats()["n_chunks"] == 0:
        raise HTTPException(
            status_code=503,
            detail="No manuals have been ingested. Run `python -m app.cli ingest` "
                   "before scoring, or every claim will come back unverifiable.")
    llm, embedder = new_providers(settings)
    return score_script(brief=request.brief, script=request.script,
                        settings=settings, llm=llm, embedder=embedder,
                        store=store, run_store=get_run_store(),
                        product_hint=request.product_hint,
                        campaign=request.campaign, run_eval=request.run_eval)


@app.get("/runs")
def runs(limit: int = 20) -> dict[str, Any]:
    return {"runs": get_run_store().recent(limit)}


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    result = get_run_store().get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return result


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Aggregate observability: score drift, retrieval health, latency, cost."""
    return {"trends": get_run_store().trends(),
            "corpus": get_vector_store().stats(),
            "prompt_version": PROMPT_VERSION,
            "provider": settings.provider}


@app.post("/eval/retrieval")
def full_eval(k: int | None = None) -> dict[str, Any]:
    """Full golden-set evaluation. Used to compare providers or chunking changes
    on identical cases, independently of any single scoring run."""
    _, embedder = new_providers(settings)
    return {"embed_model": embedder.name, "provider": settings.provider,
            **golden.run_full(settings, embedder, get_vector_store(), k=k)}
