"""HTTP-layer tests.

`app.api` resolves settings at import time (Cloud Run wants config failures at
cold start, not on the first request), so the module is reloaded against a
temp-dir environment rather than monkeypatched in place.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ingest.pipeline import ingest
from tests.fakes import FakeEmbedder, ScriptedLLM, default_responses


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "store.sqlite3"))
    monkeypatch.setenv("TOP_K", "3")

    import app.config, app.service, app.api
    app.config.get_settings.cache_clear()
    app.service.get_vector_store.cache_clear()
    app.service.get_run_store.cache_clear()
    api = importlib.reload(app.api)

    manuals = tmp_path / "manuals"
    manuals.mkdir(parents=True, exist_ok=True)
    (manuals / "TFS_Tea_Tree_Pore_Ampoule.txt").write_text(
        "Product Overview\nContains 80% tea tree leaf water from Jeju.\n\n"
        "Usage Instructions\nApply 2 to 3 drops morning and evening.\n",
        encoding="utf-8")

    embedder = FakeEmbedder()
    monkeypatch.setattr(api, "new_providers",
                        lambda *_a, **_k: (ScriptedLLM(default_responses()), embedder))

    yield TestClient(api.app), api, embedder, app.config.get_settings()

    app.config.get_settings.cache_clear()
    app.service.get_vector_store.cache_clear()
    app.service.get_run_store.cache_clear()


def _ingest(api, embedder, settings):
    from app.service import get_vector_store
    ingest(settings, embedder, get_vector_store())


def test_health_reports_degraded_before_ingestion(client):
    http, *_ = client
    body = http.get("/health").json()
    assert body["status"] == "degraded"
    assert "no manuals ingested" in body["reason"]


def test_score_refuses_on_empty_corpus_with_actionable_error(client):
    http, *_ = client
    response = http.post("/score", json={"brief": "Promote the ampoule to Gen Z.",
                                         "script": "Hey besties, it is great."})
    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]


def test_score_returns_a_full_scorecard(client):
    http, api, embedder, settings = client
    _ingest(api, embedder, settings)

    response = http.post("/score", json={
        "brief": "Promote the Tea Tree Pore Ampoule to Gen Z on TikTok.",
        "script": "Hey besties! It contains tea tree leaf water.",
        "product_hint": "Tea Tree Pore Ampoule", "campaign": "q4-tiktok"})
    assert response.status_code == 200
    body = response.json()

    assert set(body["scores"]) == {"brief_alignment", "message_quality",
                                   "claim_validity", "overall"}
    assert body["verdict"]
    assert body["overall_feedback"]
    assert body["claim_validity"]["claims"][0]["citations"]
    assert body["retrieval_eval"]["live"]["grounding_rate"] is not None
    assert body["meta"]["latency_ms"] >= 0

    # The run is retrievable afterwards, by id and in the listing.
    run_id = body["run_id"]
    assert http.get(f"/runs/{run_id}").json()["run_id"] == run_id
    assert any(r["run_id"] == run_id for r in http.get("/runs").json()["runs"])


def test_unknown_run_is_404(client):
    http, *_ = client
    assert http.get("/runs/deadbeef").status_code == 404


def test_short_input_is_rejected_before_any_model_call(client):
    http, *_ = client
    assert http.post("/score", json={"brief": "x", "script": "y"}).status_code == 422


def test_provider_failure_surfaces_as_502_not_500(client, monkeypatch):
    http, api, embedder, settings = client
    _ingest(api, embedder, settings)

    from app.providers import ProviderError

    def explode(*_a, **_k):
        raise ProviderError("rate limited after 4 attempts")

    monkeypatch.setattr(api, "new_providers", explode)
    response = http.post("/score", json={"brief": "a valid brief here",
                                         "script": "a valid script here"})
    assert response.status_code == 502
    assert response.json()["error"] == "provider_error"


def test_corpus_and_metrics_endpoints(client):
    http, api, embedder, settings = client
    _ingest(api, embedder, settings)
    corpus = http.get("/corpus").json()
    assert corpus["n_manuals"] == 1 and corpus["n_chunks"] > 0
    assert "trends" in http.get("/metrics").json()
