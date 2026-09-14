from __future__ import annotations

import pytest

from app.ingest.pipeline import ingest
from app.scoring.aggregator import combine_scores, verdict_label
from app.scoring.claims import Claim, extract_claims
from app.scoring.pipeline import score_script
from app.scoring.verifier import Verdict, claim_validity_score, verify_claims
from app.scoring.retriever import Retrieved
from tests.fakes import FakeEmbedder, ScriptedLLM, default_responses


def _verdict(kind: str, risk: str = "medium") -> Verdict:
    return Verdict(claim=Claim(id="c", text="t", risk=risk), verdict=kind,
                   confidence=0.9, rationale="", evidence_chunk_ids=[],
                   manual_quote=None, suggested_fix=None, citations=[])


# --------------------------- claim validity scoring ---------------------------

def test_all_supported_claims_score_ten():
    assert claim_validity_score([_verdict("supported"), _verdict("supported")]) == 10.0


def test_no_claims_returns_none_not_ten():
    # A claim-free script has nothing to check; scoring it 10 would reward
    # avoiding claims entirely.
    assert claim_validity_score([]) is None


def test_high_risk_contradiction_dominates():
    score = claim_validity_score([_verdict("contradicted", "high")]
                                 + [_verdict("supported")] * 10)
    assert score == 5.0


def test_contradiction_penalty_scales_with_risk():
    high = claim_validity_score([_verdict("contradicted", "high")])
    medium = claim_validity_score([_verdict("contradicted", "medium")])
    low = claim_validity_score([_verdict("contradicted", "low")])
    assert high < medium < low


def test_unverifiable_penalised_far_less_than_contradicted():
    # "The manuals are silent" is a catalogue gap, not a false claim.
    assert (claim_validity_score([_verdict("unverifiable", "high")])
            > claim_validity_score([_verdict("contradicted", "high")]))


def test_score_floors_at_zero():
    assert claim_validity_score([_verdict("contradicted", "high")] * 10) == 0.0


# ------------------------------- aggregation ---------------------------------

def test_weighted_combination():
    weights = {"brief_alignment": 0.3, "message_quality": 0.25, "claim_validity": 0.45}
    overall, applied = combine_scores(brief_score=8.0, message_score=6.0,
                                      claim_score=10.0, weights=weights)
    assert overall == pytest.approx(8.4, abs=0.01)
    assert sum(applied.values()) == pytest.approx(1.0)


def test_weights_renormalise_when_claim_axis_is_dropped():
    weights = {"brief_alignment": 0.3, "message_quality": 0.25, "claim_validity": 0.45}
    overall, applied = combine_scores(brief_score=8.0, message_score=6.0,
                                      claim_score=None, weights=weights)
    assert "claim_validity" not in applied
    assert sum(applied.values()) == pytest.approx(1.0)
    # 0.3/0.55 * 8 + 0.25/0.55 * 6
    assert overall == pytest.approx(7.09, abs=0.01)


def test_contradicted_claim_forces_blocking_verdict_regardless_of_score():
    assert verdict_label(9.5, [_verdict("contradicted", "high")]) == "needs_revision_blocking"
    assert verdict_label(9.5, [_verdict("supported")]) == "approved"
    assert verdict_label(6.5, [_verdict("supported")]) == "approved_with_edits"
    assert verdict_label(4.0, [_verdict("unverifiable")]) == "needs_revision"


# ------------------------------- verification --------------------------------

def test_claims_with_no_evidence_skip_the_llm_entirely():
    llm = ScriptedLLM({})  # any call would raise
    verdicts = verify_claims(llm, [Retrieved(claim=Claim(id="c1", text="t"), hits=[])])
    assert verdicts[0].verdict == "unverifiable"
    assert llm.seen == []


def test_verifier_rejects_hallucinated_chunk_ids(settings, store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    vector = embedder.embed(["tea tree leaf water"], stage="query_embed")[0]
    hits = store.search("tea tree leaf water", vector, top_k=2)

    llm = ScriptedLLM({"claim_verification": {
        "verdict": "supported", "confidence": 0.9, "rationale": "",
        "evidence_chunk_ids": ["totally:made:up"], "manual_quote": "q",
        "suggested_fix": None}})
    verdict = verify_claims(llm, [Retrieved(claim=Claim(id="c1", text="t"), hits=hits)])[0]
    assert "totally:made:up" not in verdict.evidence_chunk_ids
    assert verdict.evidence_chunk_ids  # falls back to the top hit, still real


def test_unknown_verdict_string_degrades_to_unverifiable(settings, store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    vector = embedder.embed(["tea tree"], stage="query_embed")[0]
    hits = store.search("tea tree", vector, top_k=1)
    llm = ScriptedLLM({"claim_verification": {"verdict": "probably fine",
                                              "confidence": 2.5, "rationale": ""}})
    verdict = verify_claims(llm, [Retrieved(claim=Claim(id="c1", text="t"), hits=hits)])[0]
    assert verdict.verdict == "unverifiable"
    assert verdict.confidence == 1.0  # clamped into range


def test_one_failed_verification_does_not_lose_the_others(settings, store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    vector = embedder.embed(["tea tree"], stage="query_embed")[0]
    hits = store.search("tea tree", vector, top_k=1)

    calls = {"n": 0}

    def flaky(_user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider exploded")
        return {"verdict": "supported", "confidence": 0.9, "rationale": "ok",
                "evidence_chunk_ids": [], "manual_quote": "q", "suggested_fix": None}

    llm = ScriptedLLM({"claim_verification": flaky})
    items = [Retrieved(claim=Claim(id=f"c{i}", text="t"), hits=hits) for i in range(3)]
    verdicts = verify_claims(llm, items, max_workers=1)
    assert len(verdicts) == 3
    assert sum(v.verdict == "supported" for v in verdicts) == 2
    assert sum(v.verdict == "unverifiable" for v in verdicts) == 1


# ------------------------------- extraction ----------------------------------

def test_extractor_reassigns_ids_and_validates_enums():
    llm = ScriptedLLM({"claim_extraction": {
        "claims": [
            {"id": "dup", "text": "claim one", "risk": "EXTREME", "type": "nonsense"},
            {"id": "dup", "text": "claim two", "risk": "high", "type": "safety"},
            {"id": "x", "text": "   "},  # dropped: empty
        ],
        "product_mentions": ["Ampoule"], "non_factual_lines": []}})
    result = extract_claims(llm, brief="b", script="s")
    assert [c.id for c in result.claims] == ["c1", "c2"]
    assert result.claims[0].risk == "medium"      # invalid enum coerced
    assert result.claims[0].type == "efficacy"
    assert result.claims[1].risk == "high"


# ---------------------------- end-to-end pipeline ----------------------------

def test_full_pipeline_produces_a_complete_audited_run(settings, store, run_store,
                                                       manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    llm = ScriptedLLM(default_responses())

    result = score_script(brief="Promote the Tea Tree Pore Ampoule to Gen Z.",
                          script="Hey besties! It contains tea tree leaf water.",
                          settings=settings, llm=llm, embedder=embedder,
                          store=store, run_store=run_store,
                          product_hint="Tea Tree Pore Ampoule", campaign="q4")

    assert result["scores"]["overall"] > 0
    assert result["verdict"] == "approved"
    assert result["claim_validity"]["tally"]["total"] == 1
    assert result["claim_validity"]["claims"][0]["citations"]
    assert result["overall_feedback"]
    assert result["retrieval_eval"]["live"]["grounding_rate"] is not None
    assert result["meta"]["prompt_version"]
    assert result["meta"]["stage_latency_ms"]["claim_verification"] >= 0
    # Persisted and replayable.
    assert run_store.get(result["run_id"])["run_id"] == result["run_id"]
    assert run_store.recent(5)[0]["run_id"] == result["run_id"]


def test_contradicted_claim_blocks_an_otherwise_strong_script(settings, store,
                                                              run_store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    responses = default_responses(
        claims=[{"id": "c1", "text": "cures acne overnight", "quote": "Cures acne!",
                 "product": "Tea Tree Pore Ampoule", "type": "efficacy",
                 "risk": "high"}],
        verdict="contradicted")
    result = score_script(brief="b" * 20, script="s" * 20, settings=settings,
                          llm=ScriptedLLM(responses), embedder=embedder,
                          store=store, run_store=run_store)
    assert result["verdict"] == "needs_revision_blocking"
    assert result["scores"]["claim_validity"] == 5.0


def test_claim_free_script_drops_the_claim_axis(settings, store, run_store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    responses = default_responses(claims=[])
    result = score_script(brief="b" * 20, script="s" * 20, settings=settings,
                          llm=ScriptedLLM(responses), embedder=embedder,
                          store=store, run_store=run_store)
    assert result["scores"]["claim_validity"] is None
    assert "claim_validity" not in result["weights_applied"]
    assert result["claim_validity"]["note"]


def test_verifier_prompt_guards_against_competitor_rows():
    # The TFS training decks carry competitive pricing tables (The Ordinary,
    # Cosrx, Anua). A rival's spec row must never support a TFS product claim.
    from app.scoring.prompts import VERIFIER_SYSTEM
    lowered = VERIFIER_SYSTEM.lower()
    assert "competitor" in lowered
    assert "the ordinary" in lowered


def test_unsubstantiated_high_risk_claim_blocks_publication():
    # "dermatologist proven to cure acne" -- the manual is silent, which is not
    # exoneration. You cannot publish a medical claim you cannot evidence.
    assert verdict_label(9.0, [_verdict("unverifiable", "high")]) == "needs_revision_blocking"


def test_unverifiable_at_lower_risk_does_not_block():
    # "contains mango butter" with a silent manual is a documentation gap, not a
    # blocker -- otherwise every catalogue hole becomes a publication stop.
    assert verdict_label(9.0, [_verdict("unverifiable", "medium")]) == "approved"
    assert verdict_label(9.0, [_verdict("unverifiable", "low")]) == "approved"


def test_high_risk_unverifiable_is_near_fatal_to_the_claim_score():
    score = claim_validity_score([_verdict("unverifiable", "high")])
    assert score == 6.0
    # Still strictly better than an outright contradiction: the manual refuting a
    # claim is worse than the manual being silent on it.
    assert score > claim_validity_score([_verdict("contradicted", "high")])
