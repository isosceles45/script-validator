from __future__ import annotations

import pytest

from app.evaluation import golden
from app.evaluation.golden import GoldenCase
from app.evaluation.metrics import grounding_rate, reciprocal_rank
from app.ingest.pipeline import ingest
from app.store.vector_store import Hit
from tests.fakes import FakeEmbedder


@pytest.fixture
def populated(settings, store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    return embedder


def _search(store, embedder, query, **kwargs):
    vector = embedder.embed([query], stage="query_embed")[0]
    return store.search(query, vector, **kwargs)


def test_search_returns_the_relevant_manual(settings, store, populated):
    hits = _search(store, populated, "tea tree leaf water from Jeju", top_k=3)
    assert hits
    assert "Tea_Tree" in hits[0].source_file


def test_search_respects_top_k(settings, store, populated):
    assert len(_search(store, populated, "skin", top_k=2)) <= 2


def test_product_filter_narrows_results(settings, store, populated):
    hits = _search(store, populated, "cleansing foam", top_k=5,
                   product="Rice Water Bright Cleansing")
    assert hits and all("Rice_Water" in h.source_file for h in hits)


def test_unmatched_product_hint_falls_back_to_full_corpus(settings, store, populated):
    # A bad hint must degrade ranking, never zero out the evidence -- otherwise
    # every claim on a mis-named product returns "unverifiable".
    hits = _search(store, populated, "tea tree leaf water", top_k=3,
                   product="Nonexistent Product XYZ")
    assert hits


def test_empty_store_returns_no_hits(settings, store):
    assert _search(store, FakeEmbedder(), "anything", top_k=5) == []


def test_hits_carry_citable_provenance(settings, store, populated):
    hit = _search(store, populated, "apply two to three drops", top_k=1)[0]
    assert hit.citation()
    assert hit.chunk_id and hit.source_file


def test_lexical_half_of_hybrid_finds_exact_tokens(settings, store, populated):
    # "78 percent" is the kind of exact token a hashed/dense score alone ranks
    # poorly; the BM25 half of the fusion is what rescues it.
    hits = _search(store, populated, "78 percent reported smoother texture", top_k=3)
    assert any("78 percent" in h.text for h in hits)


def test_reciprocal_rank():
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def _hit(dense: float) -> Hit:
    return Hit(chunk_id="c", product="p", section=None, page=None, kind="prose",
               text="t", source_file="f", score=0.0, dense_score=dense,
               lexical_score=0.0)


def test_grounding_rate_uses_raw_similarity_threshold():
    result = grounding_rate([("c1", [_hit(0.9)]), ("c2", [_hit(0.1)]), ("c3", [])],
                            min_similarity=0.25)
    assert result["grounded_claims"] == 1
    assert result["total_claims"] == 3
    assert result["grounding_rate"] == pytest.approx(0.3333, abs=1e-3)
    assert {u["claim_id"] for u in result["ungrounded_claim_ids"]} == {"c2", "c3"}


def test_grounding_rate_with_no_claims_is_none_not_zero():
    # Zero would read as "totally ungrounded" on a script that simply made no
    # claims; None says "not applicable".
    assert grounding_rate([], min_similarity=0.25)["grounding_rate"] is None


def test_golden_relevance_requires_both_source_and_keyword():
    case = GoldenCase(id="g1", claim="x", source_file="Tea_Tree",
                      keywords=["80% tea tree leaf water"])
    right_file_right_kw = Hit(chunk_id="a", product="p", section=None, page=None,
                              kind="prose", text="Contains 80% tea tree leaf water.",
                              source_file="TFS_Tea_Tree.txt", score=0, dense_score=0,
                              lexical_score=0)
    right_file_wrong_kw = Hit(chunk_id="b", product="p", section=None, page=None,
                              kind="prose", text="Lather with water.",
                              source_file="TFS_Tea_Tree.txt", score=0, dense_score=0,
                              lexical_score=0)
    wrong_file = Hit(chunk_id="c", product="p", section=None, page=None, kind="prose",
                     text="Contains 80% tea tree leaf water.",
                     source_file="Rice_Water.txt", score=0, dense_score=0,
                     lexical_score=0)
    assert case.is_relevant(right_file_right_kw)
    assert not case.is_relevant(right_file_wrong_kw)
    assert not case.is_relevant(wrong_file)


def test_golden_eval_scores_a_known_good_case(settings, store, populated, tmp_path):
    settings.golden_path.parent.mkdir(parents=True, exist_ok=True)
    settings.golden_path.write_text(
        "cases:\n"
        "  - id: g1\n"
        "    claim: contains 80% tea tree leaf water from Jeju\n"
        "    source_file: Tea_Tree\n"
        "    keywords: ['80% tea tree leaf water']\n", encoding="utf-8")
    result = golden.run_full(settings, populated, store)
    assert result["status"] == "ok"
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] > 0


def test_missing_golden_set_is_reported_not_scored_as_zero(settings, store, populated):
    result = golden.run_per_request(settings, populated, store)
    assert result["status"] == "no_golden_set"
    assert result["recall_at_k"] is None


def test_dimension_mismatch_fails_loudly(settings, store, populated):
    # Switching PROVIDER without re-ingesting must not silently score claims
    # against randomly-selected manual text.
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.search("tea tree", [0.1] * 99, top_k=3)
