"""Scoring orchestration: brief + script -> scored, cited, audited result."""
from __future__ import annotations

import concurrent.futures
import logging
import time
import uuid
from typing import Any

from ..config import PROMPT_VERSION, Settings
from ..evaluation import golden
from ..evaluation.metrics import grounding_rate
from ..logging_conf import run_id_var
from ..providers import Embedder, LLM
from ..store.run_store import RunStore
from ..store.vector_store import VectorStore
from .aggregator import combine_scores, verdict_label, write_feedback
from .claims import extract_claims
from .mandatories import run as run_mandatory_check
from .retriever import retrieve_for_claims
from .rubric_scorers import score_brief_alignment, score_message_quality
from .verifier import claim_validity_score, tally, verify_claims

log = logging.getLogger(__name__)


def score_script(*, brief: str, script: str, settings: Settings, llm: LLM,
                 embedder: Embedder, store: VectorStore,
                 run_store: RunStore | None = None,
                 product_hint: str | None = None,
                 campaign: str | None = None,
                 run_eval: bool = True) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:16]
    token = run_id_var.set(run_id)
    started = time.perf_counter()
    stage_ms: dict[str, int] = {}

    def timed(name: str, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            stage_ms[name] = int((time.perf_counter() - t0) * 1000)

    try:
        log.info("scoring run started",
                 extra={"provider": settings.provider, "llm_model": llm.name,
                        "embed_model": embedder.name, "campaign": campaign,
                        "product_hint": product_hint,
                        "brief_chars": len(brief), "script_chars": len(script)})

        # The two rubric scorers are independent of the whole claim chain, so they
        # run alongside it rather than after it -- roughly halves wall-clock.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            # The brief branch is now two serial calls (extract mandatories,
            # verify them in code, then score). It still runs alongside the claim
            # chain, which is the longer branch.
            def brief_branch():
                mandatories, rendered = timed(
                    "mandatory_check",
                    lambda: run_mandatory_check(llm, brief=brief, script=script))
                return timed("brief_alignment", lambda: score_brief_alignment(
                    llm, brief=brief, script=script,
                    mandatory_check=rendered, mandatories=mandatories))

            brief_future = pool.submit(brief_branch)
            message_future = pool.submit(
                timed, "message_quality",
                lambda: score_message_quality(llm, brief=brief, script=script))

            extraction = timed("claim_extraction",
                               lambda: extract_claims(llm, brief=brief, script=script))
            hint = product_hint or (extraction.product_mentions[0]
                                    if extraction.product_mentions else None)
            retrieved = timed("retrieval", lambda: retrieve_for_claims(
                settings, embedder, store, extraction.claims, product_hint=hint))
            verdicts = timed("claim_verification",
                             lambda: verify_claims(llm, retrieved))

            brief_result = brief_future.result()
            message_result = message_future.result()

        claim_score = claim_validity_score(verdicts)
        overall, applied_weights = combine_scores(
            brief_score=brief_result["score"],
            message_score=message_result["score"],
            claim_score=claim_score,
            weights=settings.weights)
        verdict = verdict_label(overall, verdicts)

        scores = {"brief_alignment": brief_result["score"],
                  "message_quality": message_result["score"],
                  "claim_validity": claim_score,
                  "overall": overall}

        narrative = timed("overall_feedback", lambda: write_feedback(
            llm, scores=scores, verdict=verdict, verdicts=verdicts,
            brief_result=brief_result, message_result=message_result))

        live = grounding_rate([(r.claim.id, r.hits) for r in retrieved],
                              min_similarity=settings.min_similarity)
        golden_result = timed("retrieval_eval", lambda: golden.run_per_request(
            settings, embedder, store)) if run_eval else {"status": "skipped"}

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = {"llm": llm.usage.as_dict(), "embedding": embedder.usage.as_dict()}
        total_tokens = (llm.usage.input_tokens + llm.usage.output_tokens
                        + embedder.usage.embed_tokens)

        result: dict[str, Any] = {
            "run_id": run_id,
            "created_at": time.time(),
            "campaign": campaign,
            "product_hint": hint,
            "verdict": verdict,
            "scores": scores,
            "weights_applied": applied_weights,
            "overall_feedback": narrative["feedback"],
            "top_actions": narrative["top_actions"],
            "brief_alignment": brief_result,
            "message_quality": message_result,
            "claim_validity": {
                "score": claim_score,
                "tally": tally(verdicts),
                "claims": [v.as_dict() for v in verdicts],
                "non_factual_lines": extraction.non_factual_lines,
                "note": ("no checkable product claims found; this axis was dropped "
                         "and its weight redistributed"
                         if claim_score is None else None),
            },
            "retrieval_eval": {"golden": golden_result, "live": live},
            "retrieval_detail": [r.as_dict() for r in retrieved],
            "inputs": {"brief": brief, "script": script},
            "meta": {
                "provider": settings.provider,
                "llm_model": llm.name,
                "embed_model": embedder.name,
                "prompt_version": PROMPT_VERSION,
                "top_k": settings.top_k,
                "weights_configured": settings.weights,
                "latency_ms": latency_ms,
                "stage_latency_ms": stage_ms,
                "usage": usage,
                "total_tokens": total_tokens,
            },
        }

        if run_store is not None:
            result["artifact_path"] = run_store.save(result)

        log.info("scoring run complete",
                 extra={"verdict": verdict, "scores": scores,
                        "latency_ms": latency_ms, "total_tokens": total_tokens,
                        "grounding_rate": live.get("grounding_rate"),
                        "recall_at_k": golden_result.get("recall_at_k")})
        return result
    finally:
        run_id_var.reset(token)
