"""Command line entry point.

    python -m app.cli ingest [--force] [--only tea_tree]
    python -m app.cli score --brief brief.md --script script.md [--product "Tea Tree Pore Ampoule"]
    python -m app.cli eval [--k 5]
    python -m app.cli corpus
    python -m app.cli runs [--limit 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_settings
from .evaluation import golden
from .ingest.pipeline import ingest
from .logging_conf import configure_logging
from .scoring.pipeline import score_script
from .service import get_run_store, get_vector_store, new_providers

log = logging.getLogger(__name__)


def _read(path_or_text: str) -> str:
    path = Path(path_or_text)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return path_or_text


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    _, embedder = new_providers(settings)
    report = ingest(settings, embedder, get_vector_store(),
                    force=args.force, only=args.only)
    print(report.summary())
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = get_vector_store()
    if store.stats()["n_chunks"] == 0:
        print("error: no manuals ingested; run `python -m app.cli ingest` first",
              file=sys.stderr)
        return 2
    llm, embedder = new_providers(settings)
    result = score_script(brief=_read(args.brief), script=_read(args.script),
                          settings=settings, llm=llm, embedder=embedder,
                          store=store, run_store=get_run_store(),
                          product_hint=args.product, campaign=args.campaign,
                          run_eval=not args.no_eval)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(render_report(result))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = get_settings()
    _, embedder = new_providers(settings)
    result = golden.run_full(settings, embedder, get_vector_store(), k=args.k)
    print(json.dumps(result, indent=2))
    return 0


def cmd_corpus(_: argparse.Namespace) -> int:
    stats = get_vector_store().stats()
    print(f"{stats['n_manuals']} manuals, {stats['n_chunks']} chunks\n")
    for manual in stats["manuals"]:
        print(f"  {manual['product']:45s} {manual['n_chunks']:4d} chunks  "
              f"({manual['source_file']})")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    for run in get_run_store().recent(args.limit):
        print(f"{run['run_id']}  overall={run['overall_score']}  "
              f"claims={run['n_supported']}S/{run['n_contradicted']}C/"
              f"{run['n_unverifiable']}U  recall@k={run['recall_at_k']}  "
              f"{run['latency_ms']}ms")
    return 0


def render_report(result: dict) -> str:
    """Human-readable scorecard. The claim table comes first because a
    contradicted claim outranks every other finding in consequence."""
    scores = result["scores"]
    claim_score = scores["claim_validity"]
    lines = [
        "=" * 78,
        f"  RUN {result['run_id']}    VERDICT: {result['verdict'].upper()}",
        "=" * 78,
        f"  Overall                 {scores['overall']:>5}/10",
        f"  Brief alignment         {scores['brief_alignment']:>5}/10",
        f"  Marketing message       {scores['message_quality']:>5}/10",
        f"  Product claim validity  "
        + (f"{claim_score:>5}/10" if claim_score is not None else "  n/a"),
        "",
        "-" * 78,
        "  PRODUCT CLAIMS",
        "-" * 78,
    ]
    order = {"contradicted": 0, "partially_supported": 1, "unverifiable": 2,
             "supported": 3}
    claims = sorted(result["claim_validity"]["claims"],
                    key=lambda c: order.get(c["verdict"], 9))
    if not claims:
        lines.append("  (no checkable product claims in this script)")
    for claim in claims:
        lines.append(f"  [{claim['verdict'].upper():21s}] ({claim['risk']} risk) "
                     f"{claim['text']}")
        if claim.get("citations"):
            lines.append(f"      source: {'; '.join(claim['citations'])}")
        if claim.get("manual_quote"):
            lines.append(f"      manual: \"{claim['manual_quote'][:160]}\"")
        if claim.get("suggested_fix"):
            lines.append(f"      fix:    \"{claim['suggested_fix'][:160]}\"")
        lines.append("")

    ev = result["retrieval_eval"]
    golden_ev, live = ev["golden"], ev["live"]
    lines += ["-" * 78, "  RETRIEVAL EVALUATION", "-" * 78]
    if golden_ev.get("status") == "ok":
        lines.append(f"  Golden set   recall@{golden_ev['k']}={golden_ev['recall_at_k']}  "
                     f"MRR={golden_ev['mrr']}  "
                     f"({golden_ev['n_cases']} of {golden_ev['sampled_from']} cases)")
    else:
        lines.append(f"  Golden set   {golden_ev.get('status')}: "
                     f"{golden_ev.get('note', '')}")
    lines.append(f"  This script  grounding rate={live['grounding_rate']} "
                 f"({live['grounded_claims']}/{live['total_claims']} claims above "
                 f"similarity {live['min_similarity']})")

    meta = result["meta"]
    lines += [
        "", "-" * 78, "  OVERALL FEEDBACK", "-" * 78, "",
        result["overall_feedback"], "",
        "  Top actions:",
    ]
    lines += [f"    {i}. {a}" for i, a in enumerate(result["top_actions"], 1)]
    lines += [
        "", "-" * 78,
        f"  {meta['provider']}/{meta['llm_model']} + {meta['embed_model']} | "
        f"prompts {meta['prompt_version']} | {meta['latency_ms']}ms | "
        f"{meta['total_tokens']} tokens",
        f"  artifact: {result.get('artifact_path', 'not persisted')}",
        "=" * 78,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli",
                                     description="TFS creative script validator")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest product manuals into the vector store")
    p_ingest.add_argument("--force", action="store_true",
                          help="re-embed even if the checksum is unchanged")
    p_ingest.add_argument("--only", nargs="*", help="substring filter on filenames")
    p_ingest.add_argument("--json", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_score = sub.add_parser("score", help="score a script against a brief")
    p_score.add_argument("--brief", required=True, help="file path or literal text")
    p_score.add_argument("--script", required=True, help="file path or literal text")
    p_score.add_argument("--product", default=None, help="product/SKU retrieval hint")
    p_score.add_argument("--campaign", default=None)
    p_score.add_argument("--no-eval", action="store_true",
                         help="skip the golden-set retrieval eval")
    p_score.add_argument("--json", action="store_true")
    p_score.set_defaults(func=cmd_score)

    p_eval = sub.add_parser("eval", help="run the full golden-set retrieval eval")
    p_eval.add_argument("--k", type=int, default=None)
    p_eval.set_defaults(func=cmd_eval)

    sub.add_parser("corpus", help="show what has been ingested").set_defaults(func=cmd_corpus)

    p_runs = sub.add_parser("runs", help="list recent scoring runs")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.set_defaults(func=cmd_runs)

    args = parser.parse_args(argv)
    configure_logging(args.log_level or get_settings().log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
