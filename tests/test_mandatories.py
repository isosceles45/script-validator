"""Deterministic mandatory checking.

The regression these guard against was observed for real: a scorer reported
"did not say the full product name" about a script opening with the product
name, costing 5 points on the mandatories dimension.
"""
from __future__ import annotations

from app.scoring.mandatories import (JUDGEMENT, LITERAL, Mandatory, check,
                                     extract_mandatories, format_for_prompt,
                                     normalize, summary)
from app.scoring.rubric_scorers import score_brief_alignment
from tests.fakes import ScriptedLLM


def _lit(text: str, *tokens: str) -> Mandatory:
    return Mandatory(text=text, kind=LITERAL, tokens=list(tokens))


def test_finds_a_present_hashtag():
    [m] = check("Loved this one. #PoreCheck", [_lit("Include #PoreCheck", "#PoreCheck")])
    assert m.present is True and m.matched_token == "#PoreCheck"


def test_reports_a_genuinely_absent_token():
    [m] = check("Loved this one.", [_lit("Include #PoreCheck", "#PoreCheck")])
    assert m.present is False and m.matched_token is None


def test_matching_ignores_case_and_smart_punctuation():
    script = "I tried the TFS TEA TREE PORE AMPOULE and it’s great"
    [m] = check(script, _wrap("Say the full product name", "TFS Tea Tree Pore Ampoule"))
    assert m.present is True


def _wrap(text, *tokens):
    return [_lit(text, *tokens)]


def test_hash_prefix_is_significant():
    # "#PoreCheck" and "PoreCheck" are different mandatories; folding them
    # together would pass a script that never used the hashtag.
    [m] = check("talking about PoreCheck today", _wrap("Include #PoreCheck", "#PoreCheck"))
    assert m.present is False


def test_any_token_variant_satisfies_one_mandatory():
    mandatory = _lit("Name the product", "Tea Tree Pore Ampoule", "TFS Pore Ampoule")
    [m] = check("I used the TFS Pore Ampoule", [mandatory])
    assert m.present is True and m.matched_token == "TFS Pore Ampoule"


def test_judgement_mandatories_are_left_unsettled():
    [m] = check("anything", [Mandatory(text="Show texture", kind=JUDGEMENT, tokens=[])])
    assert m.present is None


def test_literal_without_tokens_degrades_to_judgement():
    # Unsearchable, so it must not be silently passed or failed.
    llm = ScriptedLLM({"mandatory_extraction": {"mandatories": [
        {"text": "Include the disclaimer", "kind": "literal", "tokens": []}]}})
    [m] = extract_mandatories(llm, brief="b")
    assert m.kind == JUDGEMENT


def test_prompt_block_marks_verified_versus_judgement():
    rendered = format_for_prompt(check("has #PoreCheck", [
        _lit("Include #PoreCheck", "#PoreCheck"),
        Mandatory(text="Show texture", kind=JUDGEMENT, tokens=[])]))
    assert "[VERIFIED] PRESENT" in rendered
    assert "[JUDGEMENT]" in rendered


def test_summary_separates_verified_present_from_missing():
    result = summary(check("has #PoreCheck", [
        _lit("Include #PoreCheck", "#PoreCheck"),
        _lit("Include @tfs", "@tfs")]))
    assert result["verified_present"] == ["Include #PoreCheck"]
    assert result["verified_missing"] == ["Include @tfs"]
    assert result["n_literal"] == 2


def test_scorer_cannot_report_a_miss_the_string_check_disproved():
    # The prompt forbids this; the code must enforce it regardless.
    mandatories = check("I love the TFS Tea Tree Pore Ampoule",
                        _wrap("Say the full product name at least once",
                              "TFS Tea Tree Pore Ampoule"))
    llm = ScriptedLLM({"brief_alignment": {
        "score": 7, "justification": "j",
        "dimensions": {"objective": 8, "audience": 8, "tone_and_voice": 8,
                       "key_message": 7, "mandatories_and_format": 5},
        "missing_mandatories": ["Full product name not stated at least once"],
        "strengths": [], "gaps": []}})
    result = score_brief_alignment(llm, brief="b", script="s", mandatories=mandatories)
    assert result["missing_mandatories"] == []
    assert result["overruled_mandatory_misses"]


def test_genuinely_missing_mandatory_survives_the_overrule_pass():
    mandatories = check("no hashtag here", _wrap("Include #PoreCheck", "#PoreCheck"))
    llm = ScriptedLLM({"brief_alignment": {
        "score": 5, "justification": "j",
        "dimensions": {"objective": 5, "audience": 5, "tone_and_voice": 5,
                       "key_message": 5, "mandatories_and_format": 3},
        "missing_mandatories": ["Include the hashtag #PoreCheck"],
        "strengths": [], "gaps": []}})
    result = score_brief_alignment(llm, brief="b", script="s", mandatories=mandatories)
    assert result["missing_mandatories"] == ["Include the hashtag #PoreCheck"]
    assert result["overruled_mandatory_misses"] == []


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  TFS   Tea\nTree  ") == "tfs tea tree"
