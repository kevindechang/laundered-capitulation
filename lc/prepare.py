"""Per-claim preparation: filtered passages, seeds, lead pointer and decoy pointer.

Judge calls here (decisiveness, passage stance, decoy question) are made blind
to every arm result and are frozen to a JSON file before any stage-1 run.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts as P
from .agent import ItemContext
from .llm import parse_json
from .store import Claim, Page, Passage, SearchIndex, filter_pages, make_passages, parse_store_lines, unwrap

JUDGE = "gemini-3.5-flash-lite"
PAGE_CHARS = 12000
DECOY_CANDIDATES = 5


@dataclass(frozen=True)
class Prepared:
    claim: Claim
    passages: tuple[Passage, ...]
    item: ItemContext | None  # None when the claim is not eligible
    meta: dict


def _judge(llm, prompt: str, tag: str, as_json: bool = True, max_tokens: int = 256):
    return llm.complete(JUDGE, "", prompt, max_tokens=max_tokens, as_json=as_json, tag=f"judge:{tag}")


def pick_lead(llm, claim: Claim, pages: list[Page]) -> tuple[str | None, bool, list[dict]]:
    """First gold question whose page the judge confirms as decisive; else first with text."""
    by_url = {unwrap(p.url).rstrip("/"): p for p in pages if p.is_gold and p.sentences}
    checks, fallback = [], None
    for q in claim.questions:
        page = next((by_url[unwrap(u).rstrip("/")] for u in q.urls if unwrap(u).rstrip("/") in by_url), None)
        if page is None:
            continue
        fallback = fallback or q.question
        verdict = parse_json(_judge(llm, P.JUDGE_DECISIVE.format(
            claim=claim.text, page=" ".join(page.sentences)[:PAGE_CHARS]), "decisive").text)
        decisive = verdict.get("decisive") is True and verdict.get("side") == claim.side
        checks.append({"question": q.question, "url": page.url, "decisive": decisive,
                       "judge_side": verdict.get("side")})
        if decisive:
            return q.question, True, checks
    return fallback, False, checks


def pick_decoy(llm, claim: Claim, index: SearchIndex) -> tuple[str | None, str | None, list[dict]]:
    """First top-ranked non-gold passage the judge says supports the false side."""
    false_side = "false" if claim.side == "true" else "true"
    checks = []
    candidates = [p for p in index.search(claim.text, k=4 * DECOY_CANDIDATES) if not p.is_gold]
    for passage in candidates[:DECOY_CANDIDATES]:
        stance = parse_json(_judge(llm, P.JUDGE_STANCE.format(
            claim=claim.text, passage=passage.text), "stance").text).get("stance")
        checks.append({"pid": passage.pid, "stance": stance})
        if stance == false_side:
            question = _judge(llm, P.JUDGE_POINTER.format(passage=passage.text), "pointer",
                              as_json=False).text.strip().splitlines()
            return passage.pid, (question[0] if question else None), checks
    return None, None, checks


def prepare_claim(llm, claim: Claim, lines: list[str], sites: frozenset[str]) -> tuple[Prepared, SearchIndex | None]:
    raw = parse_store_lines(lines)
    pages = filter_pages(raw, claim, sites)
    meta = {"claim_id": claim.claim_id, "pages_raw": len(raw), "pages_kept": len(pages),
            "label": claim.label, "reannotated": claim.reannotated}
    gold_with_text = [p for p in pages if p.is_gold and p.sentences]
    if not gold_with_text:
        return Prepared(claim, (), None, {**meta, "eligible": False, "why": "no gold page with text"}), None
    passages = make_passages(pages)
    index = SearchIndex(passages)
    gold_hits = [p for p in index.search(claim.text, k=50) if p.is_gold]
    other_hits = [p for p in index.search(claim.text, k=50) if not p.is_gold]
    gold_passages = [p for p in passages if p.is_gold]
    seed_right = (gold_hits or gold_passages)[0]
    seed_wrong = (other_hits or [p for p in passages if not p.is_gold] or [seed_right])[0]
    lead, decisive, lead_checks = pick_lead(llm, claim, pages)
    decoy_pid, decoy_pointer, decoy_checks = pick_decoy(llm, claim, index)
    if decoy_pid:  # a wrong seed rests on a passage the judge put on the false side
        seed_wrong = next(p for p in passages if p.pid == decoy_pid)
    item = ItemContext(claim.claim_id, claim.text, claim.side, seed_right, seed_wrong, lead, decoy_pointer)
    meta.update(eligible=True, n_passages=len(passages), seed_right=seed_right.pid,
                seed_wrong=seed_wrong.pid, lead=lead, decisive=decisive, lead_checks=lead_checks,
                decoy_pid=decoy_pid, decoy_pointer=decoy_pointer, decoy_checks=decoy_checks)
    return Prepared(claim, tuple(passages), item, meta), index
