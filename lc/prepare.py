"""Per-claim preparation: filtered passages, seeds, lead pointer and decoy pointer.

Judge calls here (decisiveness, passage stance, decoy question) are made blind
to every arm result and are frozen to a JSON file before any stage-1 run.

Choices where the plan is silent (for the lock check):
- A correct seed rests on a gold-page passage the judge puts on the correct side.
- A wrong seed rests on the decoy passage (judged false-side); with none, it
  rests on no recorded evidence, and results are stratified by that.
- Decoys come from claim-query ranks 6-30, so a plain claim query (top 5) does
  not retrieve them without the pointer.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts as P
from .agent import ItemContext
from .llm import parse_json
from .store import Claim, Page, Passage, SearchIndex, filter_pages, make_passages, norm_url, parse_store_lines

JUDGE = "gemini-3.5-flash-lite"
PAGE_CHARS = 12000
DECOY_RANKS = (5, 30)
DECOY_CHECKS = 8
SEED_CHECKS = 3


@dataclass(frozen=True)
class Prepared:
    claim: Claim
    passages: tuple[Passage, ...]
    item: ItemContext | None  # None when the claim is not eligible
    meta: dict


def _judge(llm, prompt: str, tag: str, as_json: bool = True, max_tokens: int = 512):
    return llm.complete(JUDGE, "", prompt, max_tokens=max_tokens, as_json=as_json, tag=f"judge:{tag}")


def _stance(llm, claim: Claim, passage: Passage) -> str | None:
    reply = _judge(llm, P.JUDGE_STANCE.format(claim=claim.text, passage=passage.text), "stance")
    return None if reply.failed else parse_json(reply.text).get("stance")


def pick_lead(llm, claim: Claim, pages: list[Page]) -> tuple[str | None, bool, list[dict]]:
    """First gold question whose page the judge confirms as decisive; else first with text."""
    by_url = {norm_url(p.url): p for p in pages if p.is_gold and p.sentences}
    checks, fallback = [], None
    for q in claim.questions:
        page = next((by_url[norm_url(u)] for u in q.urls if norm_url(u) in by_url), None)
        if page is None:
            continue
        fallback = fallback or q.question
        reply = _judge(llm, P.JUDGE_DECISIVE.format(
            claim=claim.text, page=" ".join(page.sentences)[:PAGE_CHARS]), "decisive")
        verdict = {} if reply.failed else parse_json(reply.text)
        decisive = verdict.get("decisive") is True and verdict.get("side") == claim.side
        checks.append({"question": q.question, "url": page.url, "decisive": decisive,
                       "judge_side": verdict.get("side"), "judge_failed": reply.failed or not verdict})
        if decisive:
            return q.question, True, checks
    return fallback, False, checks


def pick_seed_right(llm, claim: Claim, index: SearchIndex, passages: list[Passage]) -> tuple[Passage, bool]:
    gold = [p for p in index.search(claim.text, k=50) if p.is_gold] or [p for p in passages if p.is_gold]
    for passage in gold[:SEED_CHECKS]:
        if _stance(llm, claim, passage) == claim.side:
            return passage, True
    return gold[0], False


def pick_decoy(llm, claim: Claim, index: SearchIndex) -> tuple[Passage | None, str | None, list[dict]]:
    """First non-gold passage at claim-query ranks 6-30 the judge puts on the false side."""
    false_side = "false" if claim.side == "true" else "true"
    lo, hi = DECOY_RANKS
    candidates = [p for p in index.search(claim.text, k=hi)[lo:] if not p.is_gold]
    checks = []
    for passage in candidates[:DECOY_CHECKS]:
        stance = _stance(llm, claim, passage)
        checks.append({"pid": passage.pid, "stance": stance})
        if stance == false_side:
            reply = _judge(llm, P.JUDGE_POINTER.format(passage=passage.text), "pointer", as_json=False)
            lines = [] if reply.failed else [l for l in reply.text.strip().splitlines() if l.strip()]
            if lines:
                return passage, lines[0].strip(), checks
    return None, None, checks


def prepare_claim(llm, claim: Claim, lines: list[str], sites: frozenset[str]) -> tuple[Prepared, SearchIndex | None]:
    raw = parse_store_lines(lines)
    pages = filter_pages(raw, claim, sites)
    meta = {"claim_id": claim.claim_id, "pages_raw": len(raw), "pages_kept": len(pages),
            "label": claim.label, "reannotated": claim.reannotated}
    if not any(p.is_gold and p.sentences for p in pages):
        return Prepared(claim, (), None, {**meta, "eligible": False, "why": "no gold page with text"}), None
    passages = make_passages(pages)
    index = SearchIndex(passages)
    lead, decisive, lead_checks = pick_lead(llm, claim, pages)
    if lead is None:
        index.close()
        return Prepared(claim, (), None, {**meta, "eligible": False, "why": "no gold question matches a page",
                                          "lead_checks": lead_checks}), None
    seed_right, seed_right_checked = pick_seed_right(llm, claim, index, passages)
    decoy, decoy_pointer, decoy_checks = pick_decoy(llm, claim, index)
    item = ItemContext(claim.claim_id, claim.text, claim.side, seed_right, decoy, lead, decoy_pointer,
                       decoy.pid if decoy else None)
    meta.update(eligible=True, n_passages=len(passages), seed_right=seed_right.pid,
                seed_right_checked=seed_right_checked, seed_wrong=decoy.pid if decoy else None,
                lead=lead, decisive=decisive, lead_checks=lead_checks,
                decoy_pid=decoy.pid if decoy else None, decoy_pointer=decoy_pointer,
                decoy_checks=decoy_checks)
    return Prepared(claim, tuple(passages), item, meta), index
