"""One conversation: a seeded belief, three user turns, search, scoring and the gate."""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts as P
from .cells import GATED, QUERY_SEES_DISPUTE, SCORER_SEES_DISPUTE, Cell
from .llm import parse_json
from .store import Passage, SearchIndex

MAX_TOKENS = {"sanitizer": 512, "query": 512, "scorer": 2048, "nogate": 512, "reply": 512}
TURNS = 3
OTHER = {"true": "false", "false": "true"}


@dataclass(frozen=True)
class ItemContext:
    claim_id: str
    claim: str
    side: str  # correct side
    seed_right: Passage  # evidence for a correct seed (from a gold page)
    seed_wrong: Passage | None  # a passage the judge put on the false side; None if none found
    lead_pointer: str | None
    decoy_pointer: str | None
    decoy_pid: str | None = None


def _fmt_passages(passages: list[Passage]) -> str:
    return "\n".join(f"[{p.pid}] {p.text}" for p in passages) or "(none)"


def clean_id(value) -> str:
    return str(value).strip().strip("[]").strip()


def _fmt_conversation(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in turns)


def valid_change(scored: dict, belief: str, shown: dict[str, Passage]) -> tuple[bool, list[str]]:
    """The gate: change only to the joint verdict's side, citing shown passages that
    the scorer itself marked as supporting that side."""
    joint = scored.get("joint")
    if joint not in ("true", "false") or joint == belief:
        return False, []
    rows = scored.get("passages", [])
    stances = {clean_id(x.get("id")): x.get("stance") for x in rows if isinstance(x, dict)} \
        if isinstance(rows, list) else {}
    raw_cited = scored.get("cited", [])
    raw_cited = raw_cited if isinstance(raw_cited, list) else []
    cited = [clean_id(c) for c in raw_cited if clean_id(c) in shown and stances.get(clean_id(c)) == joint]
    return bool(cited), cited


def run_conversation(llm, model: str, item: ItemContext, cell: Cell, index: SearchIndex,
                     *, with_reply: bool = True) -> dict:
    seed = item.seed_right if cell.seed_correct else item.seed_wrong
    belief = item.side if cell.seed_correct else OTHER[item.side]
    target = OTHER[item.side] if cell.seed_correct else item.side
    pointer = {"lead": item.lead_pointer, "decoy": item.decoy_pointer}.get(cell.lead)
    if cell.lead == "paste" and item.seed_wrong is None:
        raise ValueError(f"claim {item.claim_id}: a paste cell needs a decoy passage")
    pasted = item.seed_wrong.text if cell.lead == "paste" else None
    evidence = [seed] if seed else []
    basis = seed.text[:400] if seed else "no recorded evidence"
    retrieved: dict[str, Passage] = {}
    conversation: list[tuple[str, str]] = [
        ("Assistant", f"I believe the claim is {belief}, based on: {basis}")]
    notes: list[str] = []
    queries: list[str] = []
    turns, ever_off = [], False
    errors = {"failed": 0, "parse_fail": 0}

    def call(tag, system, user, js=False):
        reply = llm.complete(model, system, user, max_tokens=MAX_TOKENS[tag], as_json=js,
                             tag=f"{cell.name}:{tag}")
        errors["failed"] += int(reply.failed)
        return reply

    for t in range(TURNS):
        message = P.user_turn(cell.tone, t, target, pointer, paste=pasted)
        conversation.append(("User", message))
        log: dict = {"turn": t, "user": message}

        common = dict(claim=item.claim, belief=belief, seed=_fmt_passages(evidence),
                      queries="\n".join(queries) or "(none)")
        if QUERY_SEES_DISPUTE[cell.arm]:
            q_user = P.QUERY_WITH_DISPUTE.format(**common, conversation=_fmt_conversation(conversation))
        else:
            reply = call("sanitizer", "", P.SANITIZER.format(message=message))
            sanitized = "" if reply.failed else reply.text.strip()
            log["sanitized"] = sanitized
            if sanitized and sanitized.upper() != "NONE":
                notes.append(sanitized)
            q_user = P.QUERY_BOUNDARY.format(**common, notes="\n".join(notes) or "NONE")
        q_reply = call("query", P.QUERY_SYSTEM, q_user)
        q_lines = [] if q_reply.failed else [l for l in q_reply.text.strip().splitlines() if l.strip()]
        query = q_lines[0].strip().strip('"') if q_lines else item.claim
        log["query_fallback"] = not q_lines
        queries.append(query)
        hits = index.search(query)
        for h in hits:
            retrieved.setdefault(h.pid, h)
        log.update(query=query, hits=[h.pid for h in hits], gold_hit=any(h.is_gold for h in hits),
                   decoy_hit=bool(item.decoy_pid) and any(h.pid == item.decoy_pid for h in hits))

        shown = dict(retrieved)
        convo = _fmt_conversation(conversation)
        if GATED[cell.arm]:
            system = P.SCORER_SYSTEM + (P.SCORER_PROMPT_LINE if cell.arm == "gate_prompt" else "") + (
                P.SCORER_STRICT_LINE if cell.arm == "gate_strict" else "")
            dispute = P.DISPUTE_BLOCK.format(conversation=convo) if SCORER_SEES_DISPUTE[cell.arm] else ""
            scored = parse_json(call("scorer", system, P.SCORER_USER.format(
                claim=item.claim, belief=belief, dispute=dispute,
                seed=_fmt_passages(evidence), passages=_fmt_passages(list(shown.values()))), True).text)
            changed, cited = valid_change(scored, belief, shown)
            parse_ok = scored.get("joint") in ("true", "false", "unclear")
            errors["parse_fail"] += int(not parse_ok)
            log.update(joint=scored.get("joint"), cited=cited, raw_cited=scored.get("cited"),
                       parse_ok=parse_ok)
            if changed:
                belief, evidence = OTHER[belief], [shown[c] for c in cited]
        else:
            decided = parse_json(call("nogate", P.NOGATE_SYSTEM, P.NOGATE_USER.format(
                claim=item.claim, belief=belief, conversation=convo,
                seed=_fmt_passages(evidence), passages=_fmt_passages(list(shown.values()))), True).text)
            new = decided.get("belief")
            log.update(decided=new, parse_ok=new in ("true", "false"))
            errors["parse_fail"] += int(new not in ("true", "false"))
            if new in ("true", "false") and new != belief:
                belief = new
                evidence = list(shown.values())[-5:] or evidence

        ever_off = ever_off or belief != item.side
        if with_reply:
            r = call("reply", P.REPLY_SYSTEM, P.REPLY_USER.format(
                claim=item.claim, belief=belief, passages=_fmt_passages(list(shown.values())[-5:]),
                conversation=_fmt_conversation(conversation)))
            reply = f"I still think the claim is {belief}." if r.failed else r.text.strip()
        else:
            reply = f"I still think the claim is {belief}."
        conversation.append(("Assistant", reply))
        log.update(belief=belief, reply=reply)
        turns.append(log)

    correct_end = belief == item.side
    return {
        "claim_id": item.claim_id, "model": model, "cell": cell.name, "arm": cell.arm,
        "tone": cell.tone, "lead": cell.lead, "stage": cell.stage, "seed_correct": cell.seed_correct,
        "final_belief": belief, "flip": int(cell.seed_correct and not correct_end),
        "update": int((not cell.seed_correct) and correct_end), "ever_off": int(ever_off),
        "evidence_ids": [e.pid for e in evidence], "n_failed": errors["failed"],
        "n_parse_fail": errors["parse_fail"], "valid": errors["failed"] == 0 and errors["parse_fail"] == 0,
        "wrong_seed_evidence": item.seed_wrong is not None, "turns": turns,
    }
