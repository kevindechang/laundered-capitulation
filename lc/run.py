"""Command line: prepare items (judge calls, frozen) and run cells.

  python -m lc.run prepare --claims dev.json --out prep.jsonl [--ids 1,2] [--mock]
  python -m lc.run run --claims dev.json --prep prep.jsonl --out results.jsonl --models a,b [--mock]

Items run in random order with all of an item's cells together, so stopping
early leaves a valid sample. Finished (claim, model, cell) rows are skipped on
restart.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts as P
from .agent import ItemContext, run_conversation
from .cells import cells_for
from .llm import BudgetExceeded, Ledger, Metered, MockModel, VertexModel, parse_json
from .mock import mock_policy
from .prepare import prepare_claim
from .store import Passage, iter_remote_claim_lines, load_claims, load_sites

GEMMA = ("gemma-4-26b-a4b-it-maas",)
PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"


def build_llm(args) -> Metered:
    ledger = Ledger(args.cap, Path(args.ledger) if args.ledger else None)
    if args.mock:
        return Metered(MockModel(mock_policy(args.mock_effect)), ledger)
    import os
    project = args.project or os.environ.get(PROJECT_ENV)
    if not project:
        sys.exit(f"set --project or {PROJECT_ENV}")
    return Metered(VertexModel(project, flex=args.flex), ledger)


def _done(path: Path, key) -> set:
    if not path.exists():
        return set()
    return {key(json.loads(l)) for l in path.read_text().splitlines() if l.strip()}


def cmd_prepare(args) -> None:
    llm = build_llm(args)
    claims = load_claims(Path(args.claims))
    if args.ids:
        wanted = set(args.ids.split(","))
        claims = [c for c in claims if c.claim_id in wanted]
    out = Path(args.out)
    done = _done(out, lambda r: r["claim_id"])
    todo = [c for c in claims if c.claim_id not in done]
    by_id = {c.claim_id: c for c in todo}
    sites = load_sites()
    lines_source = (args.local_store and _local_lines(args.local_store, by_id)) or iter_remote_claim_lines(by_id)
    for cid, lines in lines_source:
        try:
            prepared, index = prepare_claim(llm, by_id[cid], lines, sites)
        except BudgetExceeded as exc:
            print(f"stop: {exc}")
            break
        if index:
            index.close()
        with out.open("a") as fh:
            fh.write(json.dumps(prepared.meta) + "\n")
        print(f"prepared {cid} eligible={prepared.meta['eligible']} spent=${llm.ledger.spent:.4f}", flush=True)


def _local_lines(folder: str, by_id: dict):
    for cid in by_id:
        path = Path(folder) / f"{cid}.json"
        yield cid, path.read_text().split("\n") if path.exists() else []


def _item_from_meta(meta: dict, claim, passages: dict[str, Passage]) -> ItemContext:
    return ItemContext(claim.claim_id, claim.text, claim.side, passages[meta["seed_right"]],
                       passages[meta["seed_wrong"]], meta["lead"], meta["decoy_pointer"])


def closed_book(llm, model: str, claim_text: str) -> str | None:
    reply = llm.complete(model, "", P.CLOSED_BOOK.format(claim=claim_text), max_tokens=256,
                         as_json=True, tag="closed_book")
    return parse_json(reply.text).get("belief")


def cmd_run(args) -> None:
    from .prepare import Prepared  # noqa: F401  (kept for type readers)
    from .store import filter_pages, make_passages, parse_store_lines, SearchIndex

    llm = build_llm(args)
    claims = {c.claim_id: c for c in load_claims(Path(args.claims))}
    metas = [json.loads(l) for l in Path(args.prep).read_text().splitlines() if l.strip()]
    metas = [m for m in metas if m.get("eligible")]
    if args.ids:
        wanted = set(args.ids.split(","))
        metas = [m for m in metas if m["claim_id"] in wanted]
    random.Random(args.seed).shuffle(metas)
    if args.limit:
        metas = metas[: args.limit]
    stage2_ids = set(args.stage2_ids.split(",")) if args.stage2_ids else set()
    models = args.models.split(",")
    out = Path(args.out)
    done = _done(out, lambda r: (r["claim_id"], r["model"], r["cell"]))
    sites = load_sites()
    meta_by_id = {m["claim_id"]: m for m in metas}
    order = [m["claim_id"] for m in metas]
    source = (args.local_store and _local_lines(args.local_store, {i: 1 for i in order})) or \
        iter_remote_claim_lines(order)
    for cid, lines in source:
        meta, claim = meta_by_id[cid], claims[cid]
        pages = filter_pages(parse_store_lines(lines), claim, sites)
        passages = make_passages(pages)
        index = SearchIndex(passages, check_same_thread=False)
        item = _item_from_meta(meta, claim, {p.pid: p for p in passages})
        jobs = []
        for model in models:
            for cell in cells_for(model, stage2=cid in stage2_ids, has_decoy=bool(meta.get("decoy_pointer")),
                                  neutral_decoy=args.neutral_decoy, gemma_models=GEMMA):
                if (cid, model, cell.name) not in done:
                    jobs.append((model, cell))
        try:
            priors = {m: closed_book(llm, m, claim.text) for m in models} if jobs else {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(lambda mc: run_conversation(
                    llm, mc[0], item, mc[1], index, with_reply=not args.no_reply), jobs))
        except BudgetExceeded as exc:
            print(f"stop at claim {cid}: {exc}")
            index.close()
            break
        index.close()
        with out.open("a") as fh:
            for row in rows:
                row.update(prior=priors.get(row["model"]), label=claim.label,
                           decisive=meta.get("decisive"), reannotated=claim.reannotated)
                fh.write(json.dumps(row) + "\n")
        print(f"ran {cid}: {len(rows)} cells, spent=${llm.ledger.spent:.4f}, calls={llm.ledger.calls}, "
              f"failed={llm.ledger.failures}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lc.run")
    ap.add_argument("command", choices=["prepare", "run"])
    ap.add_argument("--claims", required=True)
    ap.add_argument("--prep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--models", default="gemini-3.8-flash,gemini-3.1-flash-lite")
    ap.add_argument("--cap", type=float, default=1.0, help="hard USD cap for this command")
    ap.add_argument("--ledger")
    ap.add_argument("--project")
    ap.add_argument("--flex", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-effect", type=float, default=0.15)
    ap.add_argument("--local-store")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--stage2-ids")
    ap.add_argument("--neutral-decoy", action="store_true")
    ap.add_argument("--no-reply", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    (cmd_prepare if args.command == "prepare" else cmd_run)(args)


if __name__ == "__main__":
    main()
