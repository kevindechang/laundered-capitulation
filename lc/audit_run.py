"""CLI for the sanitizer audit.

  python -m lc.audit_run --claims dev.json --ids-file audit_ids.txt --out audit.jsonl \
      --models gemini-3.8-flash,gemini-3.1-flash-lite,gemma-4-26b-a4b-it-maas --cap 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import audit
from .llm import Ledger, Metered, MockModel, VertexModel
from .mock import mock_policy

NO_LEAD_ONLY = ("gemma-4-26b-a4b-it-maas",)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lc.audit_run")
    ap.add_argument("--claims", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--ledger")
    ap.add_argument("--project")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    ledger = Ledger(args.cap, Path(args.ledger) if args.ledger else None)
    inner = MockModel(mock_policy(0.0)) if args.mock else VertexModel(args.project)
    llm = Metered(inner, ledger)
    raw = json.loads(Path(args.claims).read_text())
    wanted = set(Path(args.ids_file).read_text().split())
    claims = {c["claim_id"]: c for c in audit.audit_claims(raw) if c["claim_id"] in wanted}
    rng = random.Random(20261113)
    base = [m for c in claims.values() for m in audit.messages(c, rng)]
    rows = []
    for model in args.models.split(","):
        todo = [r for r in base if not (model in NO_LEAD_ONLY and r["lead"] == "lead")]
        rows += audit.run_sanitizer(llm, model, todo)
        print(f"sanitized {model}: {len(todo)} messages, spent=${ledger.spent:.4f}", flush=True)
    rows = audit.run_llm_probes(llm, rows, claims)
    audit.save(rows, Path(args.out))
    summary = audit.summarise(rows)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({m: s["pass_all"] for m, s in summary.items()}), f"spent=${ledger.spent:.4f}")


if __name__ == "__main__":
    main()
