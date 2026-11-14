"""Third rater family as a robustness check on citation quality (exploratory addendum).

The primary rule (lc.ratings) stays: weak only if both primary families say weak. With a third family:
(1) the third family alone; (2) all three families decide and agree on weak; (3) two of three: a case is
eligible when at least two families give a verdict, weak or warranted by strict majority of those verdicts,
and a 1-1 split between two families is excluded; (4) agreement between family verdicts (Krippendorff's
alpha, weak vs warranted, and pairwise agreement on cases both families decided). A family's verdict is
lc.ratings.family_verdict over its runs; abstentions are excluded from every denominator.
'decided' counts distinct away-from-gold cases; each share's own 'n' counts members (freq) or cases (once).

Usage:
  python -m lc.third_family members.jsonl --family claude=r1,r2,r3 --family grok=r1,r2 \
      --family glm=r1,r2 --primary claude,grok --third glm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc.ratings import check_runs, combine, family_verdict, krippendorff_alpha, load_runs, weak_share

Verdicts = dict[str, str | None]


def two_of_three(votes: list[str | None]) -> str | None:
    decided = [v for v in votes if v is not None]
    if len(decided) < 2:
        return None
    weak, good = decided.count("weak"), decided.count("warranted")
    return "weak" if weak > good else "warranted" if good > weak else None


def agreement(a: Verdicts, b: Verdicts, cases: list[str]) -> dict:
    both = [c for c in cases if a[c] is not None and b[c] is not None]
    return {"n": len(both), "agree": float(np.mean([a[c] == b[c] for c in both])) if both else float("nan")}


def robustness(votes: dict[str, Verdicts], primary: list[str], third: str, members: list[dict],
               *, seed: int, n_boot: int) -> dict:
    cases = sorted(votes[third])
    away = sorted({m["case"] for m in members if m["away"]})
    fams = [*primary, third]
    sets = {
        "primary": {c: combine([votes[f][c] for f in primary], "both") for c in cases},
        "third_only": votes[third],
        "all_three": {c: combine([votes[f][c] for f in fams], "both") for c in cases},
        "two_of_three": {c: two_of_three([votes[f][c] for f in fams]) for c in cases},
    }
    units = [[float(votes[f][c] == "weak") for f in fams if votes[f][c] is not None] for c in away]
    shares = {k: {w: weak_share(v, members, weighted=w == "freq", rng=np.random.default_rng(seed), n_boot=n_boot)
                  for w in ("freq", "once")} for k, v in sets.items()}
    return {
        "decided": {k: sum(v[c] is not None for c in away) for k, v in sets.items()},
        "third_abstentions": sum(votes[third][c] is None for c in away),
        "two_of_three_splits": sum(sum(votes[f][c] is not None for f in fams) >= 2
                                   and sets["two_of_three"][c] is None for c in away),
        "shares": shares,
        "alpha_binary_family_verdicts": krippendorff_alpha(units, "nominal"),
        "pairwise": {f"third_vs_{f}": agreement(votes[third], votes[f], away) for f in primary}
        | {"primary_pair": agreement(votes[primary[0]], votes[primary[1]], away)},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Third rater family robustness outputs (exploratory)")
    p.add_argument("members", type=Path, help="jsonl: case, claim_id, model, cell, away")
    p.add_argument("--family", action="append", required=True, help="name=run1.jsonl,run2.jsonl")
    p.add_argument("--primary", required=True, help="the two primary families, e.g. claude,grok")
    p.add_argument("--third", required=True, help="the added family")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    members = [json.loads(line) for line in args.members.read_text().splitlines() if line.strip()]
    case_ids = sorted({m["case"] for m in members})
    families, paths = {}, {}
    for spec in args.family:
        name, _, files = spec.partition("=")
        families[name] = [f"{name}{i + 1}" for i, _ in enumerate(files.split(","))]
        paths.update({run: Path(f) for run, f in zip(families[name], files.split(","))})
    primary = args.primary.split(",")
    if len(set(primary)) != 2 or args.third in primary or args.third not in families or not set(primary) <= set(families):
        raise SystemExit("--primary needs two different named families and --third one more named family")
    runs = load_runs(paths)
    check_runs(runs, set(case_ids))
    votes = {f: {c: family_verdict([runs[r][c] for r in names]) for c in case_ids} for f, names in families.items()}
    print(json.dumps(robustness(votes, primary, args.third, members, seed=args.seed, n_boot=args.n_boot), indent=1))


if __name__ == "__main__":
    main()
