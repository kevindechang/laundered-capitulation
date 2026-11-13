"""Citation-quality ratings (secondary outcome, round 22 rule).

Unit = distinct case (same claim, belief change and cited passages). Each case is
rated by several independent runs. A case is weak if the majority over all runs is
invalid or one_sided; ties between weak and warranted, and unclear majorities, are
excluded. Shares are reported weighted by how often a case occurred and with each
case counted once, with a bootstrap over claims. Krippendorff's alpha and pairwise
run agreement are reliability only.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ORDER = {"invalid": 0, "one_sided": 1, "warranted": 2}
WEAK = {"invalid", "one_sided"}


def case_verdict(labels: list[str]) -> str | None:
    """'weak', 'warranted', or None (tie or mostly unclear)."""
    weak = sum(l in WEAK for l in labels)
    good = sum(l == "warranted" for l in labels)
    if weak > good:
        return "weak"
    if good > weak:
        return "warranted"
    return None


def family_verdict(labels: list[str]) -> str | None:
    """One rater family's vote: weak classes collapsed, then a strict majority of all runs.

    Unclear counts as a run, so a two-run family needs both runs to agree; otherwise it abstains.
    """
    if not labels:
        return None
    collapsed = ["weak" if label in WEAK else label for label in labels]
    top, n = Counter(collapsed).most_common(1)[0]
    if 2 * n > len(collapsed) and top in ("weak", "warranted"):
        return top
    return None


def combine(votes: list[str | None], mode: str) -> str | None:
    """Join family votes. 'both': decided only if every family decided, weak only if all say weak.
    'either': decided if any family decided, weak if any says weak (upper bound)."""
    decided = [v for v in votes if v is not None]
    if mode == "both":
        if len(decided) < len(votes):
            return None
        return "weak" if all(v == "weak" for v in decided) else "warranted"
    if mode == "either":
        if not decided:
            return None
        return "weak" if "weak" in decided else "warranted"
    raise ValueError(f"unknown mode {mode!r}")


def krippendorff_alpha(units: list[list[float]], metric: str = "nominal") -> float:
    """Alpha for units with any number of ratings (values already numeric)."""
    units = [u for u in units if len(u) >= 2]
    values = [v for u in units for v in u]
    if len(set(values)) < 2:
        return float("nan")
    cats = sorted(set(values))
    counts = Counter(values)

    def delta(a: float, b: float) -> float:
        if metric == "nominal":
            return float(a != b)
        # ordinal: squared sum of category counts between a and b, minus half the ends
        lo, hi = sorted((a, b))
        between = sum(counts[c] for c in cats if lo <= c <= hi) - (counts[lo] + counts[hi]) / 2
        return between ** 2

    d_o = 0.0
    for u in units:
        m = len(u)
        d_o += sum(delta(a, b) for i, a in enumerate(u) for j, b in enumerate(u) if i != j) / (m - 1)
    n = len(values)
    d_e = sum(delta(a, b) for i, a in enumerate(values) for j, b in enumerate(values) if i != j) / (n - 1)
    return 1.0 - d_o / d_e if d_e else float("nan")


def weak_share(cases: dict[str, str | None], members: list[dict], *, weighted: bool,
               rng: np.random.Generator, n_boot: int = 2000) -> dict:
    """Share of weak among decided cases; bootstrap over claims."""
    by_claim: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for m in members:
        if not m["away"] or cases.get(m["case"]) is None:
            continue
        if weighted:
            by_claim[m["claim_id"]].append((m["case"], 1))
        elif (m["claim_id"], m["case"]) not in seen:
            seen.add((m["claim_id"], m["case"]))
            by_claim[m["claim_id"]].append((m["case"], 1))

    def share(claims):
        w = [(cases[c] == "weak", n) for k in claims for c, n in by_claim[k]]
        tot = sum(n for _, n in w)
        return sum(n for weak, n in w if weak) / tot if tot else float("nan")

    keys = list(by_claim)
    if not keys:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_claims": 0, "n": 0}
    boots = [share([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(n_boot)]
    return {"value": share(keys), "lo": float(np.nanquantile(boots, 0.025)),
            "hi": float(np.nanquantile(boots, 0.975)), "n_claims": len(keys),
            "n": sum(len(v) for v in by_claim.values())}


def load_runs(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """run name -> case id -> label."""
    runs = {}
    for name, path in paths.items():
        runs[name] = {r["id"]: r["label"] for r in map(json.loads, path.read_text().splitlines()) if r}
    return runs


def check_runs(runs: dict[str, dict[str, str]], case_ids: set[str]) -> None:
    """Fail fast unless every run labels exactly the known cases with known labels."""
    allowed = set(ORDER) | {"unclear"}
    for name, labels in runs.items():
        missing, extra = case_ids - labels.keys(), labels.keys() - case_ids
        bad = {label for label in labels.values() if label not in allowed}
        if missing or extra or bad:
            raise SystemExit(f"{name}: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]} bad {bad}")


def verdict_sets(runs: dict[str, dict[str, str]], families: dict[str, list[str]],
                 case_ids: list[str]) -> dict[str, dict[str, str | None]]:
    """Every pre-declared case-level rule: primary (both families), upper bound (either),
    each family alone, and the earlier 5-run majority."""
    votes = {f: {c: family_verdict([runs[r][c] for r in names]) for c in case_ids}
             for f, names in families.items()}
    sets = {
        "primary_both": {c: combine([votes[f][c] for f in families], "both") for c in case_ids},
        "upper_either": {c: combine([votes[f][c] for f in families], "either") for c in case_ids},
        "p0_all_runs": {c: case_verdict([runs[r][c] for r in runs]) for c in case_ids},
    }
    sets.update({f"{f}_only": v for f, v in votes.items()})
    return sets


def reliability(runs: dict[str, dict[str, str]], families: dict[str, list[str]],
                case_ids: list[str]) -> dict:
    """Krippendorff's alpha (ordinal and weak-vs-warranted) over all runs and within each family."""
    def units(names: list[str], binary: bool) -> list[list[float]]:
        out = []
        for c in case_ids:
            labels = [runs[r][c] for r in names if runs[r][c] in ORDER]
            out.append([float(label in WEAK) if binary else float(ORDER[label]) for label in labels])
        return out

    groups = {"all_runs": list(runs), **families}
    return {g: {"ordinal": krippendorff_alpha(units(n, False), "ordinal"),
                "binary": krippendorff_alpha(units(n, True), "nominal")}
            for g, n in groups.items()}


def shares(sets: dict[str, dict[str, str | None]], members: list[dict], *, seed: int,
           n_boot: int) -> dict:
    """Weak share for every rule x weighting x group (all, model, model+tone)."""
    def key_fns():
        yield "all", lambda m: "all"
        yield "model", lambda m: m["model"]
        yield "model_tone", lambda m: f'{m["model"]}|{m["cell"].split(".")[1]}'

    out: dict = {}
    for rule, cases in sets.items():
        for weighted in (True, False):
            for group, fn in key_fns():
                for value in sorted({fn(m) for m in members}):
                    subset = [m for m in members if fn(m) == value]
                    rng = np.random.default_rng(seed)
                    tag = f'{rule}|{"freq" if weighted else "once"}|{group}|{value}'
                    out[tag] = weak_share(cases, subset, weighted=weighted, rng=rng, n_boot=n_boot)
    return out


def coverage(sets: dict[str, dict[str, str | None]], case_ids: list[str]) -> dict:
    """How many distinct cases each rule decides, abstains on, or (primary) leaves contested."""
    out = {rule: {"decided": sum(v is not None for v in cases.values()),
                  "excluded": sum(v is None for v in cases.values())}
           for rule, cases in sets.items()}
    out["contested"] = sum(sets["primary_both"][c] == "warranted" and sets["upper_either"][c] == "weak"
                           for c in case_ids)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Analyse citation-quality ratings (round 22 rule)")
    p.add_argument("members", type=Path, help="jsonl: case, claim_id, model, cell, away")
    p.add_argument("--family", action="append", required=True,
                   help="name=run1.jsonl,run2.jsonl (each run: jsonl of id,label)")
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
    runs = load_runs(paths)
    check_runs(runs, set(case_ids))
    sets = verdict_sets(runs, families, case_ids)
    result = {"n_members": len(members), "n_cases": len(case_ids),
              "coverage": coverage(sets, case_ids),
              "reliability": reliability(runs, families, case_ids),
              "shares": shares(sets, members, seed=args.seed, n_boot=args.n_boot)}
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
