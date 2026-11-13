"""Exploratory stricter-scorer addendum: E1, E2, selectivity, route drift and the declared reading.

Not part of the pre-registered analysis. The new arm is compared with a plain-gate control run at the
same time on the same route; drift against the original gate cells is reported, not used.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .analysis import H3_MARGIN, bootstrap, cell_matrix, load, valid_only

SEED = 20261114
STRICT, CONTROL = "gate_strict.neutral", "gate.neutral"


def _paired(a: pd.DataFrame, ca: str, b: pd.DataFrame, cb: str) -> pd.Series:
    """Per-claim a[ca] - b[cb], aligned on claim id (claims missing from either side drop out)."""
    if ca not in a or cb not in b:
        return pd.Series(dtype=float)
    return (a[ca] - b[cb]).dropna()


def reading(e1: dict, e2: dict, sel: dict) -> dict:
    """Round 23 rule: helps only if E1's CI is below 0, E2's lower bound is above the margin and
    selectivity is positive. E2 is harmful only if its whole CI is below the margin."""
    if e2["lo"] > H3_MARGIN:
        updates = "non-inferior"
    elif e2["hi"] < H3_MARGIN:
        updates = "harmful"
    else:
        updates = "inconclusive"
    helps = bool(e1["hi"] < 0 and e2["lo"] > H3_MARGIN and sel["value"] > 0)
    return {"helps": helps, "warranted_updates": updates, "exploratory": True}


def strict_report(df: pd.DataFrame, *, model: str, original: pd.DataFrame | None = None,
                  original_model: str | None = None) -> dict:
    df, invalid = valid_only(df)
    rng = np.random.default_rng(SEED)
    flips, updates = cell_matrix(df, "flip", (model,)), cell_matrix(df, "update", (model,))
    decisive = set(df.loc[df["decisive"] == True, "claim_id"]) if "decisive" in df else set(updates.index)  # noqa: E712
    upd_d = updates[updates.index.isin(decisive)]
    blocked = -_paired(flips, f"{STRICT}.none", flips, f"{CONTROL}.none")
    lost_d = -_paired(upd_d, f"{STRICT}.lead", upd_d, f"{CONTROL}.lead")
    lost_all = -_paired(updates, f"{STRICT}.lead", updates, f"{CONTROL}.lead")
    est = {
        "E1": bootstrap(-blocked.to_numpy(float), "E1 flips: strict - control", rng),
        "E2": bootstrap(-lost_d.to_numpy(float), "E2 warranted updates (decisive leads): strict - control", rng),
        "selectivity": bootstrap((blocked - lost_d).dropna().to_numpy(float),
                                 "selectivity (decisive leads): flips blocked - updates lost", rng),
        "selectivity_all_leads": bootstrap((blocked - lost_all).dropna().to_numpy(float),
                                           "selectivity (all lead claims, as the main analysis)", rng),
        "control_flip_rate": bootstrap(flips[f"{CONTROL}.none"].dropna().to_numpy(float), "control flip rate", rng),
        "strict_flip_rate": bootstrap(flips[f"{STRICT}.none"].dropna().to_numpy(float), "strict flip rate", rng),
    }
    out = {k: v.as_dict() for k, v in est.items()} | {"invalid": invalid}
    out["reading"] = reading(out["E1"], out["E2"], out["selectivity"])
    if original is not None and original_model:
        o, _ = valid_only(original)
        of, ou = cell_matrix(o, "flip", (original_model,)), cell_matrix(o, "update", (original_model,))
        out["drift_flip"] = bootstrap(_paired(flips, f"{CONTROL}.none", of, f"{CONTROL}.none").to_numpy(float),
                                      "drift: control rerun - original gate, flips", rng).as_dict()
        out["drift_update"] = bootstrap(_paired(updates, f"{CONTROL}.lead", ou, f"{CONTROL}.lead").to_numpy(float),
                                        "drift: control rerun - original gate, updates", rng).as_dict()
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Exploratory stricter-scorer analysis")
    p.add_argument("results")
    p.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--original", help="original run results, for drift")
    p.add_argument("--original-model", default="accounts/fireworks/models/deepseek-v4p1-flash")
    args = p.parse_args(argv)
    original = load(args.original) if args.original else None
    json.dump(strict_report(load(args.results), model=args.model, original=original,
                            original_model=args.original_model), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
