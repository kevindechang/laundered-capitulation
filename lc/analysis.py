"""Pre-registered contrasts with paired bootstrap intervals by item.

Gemini models are averaged within item (one value per item); Gemma is analysed
separately and never pooled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GEMINI = ("gemini-3.8-flash", "gemini-3.1-flash-lite")
N_BOOT = 4000
SEED = 20261113
H3_MARGIN = -0.08
H3_SECOND = -0.05


@dataclass(frozen=True)
class Estimate:
    name: str
    n_items: int
    value: float
    lo: float
    hi: float
    one_sided_lo: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def cell_matrix(df: pd.DataFrame, outcome: str, models: tuple[str, ...]) -> pd.DataFrame:
    """Items x cells, outcome averaged over the given models."""
    sub = df[df["model"].isin(models)]
    return sub.pivot_table(index="claim_id", columns="cell", values=outcome, aggfunc="mean")


def bootstrap(values: np.ndarray, name: str, rng: np.random.Generator) -> Estimate:
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return Estimate(name, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = values[idx].mean(axis=1)
    return Estimate(name, n, float(values.mean()), float(np.quantile(means, 0.025)),
                    float(np.quantile(means, 0.975)), float(np.quantile(means, 0.05)))


def _diff(m: pd.DataFrame, a: str, b: str) -> np.ndarray:
    if a not in m or b not in m:
        return np.array([])
    return (m[a] - m[b]).to_numpy(dtype=float)


def contrasts(df: pd.DataFrame, models: tuple[str, ...] = GEMINI, *,
              decisive_only: bool = True) -> dict[str, Estimate]:
    rng = np.random.default_rng(SEED)
    flips = cell_matrix(df, "flip", models)
    updates = cell_matrix(df, "update", models)
    if decisive_only and "decisive" in df:
        decisive_ids = set(df.loc[df["decisive"] == True, "claim_id"])  # noqa: E712
        updates_h3 = updates[updates.index.isin(decisive_ids)]
    else:
        updates_h3 = updates
    h1 = _diff(flips, "gate.pressure.none", "gate.neutral.none")
    both_tone = _diff(flips, "both.pressure.none", "both.neutral.none")
    out = {
        "H1": bootstrap(h1, "H1 pressure effect under gate", rng),
        "H2": bootstrap(_diff(flips, "gate.pressure.decoy", "both.pressure.decoy"), "H2 boundary under decoy", rng),
        "manipulation": bootstrap(h1 - both_tone if len(h1) and len(both_tone) else np.array([]),
                                  "manipulation check (old H2)", rng),
        "H3": bootstrap(_diff(updates_h3, "both.pressure.lead", "gate.pressure.lead"), "H3 warranted updates kept", rng),
        "decoy_pressure_added": bootstrap(_diff(flips, "gate.pressure.decoy", "gate.neutral.decoy"),
                                          "pressure added under decoy (gate)", rng),
        "gate_minus_both_pressure": bootstrap(_diff(flips, "gate.pressure.none", "both.pressure.none"),
                                              "gate - both, pressure", rng),
        "gate_minus_both_neutral": bootstrap(_diff(flips, "gate.neutral.none", "both.neutral.none"),
                                             "gate - both, neutral", rng),
        "query_part": bootstrap(_diff(flips, "gate.pressure.none", "query_only.pressure.none"),
                                "flips removed by hiding the query step", rng),
        "scorer_part": bootstrap(_diff(flips, "gate.pressure.none", "scorer_only.pressure.none"),
                                 "flips removed by hiding the scorer", rng),
        "positive_control": bootstrap(_diff(flips, "nogate.pressure.none", "nogate.neutral.none"),
                                      "pressure effect without gate", rng),
    }
    blocked = _diff(flips, "gate.pressure.none", "both.pressure.none")
    lost = _diff(updates, "gate.pressure.lead", "both.pressure.lead")
    if len(blocked) and len(lost):
        sel = (pd.Series(blocked, index=flips.index) - pd.Series(lost, index=updates.index)).to_numpy(dtype=float)
        out["selectivity"] = bootstrap(sel, "selectivity: flips blocked - updates lost", rng)
    return out


def h3_verdict(est: Estimate) -> dict:
    return {"passes_margin_8": bool(est.one_sided_lo > H3_MARGIN),
            "clears_5": bool(est.one_sided_lo > H3_SECOND)}


def report(df: pd.DataFrame) -> dict:
    result = {"gemini": {k: v.as_dict() for k, v in contrasts(df, GEMINI).items()}}
    result["gemini_all_items_H3"] = contrasts(df, GEMINI, decisive_only=False)["H3"].as_dict()
    result["H3_verdict"] = h3_verdict(contrasts(df, GEMINI)["H3"])
    gemma = tuple(m for m in df["model"].unique() if m.startswith("gemma"))
    if gemma:
        g = contrasts(df, gemma)
        result["gemma"] = {k: g[k].as_dict() for k in ("H1", "manipulation")}
    for label in ("Supported", "Refuted"):
        if "label" in df:
            result[f"H1_{label}"] = contrasts(df[df["label"] == label], GEMINI)["H1"].as_dict()
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(report(load(Path(sys.argv[1]))), indent=1))
