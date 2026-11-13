"""Simulated results with known effects, to check the analysis recovers them.

Every row is labelled simulated. Items differ in susceptibility (shared by both
models), so the two models move together, as expected of real agents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import GEMINI
from .cells import DECOY_CONFIRMATORY, DECOY_OPTIONAL, STAGE1

# True per-cell probabilities: flips for correct seeds, updates for wrong seeds.
TRUTH = {
    "gate.neutral.none": 0.05, "gate.pressure.none": 0.13,
    "both.neutral.none": 0.05, "both.pressure.none": 0.05,
    "nogate.neutral.none": 0.08, "nogate.pressure.none": 0.30,
    "scorer_only.pressure.none": 0.10, "query_only.pressure.none": 0.07,
    "gate_prompt.pressure.none": 0.11,
    "gate.pressure.decoy": 0.20, "both.pressure.decoy": 0.14, "gate.neutral.decoy": 0.12,
    "gate.neutral.lead": 0.40, "gate.pressure.lead": 0.42,
    "both.neutral.lead": 0.38, "both.pressure.lead": 0.39,
}
TRUE_EFFECTS = {
    "H1": TRUTH["gate.pressure.none"] - TRUTH["gate.neutral.none"],
    "H2": TRUTH["gate.pressure.decoy"] - TRUTH["both.pressure.decoy"],
    "H3": TRUTH["both.pressure.lead"] - TRUTH["gate.pressure.lead"],
    "manipulation": (TRUTH["gate.pressure.none"] - TRUTH["gate.neutral.none"])
    - (TRUTH["both.pressure.none"] - TRUTH["both.neutral.none"]),
}


def simulate(n_items: int, rng: np.random.Generator, truth: dict[str, float] = TRUTH) -> pd.DataFrame:
    cells = [c for c in STAGE1 + DECOY_CONFIRMATORY + DECOY_OPTIONAL if c.name in truth]
    susceptibility = 2 * rng.beta(2, 2, size=n_items)  # mean 1, shared by models
    rows = []
    for i in range(n_items):
        for model in GEMINI:
            for cell in cells:
                p = min(1.0, truth[cell.name] * susceptibility[i])
                hit = int(rng.random() < p)
                rows.append({"claim_id": str(i), "model": model, "cell": cell.name,
                             "flip": hit if cell.seed_correct else 0,
                             "update": 0 if cell.seed_correct else hit,
                             "decisive": True, "label": "Refuted", "simulated": True})
    return pd.DataFrame(rows)


def coverage(n_sims: int, n_items: int, seed: int = 7) -> dict[str, float]:
    """Share of simulations whose 95% interval covers the true effect."""
    from . import analysis

    rng = np.random.default_rng(seed)
    hits = {k: 0 for k in TRUE_EFFECTS}
    for _ in range(n_sims):
        est = analysis.contrasts(simulate(n_items, rng))
        for k, truth in TRUE_EFFECTS.items():
            hits[k] += int(est[k].lo <= truth <= est[k].hi)
    return {k: v / n_sims for k, v in hits.items()}


if __name__ == "__main__":
    from . import analysis

    analysis.N_BOOT = 1000
    print(coverage(200, 300))
