"""Arms, conditions and the cell table from the plan."""
from __future__ import annotations

from dataclasses import dataclass

ARMS = ("nogate", "gate", "gate_prompt", "query_only", "scorer_only", "both", "gate_strict")
# Which parts of the pipeline see the dispute.
QUERY_SEES_DISPUTE = {"nogate": True, "gate": True, "gate_prompt": True,
                      "query_only": False, "scorer_only": True, "both": False, "gate_strict": True}
SCORER_SEES_DISPUTE = {"nogate": True, "gate": True, "gate_prompt": True,
                       "query_only": True, "scorer_only": False, "both": False, "gate_strict": True}
GATED = {"nogate": False, "gate": True, "gate_prompt": True,
         "query_only": True, "scorer_only": True, "both": True, "gate_strict": True}


@dataclass(frozen=True)
class Cell:
    arm: str
    tone: str  # neutral | pressure
    lead: str  # none | lead | decoy
    stage: int

    @property
    def name(self) -> str:
        return f"{self.arm}.{self.tone}.{self.lead}"

    @property
    def seed_correct(self) -> bool:
        """No-lead and decoy cells start correct; lead cells start wrong."""
        return self.lead != "lead"


def _c(arm: str, tone: str, lead: str, stage: int) -> Cell:
    return Cell(arm, tone, lead, stage)


STAGE1 = (
    _c("gate", "neutral", "none", 1), _c("gate", "pressure", "none", 1),
    _c("gate", "neutral", "lead", 1), _c("gate", "pressure", "lead", 1),
    _c("both", "neutral", "none", 1), _c("both", "pressure", "none", 1),
    _c("both", "neutral", "lead", 1), _c("both", "pressure", "lead", 1),
    _c("nogate", "neutral", "none", 1), _c("nogate", "pressure", "none", 1),
    _c("scorer_only", "pressure", "none", 1), _c("query_only", "pressure", "none", 1),
    _c("gate_prompt", "pressure", "none", 1),
)
STAGE2 = (
    _c("nogate", "pressure", "lead", 2), _c("scorer_only", "pressure", "lead", 2),
    _c("query_only", "pressure", "lead", 2), _c("gate_prompt", "pressure", "lead", 2),
)
DECOY_CONFIRMATORY = (_c("gate", "pressure", "decoy", 1), _c("both", "pressure", "decoy", 1))
DECOY_OPTIONAL = (_c("gate", "neutral", "decoy", 1),)
STRONG_CELLS = (_c("gate", "strong", "none", 1), _c("nogate", "strong", "none", 1))  # exploratory
# Exploratory addendum (stricter scorer): new arm plus a same-route control, neutral tone.
STRICT_CELLS = (_c("gate_strict", "neutral", "none", 1), _c("gate_strict", "neutral", "lead", 1),
                _c("gate", "neutral", "none", 1), _c("gate", "neutral", "lead", 1))
GEMMA_CELLS = tuple(c for c in STAGE1 if c.arm in ("gate", "both") and c.lead == "none")

assert len(STAGE1) == 13 and len(STAGE2) == 4 and len(GEMMA_CELLS) == 4


def cells_for(model: str, *, stage2: bool, has_decoy: bool, neutral_decoy: bool,
              gemma_models: tuple[str, ...], strong: bool = False,
              strict_only: bool = False) -> tuple[Cell, ...]:
    if strict_only:
        return STRICT_CELLS
    if model in gemma_models:
        return GEMMA_CELLS
    cells = list(STAGE1)
    if has_decoy:
        cells += DECOY_CONFIRMATORY
        if neutral_decoy:
            cells += DECOY_OPTIONAL
    if stage2:
        cells += STAGE2
    if strong:
        cells += STRONG_CELLS
    return tuple(cells)
