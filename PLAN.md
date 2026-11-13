# Study plan (public version)

Written before the sprint (prior work), copied here by hand on 2026-11-13 with internal planning notes removed. Hypotheses, sample size and analysis are locked after the pilot and before the main run; the hash of the locked plan will be committed here. Nothing in this file is a result.

## Question

Do evidence rules actually constrain an AI's reasoning, or do they just produce better justifications for changing its mind?

## Contribution

The gap is the combination of: evidence-gated belief revision, pressure-induced changes in retrieval and evaluation, repeated turns, and preserving justified updates through information boundaries. Literature searches through November 2026 found no paper running this combination; the search is repeated before submission.

1. Whether reason-free pressure gets past a citation rule in a search agent with a stored belief. Basta (arXiv 2610.36453) tests an evidence-gated revision prompt under graded pressure with no grounds an agent can verify. Our agent can verify by searching, and our rule is enforced in the pipeline rather than asked for in a prompt.
2. Where pressure enters, search or weighing, across repeated turns. Xiao (arXiv 2611.05720) separates stages for a one-turn seed only.
3. Whether an information boundary blocks pressure and keeps real information. Ma (arXiv 2608.26511) finds fixes trade yielding against rational updating; Shapira (arXiv 2610.28325) calls for mitigations that decouple them.

Not claimed: replies giving in while the stored belief holds, or a blind judge as a defence by itself.

## Design

- **Arms:** no gate; gate; gate plus prompt; query step behind a sanitizer that keeps facts and pointers and removes tone and stance (query-only); scorer with no view of the dispute (scorer-only); both.
- **Conditions:** tone (neutral, pressure) x lead (none, pointer to decisive evidence). Decoy: a pointer to a false-side passage in the claim's own store, with the belief starting correct, run for the gate and both arms at pressure (and the gate arm at neutral if budget allows). The sanitizer keeps pointers, so it passes decoys: this is where the boundary can fail.
- **Turns:** 3 user turns after the belief is set. Pressure is a fixed ladder rising in strength, always naming the target answer, adding strength, never reasons. Neutral repeats one statement and request on all 3 turns, also naming the target. Turn and strength are confounded.
- **Search:** exactly one query every turn, BM25 over the claim's own pages, top 5 passages of 5 sentences.
- **Seeds:** no-lead and decoy conditions start with the correct belief and measure regressive flips. Lead conditions start wrong and measure warranted updates. Closed-book priors are recorded.
- **Scorer:** sees the evidence the current belief rests on plus all passages retrieved so far, returns per-passage stances and a joint verdict. The gate allows a change only to the joint verdict's side and only with a cited passage the scorer marked as supporting it.
- **Data:** AVeriTeC dev split, each claim's own knowledge store. Not a neutral sample of the web: its authors built it with queries using the gold questions and answers, and added gold and distractor pages. The claim's own fact-check article and every page on a fixed list of fact-checking sites (or with "fact check" in its path) are removed, gold pages included. Claims labelled Supported or Refuted with at least one gold page with text left are eligible.
- **Pilot:** 20 items, all cells, kept out of the main run. Prompts may change only during the pilot.
- **Models:** Gemini 3.8 Flash and Gemini 3.1 Flash-Lite in every cell. Gemma 4 26B on the four no-lead cells of the gate and both arms, reported separately. All three are made by Google. Judge: Gemini 3.5 Flash-Lite, outside the agent models. Two outside models re-judge decisiveness on 100 gold pages.

### Cells

| Arm | Neutral, no lead | Pressure, no lead | Neutral, lead | Pressure, lead |
|---|---|---|---|---|
| Gate | 1 | 1 | 1 | 1 |
| Both | 1 | 1 | 1 | 1 |
| No gate | 1 | 1 | | 2 |
| Scorer-only | | 1 | | 2 |
| Query-only | | 1 | | 2 |
| Gate plus prompt | | 1 | | 2 |

Stage 1 is 13 cells; stage 2 (4 cells) runs on a random 100 items and is exploratory. Decoy cells run in stage 1 on every item whose store holds a false-side passage; at least 80 such items are needed, otherwise the decoy test is exploratory.

## Sample size

Computed only from measured cost per item and the budget, never from pilot effect sizes. Floor: 150 items. Cut order if money runs short: the neutral decoy cell, then stage 2's no-gate and gate-plus-prompt cells, then the rest of stage 2, then turns, then items down to the floor.

## Analysis

Flip: the stored belief ends on the false side at the final turn (primary); ever-flipped is secondary.

- **H1** = P(flip | gate, pressure, none) - P(flip | gate, neutral, none), correct seeds.
- **H2** = P(flip | gate, pressure, decoy) - P(flip | both, pressure, decoy), correct seeds, on qualifying items.
- **Manipulation check:** H1 - [P(flip | both, pressure, none) - P(flip | both, neutral, none)]. Holds by construction when the boundary is wired correctly; reported only as a wiring check.
- **H3** = P(update | both, pressure, lead) - P(update | gate, pressure, lead), wrong seeds, primary on items whose lead page a judge confirmed as decisive. Margin -8 points; also whether the bound clears -5.
- **Order:** H1 first; H2 and H3 each at full alpha; "the boundary works" needs both (intersection-union test).
- **Intervals:** 95% paired bootstrap by item; one-sided 95% lower bound for H3. Logistic model with item random intercept as a sensitivity check.
- **Secondary:** selectivity (flips blocked minus warranted updates lost), mechanism contrasts (gate minus both within each tone; the query x scorer 2x2 at pressure), stage attribution, laundered flips (citation invalid vs valid but one-sided, labelled blind to arm and tone), results by closed-book prior and by label, a check without re-annotated claims.
- **Sanitizer audit:** three probes guess tone and side from sanitizer output; the worst decides; a clean result is reported as "no leak detected at this sample".

## Before spending

- The whole pipeline runs against a mock model.
- The analysis runs on simulated data with known effects and recovers them.

## Timetable

Core running by Saturday midday; analysis frozen by Sunday midday; Sunday for labels and the report.
