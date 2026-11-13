# Laundered Capitulation

Entry to the Apart AI x Epistemics research sprint, Nov 13 to Nov 15, 2026.

Status: sprint started 2026-11-13 00:03 UTC (08:03 Taipei), when this repository was created. This repository holds plans, process notes and code only. It holds no results. Results stay private until submission, following Apart's responsible disclosure policy.

## Question

Do evidence rules actually constrain an AI's reasoning, or do they just produce better justifications for changing its mind?

A search agent keeps a belief about a claim and may change it only if retrieved evidence supports the change (the "gate"). A user pushes back over three turns with no reasons, only pressure. We test whether pressure still gets through, by shaping what the agent searches for or how it weighs what it finds, and whether separating the search step from the dispute blocks that while keeping real information the user offers.

## Prior work (preexisting, from before the sprint)

Everything in this list was written or decided before the sprint started on 2026-11-13 00:03 UTC. It is not new work from the sprint.

- Literature review on belief revision, sycophancy and evidence gating in LLMs. 2026-11-08 to 2026-11-12.
- Study design and analysis plan: arms, conditions, hypotheses, sample-size method, analysis. 2026-11-09 to 2026-11-13.
- Dataset feasibility check on the AVeriTeC dev split: claim, page and evidence counts only, with no model runs. 2026-11-09.
- Draft prompt wording for the pressure ladder, the neutral turns and the sanitizer. 2026-11-09 to 2026-11-10.
- Design votes: each design decision was put to three different AI systems (Claude, OpenAI Codex, Google Gemini) over 19 rounds, and adopted only when all three agreed. 2026-11-09 to 2026-11-13.

Preexisting material in this repo: `lc/factcheck_sites.txt` (the fixed list of fact-checking sites, written 2026-11-09) and the prompt wording in `lc/prompts.py` that comes from the draft above (pressure ladder, neutral turn, lead line, sanitizer instruction). All code is written during the sprint.

## New during the sprint

Everything below was started after kickoff. Each item has a commit that shows when.

- Public version of the study plan (`PLAN.md`).
- Pipeline, scorer, mock model and analysis code (`lc/`), with tests (`tests/`). First written 2026-11-13.
- Pilot on 20 items, kept out of the main run (described here only once finished, with no numbers).
- Hash of the locked hypotheses, sample size and analysis, committed after the pilot and before the main run.

## Data

- Dataset: AVeriTeC (Schlichtkrull et al.), dev split and its knowledge store. License: CC BY-NC 4.0.
- The data is not redistributed in this repo. Scripts fetch it from the original source.
- Personal data: none collected. The claims and web pages are public; no real users take part.

## Disclosure approach

- This repo follows Apart's responsible disclosure policy (https://apartresearch.com/info/responsible-disclosure-policy).
- During the sprint it holds plans, process notes and code with no results.
- At submission, code and data that the report links to are published only if the project is clearly low risk. Otherwise the report's Code and Data section says why they are withheld.

## License

Code: MIT (see `LICENSE`). Data: see above.
