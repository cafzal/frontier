# Budget allocation

Split a fixed growth budget across 8 initiatives, trading **ROI** against **strategic reach**, with no initiative above 35%. Two purely linear objectives over a continuous allocation make this the simplest **exact multi-objective LP** — a clean end-to-end pass of the Frontier workflow.

- **`problem.json`** — 2 objectives (ROI / Strategic Reach, both maximize), proportional approach, one 35% per-initiative cap.
- **`scores.json`** — the 8 initiatives × (ROI %, reach 0–10).
- **`solutions.json`** — the exploratory NSGA `run` + the exact-LP `exact_run` overlay (HiGHS), with solver-exact duals per point.

Load with `model load source="budget_allocation"`, or paste this to an agent connected to Frontier:

> Split a fixed growth budget across the initiatives in scores.json to maximize ROI and strategic reach, with no initiative over 35%. Show me the frontier, solve it exactly (solver=highs), certify it, and read the duals for the closest near-miss and the binding limit — not one "best."

## The workflow

- **Explore** (`solve run`) — the ROI↔reach frontier: the extremes, a balanced plan, the knees.
- **Certify** (`solve solver="highs"` → `explore certify`) — the exact LP overlay audits the heuristic frontier and sharpens the ROI corner.
- **Sensitivity** (`explore sensitivity`) — solver-exact duals at the balanced plan: the **Strategic Reach floor** prices at ~4.0 (each point of reach costs ~4% ROI, rising along the frontier — exact diminishing returns); **Localization** is the closest near-miss (reduced cost ~10); **AI Copilot** and **Self-Serve Onboarding** sit at the 35% cap (they'd take more if allowed).
- **Decide** (`explore curate`) — pin a few plans and choose on the tradeoffs.

(The read: a small near-miss says "improve the option," a binding cap says "lift your own limit" — the same duals HiGHS and cuOpt expose on the LP path. For the richer product-mix LP see [`production_mix`](../production_mix/); for the mean-variance QP, [`investment_portfolio`](../investment_portfolio/); for binary selection with no duals, [`capital_project_selection_120`](../capital_project_selection_120/).)
