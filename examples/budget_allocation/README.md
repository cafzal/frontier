# Budget allocation

Split a fixed growth budget across 8 initiatives, trading ROI against strategic reach, with no initiative above 35%. Two purely linear objectives over a continuous allocation make this the simplest exact multi-objective LP: a clean end-to-end pass of the Frontier workflow.

- **`problem.json`**: 2 objectives (ROI and Strategic Reach, both maximize), proportional approach, one 35% per-initiative cap.
- **`scores.json`**: the 8 initiatives scored on ROI (%) and reach (0–10).
- **`solutions.json`**: the exploratory NSGA `run` plus the exact-LP `exact_run` overlay (HiGHS), with solver-exact duals per point.

Load with `model load source="budget_allocation"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“How should we split the growth budget across these eight initiatives? Show me the real ROI-versus-reach choices.”*
   `solve run` → `explore tradeoffs`: the ROI/reach frontier — extremes, a balanced plan, and the knees.
2. *“Keep the balanced split and the ROI-max one. Are these actually optimal?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify`: the exact LP overlay audits the heuristic frontier and sharpens the ROI corner.
3. *“Which of our limits is costing us the most, and which initiative just missed the cut?”*
   `explore sensitivity`: solver-exact duals — the Strategic Reach floor pricing at ~4.0 (each point of reach costs ~4% ROI, rising into diminishing returns), Localization the closest near-miss (reduced cost ~10), AI Copilot and Self-Serve Onboarding pinned at the 35% cap. (Values as read at the balanced anchor of the shipped exact overlay — duals are anchor-specific, so expect them to shift at another frontier point or after a re-solve.)
4. *“Write it up for the planning review.”*
   `explore curated format="markdown"`: the handoff table.

A small near-miss says "improve the option"; a binding cap says "lift your own limit." For the richer product-mix LP see [`production_mix`](../production_mix/); for the mean-variance QP, [`investment_portfolio`](../investment_portfolio/); for binary selection with no duals, [`capital_project_selection_120`](../capital_project_selection_120/).
