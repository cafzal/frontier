# Supplier selection

Split a production order across 25 global suppliers balancing cost, reliability, lead time, ESG risk, and correlated regional disruption: the cheap suppliers are slower, less reliable, and riskier, and the per-region caps plus the quadratic concentration term make it too combinatorial for a spreadsheet or an LLM alone. The minimize-risk quadratic puts it on Frontier's exact mean-variance (QP) path.

- **`problem.json`**: 5 objectives (Cost / LeadTime / ESGRisk / ConcentrationRisk minimize, Reliability maximize; ConcentrationRisk is quadratic), proportional approach, constraints (≤15% per supplier, ≤3 active suppliers per region, weighted reliability ≥78), and two scenarios (`china_disruption`, `demand_surge`).
- **`scores.json`**: 25 suppliers across 6 regions with per-objective scores, plus the `ConcentrationRisk` interaction matrix (high within-region correlation, ~0 across regions).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="supplier_selection"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) + [concentration_interactions.csv](concentration_interactions.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“How should we split the order across these 25 suppliers, keeping regional concentration in check? Show me the real cost/reliability/lead-time/ESG choices — and how they hold up if China goes offline or demand surges.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the five-objective sourcing frontier plus per-scenario frontiers for `china_disruption` (those suppliers offline) and `demand_surge` (per-supplier capacity tightens to 10%).
2. *“Keep the balanced mix and the most reliable one. Are these actually optimal?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify`: the exact mean-variance QP overlay, sharpest at the minimum-concentration-risk corner.
3. *“Which of our rules costs us the most, and which supplier just missed the cut?”*
   `explore sensitivity`: solver-exact duals — whether a region cap or the reliability floor is the binding lever, the closest near-miss supplier, and which sit pinned at a cap.
4. *“Write it up for the sourcing review.”*
   `explore curated format="markdown"`: the handoff table.

For the same QP shape in finance, see [`investment_portfolio`](../investment_portfolio/); for the energy version, [`capacity_planning`](../capacity_planning/).
