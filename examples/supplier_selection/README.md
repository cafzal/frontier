# Supplier selection

Split a production order across 25 global suppliers: cost, reliability, lead time, ESG risk, and correlated regional disruption all pull against each other (the cheap suppliers are slower, less reliable, and riskier), and the per-region caps plus the quadratic concentration term make it too combinatorial for a spreadsheet or an LLM alone.

- **`problem.json`** — definition: 5 objectives (Cost ↓, Reliability ↑, LeadTime ↓, ESGRisk ↓, ConcentrationRisk ↓ quadratic), proportional approach, constraints (≤15% per supplier, ≤3 active suppliers per region, weighted reliability ≥78), and two scenarios: `china_disruption` (China suppliers offline) and `demand_surge` (a volume spike tightens per-supplier capacity to 10%).
- **`scores.json`** — 25 suppliers across 6 regions with per-objective scores plus the `ConcentrationRisk` interaction matrix (high within-region correlation, ~0 across regions).

Load both into Frontier (`model create` → `model update` with the objectives/options/scores/constraints/interaction_matrices/scenarios → `solve run` → `explore`), or paste this to an agent connected to Frontier:

> Allocate our order across the 25 suppliers in scores.json — minimize unit cost, maximize reliability, minimize lead time and ESG risk, and minimize concentration risk using the ConcentrationRisk interaction matrix (correlated regional disruption, not weighted-average). Constraints: no supplier over 15%, at most 3 active suppliers per region, weighted reliability at least 78. Show the tradeoffs across the base case, a China-region disruption (those suppliers offline), and a demand surge (per-supplier cap tightens to 10%) — the range of non-dominated multi-sourcing plans and where the knees are, not one "best."
