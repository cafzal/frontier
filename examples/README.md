# Examples

Loadable Frontier problems: combinatorial, multi-objective decisions beyond a spreadsheet. Each ships a `problem.json` (objectives, approach, constraints, scenarios), a `scores.json` (options, scores, interaction matrices), a paste-able prompt, and a workflow walkthrough in its own README. Each is pre-framed and pre-scored, so its walkthrough picks up the [main workflow](../README.md#workflow) at **Solve** (steps 1–2, Frame and Score, are already done).

| Example | Decision & objectives | Files |
|---|---|---|
| **[Capital project selection](capital_project_selection_120/)** | select from 120 projects – NPV / cost / risk / strategic fit *(all totals)*, $610M budget + dependencies + exclusions + category caps – the binary explore→certify showcase | [problem.json](capital_project_selection_120/problem.json) · [scores.json](capital_project_selection_120/scores.json) |
| **[Investment portfolio](investment_portfolio/)** | allocate across 30 ETFs – return / volatility *(quadratic covariance)* / yield, with scenarios | [problem.json](investment_portfolio/problem.json) · [scores.json](investment_portfolio/scores.json) |
| **[Budget allocation](budget_allocation/)** | split a growth budget across 8 initiatives – ROI / strategic reach *(purely linear)*, 35% cap – the **exact-LP duals** showcase | [problem.json](budget_allocation/problem.json) · [scores.json](budget_allocation/scores.json) |
| **[Production mix](production_mix/)** | allocate plant capacity across 10 products – margin / throughput / sustainability *(purely linear)*, 30% cap + ≤2 SKUs/line, with scenarios – the **richer exact-LP duals** showcase (swing SKU, structural exclusions, scenario stress) | [problem.json](production_mix/problem.json) · [scores.json](production_mix/scores.json) |
| **[Marketing channel budget](channel_budget/)** | allocate budget across 22 channels – conversions / reach *(quadratic overlap)* / ROAS / brand, per-platform caps | [problem.json](channel_budget/problem.json) · [scores.json](channel_budget/scores.json) |
| **[Supplier selection](supplier_selection/)** | multi-source across 25 suppliers – cost / reliability / lead time / ESG / *quadratic concentration risk*, per-region caps | [problem.json](supplier_selection/problem.json) · [scores.json](supplier_selection/scores.json) |
| **[Generation capacity planning](capacity_planning/)** | mix 22 generation projects – cost / CO2 / firmness + *quadratic intermittency*, emissions cap | [problem.json](capacity_planning/problem.json) · [scores.json](capacity_planning/scores.json) |

**Solver-exact explainability:** the [budget allocation](budget_allocation/) and [production mix](production_mix/) examples (purely linear **LP**) and the [investment portfolio](investment_portfolio/) (mean-variance **QP**) each run the full workflow – `solve run` → `solve solver="highs"` → `explore certify` → `explore sensitivity` – surfacing shadow prices ("where to invest") and reduced costs ("near-misses") straight from the optimizer on the continuous (LP / QP) paths. (Binary MILP shapes have no duals and fall back to the frontier-inferred estimate.)

**Load by name:** with the engine running, `model load source="investment_portfolio"` rebuilds any example directly – scenarios, interaction matrices, and all – with no manual re-entry. Problems you build save back to this same format (into a gitignored `saved/` library) via `model save`. See [Saving & loading problems](../README.md#saving--loading-problems).

See the [main README](../README.md) for setup and [architecture.md](../architecture.md) for technical reference.
