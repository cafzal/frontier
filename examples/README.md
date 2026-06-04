# Examples

Loadable Frontier problems — combinatorial, multi-objective decisions that need a real solver, not a spreadsheet. Each has a `problem.json` (objectives, approach, constraints, scenarios), a `scores.json` (options, scores, interaction matrices), and a paste-able prompt in its own README. The bundled `solutions.json` is the default NSGA frontier — heuristic (`exact=False`); re-solve with `solver="highs"` or `"cuopt"` to certify it.

| Example | Decision & objectives | Files |
|---|---|---|
| **[Investment portfolio](investment_portfolio/)** | allocate across 30 ETFs — return / volatility *(quadratic covariance)* / yield, with scenarios | [problem.json](investment_portfolio/problem.json) · [scores.json](investment_portfolio/scores.json) |
| **[Marketing channel budget](channel_budget/)** | allocate budget across 22 channels — conversions / reach *(quadratic overlap)* / ROAS / brand, per-platform caps | [problem.json](channel_budget/problem.json) · [scores.json](channel_budget/scores.json) |
| **[Supplier selection](supplier_selection/)** | multi-source across 25 suppliers — cost / reliability / lead time / ESG / *quadratic concentration risk*, per-region caps | [problem.json](supplier_selection/problem.json) · [scores.json](supplier_selection/scores.json) |
| **[Generation capacity planning](capacity_planning/)** | mix 22 generation projects — cost / CO2 / firmness + *quadratic intermittency*, emissions cap | [problem.json](capacity_planning/problem.json) · [scores.json](capacity_planning/scores.json) |
| **[Capital project selection](capital_project_selection/)** | fund 24 capital projects — NPV / cost / risk / strategic fit, budget + dependencies + exclusions | [problem.json](capital_project_selection/problem.json) · [scores.json](capital_project_selection/scores.json) |
| **[Capital project selection (120)](capital_project_selection_120/)** | the same decision at 120 projects — the scale where the heuristic frontier is part-dominated and an exact follow-on grounds it | [problem.json](capital_project_selection_120/problem.json) · [scores.json](capital_project_selection_120/scores.json) |
| **[cuOpt portfolio](cuopt_portfolio/)** | the portfolio problem via the opt-in GPU cuOpt backend | [notebook](cuopt_portfolio/cuopt_portfolio_frontier.ipynb) |
| **[cuOpt bidirectional](cuopt_bidirectional/)** | portfolio QP + capital MILP + duals via cuOpt | [notebook](cuopt_bidirectional/cuopt_bidirectional.ipynb) |
| **[Exact solver comparison](exact_solver_comparison/)** | cuOpt (GPU) vs HiGHS (CPU) as the exact follow-on — coverage / optimality / speed | [notebook](exact_solver_comparison/exact_solver_comparison.ipynb) |

**Load by name:** with the engine running, `model load source="investment_portfolio"` rebuilds any example directly — scenarios, interaction matrices, and all — with no manual re-entry. Problems you build save back to this same format (into a gitignored `saved/` library) via `model save`. See [Saving & loading problems](../README.md#saving--loading-problems).

See the [main README](../README.md) for setup and [architecture.md](../architecture.md) for technical reference.
