# Investment portfolio

A 30-ETF portfolio balancing return, volatility (via covariance), and yield, across three macro scenarios. The quadratic mean-variance objective makes the exact path a QP: the full Frontier workflow on a continuous problem with stress testing.

- **`problem.json`**: 3 objectives (Return / Volatility / Yield), proportional approach, constraints (single-fund ≤30%, ≤3 per sector, volatility ≤20%), and three macro scenarios (`recession`, `inflation`, `rate_cuts`).
- **`scores.json`**: the 30 funds, their per-objective scores, and the covariance matrix (the `Volatility` interaction matrix).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="investment_portfolio"`, or paste this to an agent connected to Frontier:

> Build a diversified ETF portfolio from the funds in scores.json: maximize return, minimize volatility from the covariance matrix, maximize yield. Constraints: no fund over 30%, ≤3 per sector, volatility under 20%. Explore the tradeoffs across the base case and the macro scenarios, solve it exactly (solver=highs), certify it, and read the duals.

## The workflow

1. **Solve** (`solve run`, plus `solve run_scenarios` for the macro regimes): the optimizer produces the return/volatility/yield frontier (covariance-based risk, sector caps binding) and a per-scenario frontier for `recession` (US-equity correlations up ~50%), `inflation`, and `rate_cuts`.
2. **Explore the tradeoffs** (`explore tradeoffs`): the extremes, a balanced portfolio, the knees, and how the frontier shifts across the scenarios.
3. **Certify and examine** (`solve solver="highs"` → `explore certify` → `explore sensitivity`): the exact mean-variance QP overlay audits the heuristic frontier and sharpens the volatility risk corner; the duals at the balanced portfolio show Yield as the costlier axis to push (~+57 versus Return ~+17), the Return shadow price falling ~51→0 along the frontier, GLD as the closest near-miss, and HYG pinned at its 30% cap. The read travels with the scenario.
4. **Decide** (`explore curate`): pin a few portfolios and commit on the tradeoffs.

**Scope.** Exact duals cover the continuous mean-variance (QP) shape; integer/MILP selection problems carry none, falling back to the frontier-inferred estimate (`source=frontier_inferred`). For the linear LP counterparts, see [`budget_allocation`](../budget_allocation/) and [`production_mix`](../production_mix/).
