# Investment portfolio

A 30-ETF portfolio balancing return, volatility (via covariance), and yield, across three macro scenarios. The quadratic mean-variance objective makes the exact path a QP: the full Frontier workflow on a continuous problem with stress testing.

- **`problem.json`**: 3 objectives (Return / Volatility / Yield), proportional approach, constraints (single-fund ≤30%, ≤3 per sector, volatility ≤20%), and three macro scenarios (`recession`, `inflation`, `rate_cuts`).
- **`scores.json`**: the 30 funds, their per-objective scores, and the covariance matrix (the `Volatility` interaction matrix).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="investment_portfolio"`, or paste this to an agent connected to Frontier:

> Build a diversified ETF portfolio from the funds in scores.json: maximize return, minimize volatility from the covariance matrix, maximize yield. Constraints: no fund over 30%, ≤3 per sector, volatility under 20%. Explore the tradeoffs across the base case and the macro scenarios, solve it exactly (solver=highs), certify it, and read the duals.

## The workflow

1. **Explore** (`solve run`): the return/volatility/yield frontier with its extremes, a balanced portfolio, and the knees; covariance-based risk, sector caps binding.
2. **Stress-test** (`solve run_scenarios`): re-solve per macro regime (`recession` raises US-equity correlations ~50%, plus `inflation` and `rate_cuts`) to see how the frontier shifts.
3. **Certify** (`solve solver="highs"`, then `explore certify`): the exact mean-variance QP overlay audits the heuristic frontier and sharpens the volatility risk corner.
4. **Examine sensitivity** (`explore sensitivity`): solver-exact duals at the balanced portfolio. Yield is the costlier axis to push (shadow ~+57 versus Return ~+17); the Return shadow price falls ~51→0 along the frontier as diminishing returns set in; GLD is the closest near-miss; HYG sits at its 30% cap. Sector-capped funds are filtered as structural exclusions, and the read travels with the scenario.
5. **Decide** (`explore curate`): pin a few portfolios and choose on the tradeoffs.

**Scope.** Exact duals cover the continuous mean-variance (QP) shape; integer/MILP selection problems carry none, falling back to the frontier-inferred estimate (`source=frontier_inferred`). For the linear LP counterparts, see [`budget_allocation`](../budget_allocation/) and [`production_mix`](../production_mix/).
