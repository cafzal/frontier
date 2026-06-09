# Investment portfolio

Loadable Frontier example — a 30-ETF portfolio balancing **return**, **volatility** (via covariance), and **yield**, across three macro scenarios. A quadratic (mean-variance) objective makes the exact path a **QP** — the full Frontier workflow on a continuous problem with stress testing.

- **`problem.json`** — 3 objectives (Return / Volatility / Yield), proportional approach, constraints (single-fund ≤30%, ≤3 per sector, volatility ≤20%), and three macro scenarios (`recession`, `inflation`, `rate_cuts`).
- **`scores.json`** — the 30 funds, their per-objective scores, and the covariance matrix (the `Volatility` interaction matrix).
- **`solutions.json`** — the exploratory NSGA `run` + the per-scenario `scenario_run`.

Load with `model load source="investment_portfolio"`, or paste this to an agent connected to Frontier:

> Build a diversified ETF portfolio from the funds in scores.json — maximize return, minimize volatility (use the covariance matrix, not weighted-average vol), maximize yield. Constraints: no fund over 30%, ≤3 per sector, volatility under 20%. Explore the tradeoffs across the base and the macro scenarios, solve it exactly (solver=highs), certify it, and read the duals — not one "best."

## The workflow

- **Explore** (`solve run`) — the return↔volatility↔yield frontier: extremes, a balanced portfolio, the knees; covariance-based risk, sector caps binding.
- **Stress** (`solve run_scenarios`) — re-solve per macro regime (`recession` — US-equity correlations up ~50%, `inflation`, `rate_cuts`) to see how the frontier shifts under stress.
- **Certify** (`solve solver="highs"` → `explore certify`) — the exact mean-variance QP overlay audits the heuristic frontier and sharpens the volatility risk corner.
- **Sensitivity** (`explore sensitivity`) — solver-exact duals at the balanced portfolio: **Yield** is the costlier axis to push (shadow ~+57 vs Return ~+17); the Return shadow price falls ~51→0 along the frontier (diminishing returns); **GLD** is the closest near-miss; **HYG** sits at its 30% cap. Sector-capped funds are filtered as structural exclusions — and the read travels with the scenario.
- **Decide** (`explore curate`) — pin a few portfolios and choose on the tradeoffs.

**Scope.** Exact duals on the **continuous** mean-variance (QP) shape; integer/MILP selection problems carry none — there `explore sensitivity` falls back to the frontier-inferred estimate (`source=frontier_inferred`). (For the linear LP counterparts, see [`budget_allocation`](../budget_allocation/) and [`production_mix`](../production_mix/).)
