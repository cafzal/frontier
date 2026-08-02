# Investment portfolio

**The decision.** A 30-ETF portfolio balancing return, volatility (via covariance), and yield, across three macro scenarios.

**Why Frontier.** The quadratic mean-variance objective makes the exact path a QP: the full Frontier workflow on a continuous problem with stress testing.

**What ships here** — the raw inputs (step 1), the canonical model they frame into, and pre-solved results:

- **`data.csv` + `covariance.csv`**: the raw inputs a decision owner would actually have — everything step 1 pastes.
- **`problem.json`**: 3 objectives (Return / Volatility / Yield), proportional approach, constraints (single-fund ≤30%, ≤3 per sector, volatility ≤20%), and three macro scenarios (`recession`, `inflation`, `rate_cuts`).
- **`scores.json`**: the 30 funds, their per-objective scores, and the covariance matrix (the `Volatility` interaction matrix).
- **`solutions.json`**: the exploratory NSGA `run`, the exact mean-variance QP `exact_run` overlay (HiGHS) with solver-exact duals per point, and the per-scenario `scenario_run`.

## The runbook

1. **Frame it from the raw inputs** — paste this ask, together with `data.csv` + `covariance.csv`, into a fresh session:

   > We're allocating a portfolio across 30 ETFs (`data.csv`): expected return, volatility,
   > and dividend yield per fund, plus each fund's asset-class group. Portfolio risk is
   > covariance-driven — `covariance.csv` carries the pairwise matrix from our analyst.
   >
   > The decision is each fund's weight — weights total 100%. Maximize return and yield
   > (weighted averages); minimize portfolio volatility (through the covariance matrix).
   >
   > Hard rules:
   > - No single fund above 30%.
   > - At most 3 active holdings per asset-class group.
   > - Portfolio volatility at or below 20%.
   >
   > Three macro futures to stress-test:
   > - **Rate cuts** — yields read 15% lower across the board.
   > - **Inflation** — returns −10% and yields +15% across the board.
   > - **Recession** — co-movement changes: US equities move together 1.5x more than baseline;
   >   everything else unchanged.

   Framing that input (`model create` + `model update`) lands on exactly this problem — the ask plus the data reconstruct `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load source="investment_portfolio"` is the shortcut: it skips framing and restores the pre-solved runs too.

2. *“How should we allocate? Show me the return/risk/yield tradeoffs, and how the picture changes in a recession, an inflation run, or rate cuts.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs` + `explore scenario_frontiers`: the covariance-based frontier (sector caps binding) plus one frontier per macro regime.
3. *“Keep the balanced portfolio and the calmest one. Are these optimal, or just decent?”*
   `explore curate` per pick (the proportional quality checks are live — concentration, cap-pinning) → `solve solver="highs"` → `explore certify`: the exact mean-variance QP overlay, sharpest at the volatility corner.
4. *“At the balanced portfolio, which of my rules is costing me the most — and which allocations hold up across all three macro futures?”*
   `explore sensitivity` → `explore scenario_results`: solver-exact duals — Return the costlier axis to push at the balanced anchor (floor price ~15, the Yield floor slack there), the Return shadow price falling to ~0 toward the calm end of the frontier, the near-miss read naming the closest excluded fund, VGSH pinned at its 30% cap (duals are anchor-specific — expect the values to shift at another frontier point or after a re-solve) — then robustness tiers, scenario risk, and per-scenario `varies`/`held_fixed`.
5. *“Write the shortlist up for the investment committee.”*
   `explore curated format="markdown"`: the handoff table with per-finalist quality.

**Scope.** Exact duals cover the continuous mean-variance (QP) shape; integer/MILP selection problems carry none, falling back to the frontier-inferred estimate (`source=frontier_inferred`). For the linear LP counterparts, see [`budget_allocation`](../budget_allocation/) and [`production_mix`](../production_mix/); for a quadratic shape the exact path *declines* (the approx-first showcase), [`channel_budget`](../channel_budget/).
