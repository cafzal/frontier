# Capital project selection (120 projects)

Pick which of 120 capital projects to fund, maximizing total NPV and strategic value while holding down total cost and risk exposure, under a hard $610M budget with per-category caps, dependencies, mutual exclusions, and a portfolio-size range (18–40). Binary (each project in or out), 4 objectives, combinatorial constraints: at this scale the exact-MILP frontier covers materially more of the tradeoff surface than a fixed-resolution metaheuristic, the canonical explore-fast-then-certify showcase.

- **`problem.json`**: 4 objectives (NPV and StrategicFit maximize, Cost and Risk minimize, all `sum` totals), binary approach, and the combinatorial constraints (budget, category caps, dependencies, exclusions, portfolio-size range).
- **`scores.json`**: the 120 projects scored on each objective.
- **`solutions.json`**: the exploratory NSGA `run` plus the exact-MILP `exact_run` overlay (HiGHS or cuOpt).

Load with `model load source="capital_project_selection_120"`, or paste this to an agent connected to Frontier:

> Map the efficient frontier of funding plans across NPV, cost, risk, and strategic fit within the $610M budget, solve it exactly to certify the finalists, and walk me through a few representative plans.

## The workflow

1. **Explore** (`solve run`): the NPV/cost/risk/strategic-fit frontier of funding plans, with its extremes, a balanced plan, and the knees.
2. **Certify** (`solve solver="highs"` or `"cuopt"`, then `explore certify`): the exact MILP overlay returns the optimal subset for each scalarization and audits which heuristic points it dominates. At 120 binary options this is the headline step, reclaiming tradeoff surface a fixed-resolution metaheuristic misses.
3. **Examine** (`explore sensitivity`): integer selections carry no solver duals, so this reports the frontier-inferred binding analysis (which caps and the budget bind) rather than exact shadow prices.
4. **Decide** (`explore curate`): pin a few funding plans and choose on the tradeoffs.

**Aggregation note.** All four objectives are totals (`sum`): a capital *deployment* decision wants the most total value the budget buys, with the binding budget and caps mediating portfolio size. Per-project *quality* (average strategic-fit or risk *level*) would be `avg`, which answers a different fixed-size question and falls outside the exact-MILP's linear scope. For `avg` / `quadratic` aggregation on a continuous shape, see [`investment_portfolio`](../investment_portfolio/).
