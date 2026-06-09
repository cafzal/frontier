# Capital Project Selection (120 projects)

Pick which of **120 capital projects** to fund — maximize total **NPV** and total **strategic value delivered**, hold down total **cost** and total **risk exposure**, under a hard **$610M budget**, per-category caps, dependencies, mutual exclusions, and a portfolio-size range (18–40). Binary (each project in or out); 4 objectives; combinatorial constraints. At this scale the exact-MILP frontier (one solve per scalarization, HiGHS on CPU or cuOpt on GPU) covers materially more of the tradeoff surface than a fixed-resolution metaheuristic: the canonical *explore-fast-then-certify* showcase.

**Aggregation — all four objectives are totals (`sum`).** This is a capital *deployment* decision: you want the most total value your budget buys, against total spend and total risk exposure, so the natural unit is the portfolio total and the binding budget (plus the caps) mediates portfolio size. If instead you cared about per-project *quality* (average strategic-fit or average risk *level*), you'd model Risk/StrategicFit as `avg`; but that answers a different (fixed-size rating) question, and `avg` over a variable-size selection is fractional, outside the exact-MILP's linear scope (it would no longer certify). See the [investment portfolio](../investment_portfolio/) for `avg`/`quadratic` aggregation on a continuous shape.

**Prompt:** "Map the efficient frontier of funding plans across NPV, cost, risk, and strategic fit within the $610M budget, and walk me through a few representative plans."

Files: [problem.json](problem.json) · [scores.json](scores.json)
