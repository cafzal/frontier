# Capital project selection (120 projects)

Pick which of 120 capital projects to fund, maximizing total NPV and strategic value while holding down total cost and risk exposure, under a hard $610M budget with per-category caps, dependencies, mutual exclusions, and a portfolio-size range (18–40). Binary (each project in or out), 4 objectives, combinatorial constraints: at this scale the exact-MILP frontier covers materially more of the tradeoff surface than a fixed-resolution metaheuristic, the canonical explore-fast-then-certify showcase.

- **`problem.json`**: 4 objectives (NPV and StrategicFit maximize, Cost and Risk minimize, all `sum` totals), binary approach, and the combinatorial constraints (budget, category caps, dependencies, exclusions, portfolio-size range).
- **`scores.json`**: the 120 projects scored on each objective.
- **`solutions.json`**: the exploratory NSGA `run` plus the exact-MILP `exact_run` overlay (HiGHS or cuOpt).

Load with `model load source="capital_project_selection_120"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) — the raw project list a planner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the brief + CSV reconstruct `problem.json` and `scores.json` verbatim (binary approach, the budget bound, cardinality 18–40, the four category caps, all dependencies/exclusions/force-includes, every score). `model load` is the shortcut that skips this step.
1. *“Which projects should we fund? Show me the real choices — where we can push value, where risk bites, and where the budget pinches.”*
   `solve run` → `explore tradeoffs`: the NPV/cost/risk/strategic-fit frontier — extremes, a balanced plan, inflection points, and the binding read (the budget and category caps).
2. *“Keep the balanced plan and the safest one as finalists. How much should I trust these?”*
   `explore curate` per pick (each carries a `quality` gate) → `solve solver="highs"` → `explore certify`: the exact-MILP overlay names which heuristic points it dominates — the headline at 120 binary options — plus coverage, the NSGA-never-dominates invariant, and corner sharpening.
3. *“What's the tightest lever, what should we stress-test, and can you guarantee we never blow the risk ceiling, whichever feasible plan we land on?”*
   `explore sensitivity` → `explore audit`: integer selections carry no solver duals, so sensitivity returns the frontier-inferred binding analysis plus `suggested_scenarios`; the audit proves a property over **every** feasible plan (`holds`) or returns a counterexample witness.
4. *“Write the shortlist up for the investment review.”*
   `explore curated format="markdown"`: the handoff table with per-finalist quality and the stakeholder-writeup pointer.

**Aggregation note.** All four objectives are totals (`sum`): a capital *deployment* decision wants the most total value the budget buys, with the binding budget and caps mediating portfolio size. Per-project *quality* (average strategic-fit or risk *level*) would be `avg`, which answers a different fixed-size question and falls outside the exact-MILP's linear scope. For `avg` / `quadratic` aggregation on a continuous shape, see [`investment_portfolio`](../investment_portfolio/).
