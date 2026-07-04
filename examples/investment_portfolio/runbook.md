# Runbook — investment portfolio (continuous QP path, exact duals + scenarios)

Ordered prompts walking FRAME → SCORE → EXPLORE → CURATE → CERTIFY → EXAMINE → DECIDE over this
example, each naming the MCP tool it exercises and the response shape to expect. FRAME and SCORE
are pre-baked in `problem.json` / `scores.json` (including the covariance matrix and three macro
scenarios), so step 1 loads them. The README stays the narrative overview; this is the
paste-and-drive script.

1. **FRAME + SCORE — load the model**
   Prompt: *"Load the investment_portfolio example and summarize what decision it frames."*
   Tool: `model` (action=load, then get/summary).
   Expect: a `problem_id`; 3 objectives (Return↑, Volatility↓ quadratic, Yield↑), 30 options,
   proportional approach, the caps (single-fund, per-sector, volatility bound), and the
   `recession` / `inflation` / `rate_cuts` scenarios echoed back.

2. **EXPLORE — base frontier and the scenario futures**
   Prompt: *"Map the return/volatility/yield frontier, and how it shifts across the macro scenarios."*
   Tools: `solve` (action=run) → `solve` (action=run_scenarios) → `explore` (action=tradeoffs) →
   `explore` (action=scenario_frontiers).
   Expect: the base frontier with extremes / balanced / `inflection_point_candidates` (with
   `rationale`); a per-scenario frontier per macro regime; tradeoffs flags
   `scenarios_available` so the narration layers in robustness.

3. **CURATE — shortlist with names**
   Prompt: *"Pin the balanced portfolio and the low-volatility corner."*
   Tool: `explore` (action=curate, per solution).
   Expect: `curated: true` per pick plus a `quality` gate — on this proportional shape the
   distribution checks are live (single-fund concentration, allocations pinned at the cap).

4. **CERTIFY — exact QP overlay**
   Prompt: *"Solve it exactly and certify the frontier."*
   Tools: `solve` (solver="highs") → `explore` (action=certify).
   Expect: the exact mean-variance QP overlay; the certificate's `dominance_audit`, `coverage`,
   `invariant`, and `corner_sharpening` (strongest at the volatility corner), plus
   `quality_gates` over the certified points.

5. **EXAMINE — solver-exact duals, then the scenario handoff**
   Prompt: *"Which lever matters most at the balanced portfolio, and what should we stress-test?"*
   Tool: `explore` (action=sensitivity).
   Expect: `source: solver_exact` with `where_to_invest` (shadow prices per floor lever),
   `near_misses` and `capped_options` (reduced costs), the `frontier_shadow_price_trend`, and
   `suggested_scenarios` seeded from the top levers — duals rank, scenarios quantify. Create a
   scenario from a suggestion and copy its `motivated_by` onto it.

6. **EXAMINE — read the scenarios back**
   Prompt: *"Which portfolios survive the macro scenarios?"*
   Tool: `explore` (action=scenario_results).
   Expect: `option_robustness` tiers (core / common / marginal), `scenario_risk` (expected,
   worst-case, CVaR), and per-scenario `varies` + `held_fixed` lines restating exactly what each
   scenario changed; a scenario seeded in step 5 cites its `motivated_by`.

7. **DECIDE — the handoff**
   Prompt: *"Export the shortlist and write it up."*
   Tool: `explore` (action=curated, format="markdown").
   Expect: the handoff table with per-finalist `quality`, and the stakeholder-writeup pointer
   (context → decided → why → confidence → impact → next steps).
