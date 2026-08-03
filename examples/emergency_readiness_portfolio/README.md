# Regional emergency readiness portfolio

**The decision.** A regional resilience authority funds 20–24 of 96 response assets — standing teams, equipment caches, mobile units, and standby contracts — across five hazard lines (flood, wildfire, quake, storm, hazmat) and four subregions, under a $58M/yr budget and a 4,600 response-hour surge floor. **Readiness does not add up.** When a hazard hits, every asset the authority holds on that line is tasked, and the line performs at the level of the least-ready unit it fields — so portfolio readiness is the *weakest hazard line*, a `min`, not a total. Adding an asset can lower it.

**Why Frontier.** 2^96 candidate portfolios, coupled by shared station crews, mobile-unit cache dependencies, and line/subregion floors and caps — and the first bundled example whose signature objective is **`min`-aggregated**, which changes what every layer does. The flagship receipt is the proof layer's: the same NSGA search, rerun under another seed, tops out at a weakest link of **38.7** while the exact audit proves **41.8** attainable — the search missed 3.1 points the proof layer found. The lesson generalizes: on a `min` (weakest-link) objective, a heuristic's best value is a lower bound on what's attainable; the audit probe confirms the true ceiling. The baked run reaches 41.8, and the same probes certify it — feasible at 41.8, every plan above 41.85 proven out of reach — a verdict a search can only approach from below.

The naive plays fail here in ways worth naming:

- A **capability-per-dollar ranking is illegal here before it is even suboptimal**: the top 22 by surge-hours-per-$M is 22 standby contracts, breaking 8 rules at once (no hazmat anywhere, no swift-water team, two lines over cap). Patched into legality it costs $35.2M for a weakest link of 27.2 while over-buying surge to 2.2× the floor; for the same money the frontier offers **41.8** at $33.2M, still above the surge floor and 2.7 hours faster to activate.
- **Stacking the best assets fails the other way.** The 22 highest-readiness assets read like an 81.4 roster and are illegal on 12 counts (starting with $87.2M against a $58M budget). Patched to legality — the patch buys the surge hours an elite roster never had, and the cheapest hours are the least ready — it lands at **37.7**: a 43.7-point collapse, and still short of the frontier's 41.8.
- The frontier's key finding is a **bottleneck**, not a total: no legal portfolio can ever exceed a weakest link of **41.8**, because every subregion must field a hazmat unit and the Delta's best hazmat asset is a contracted spill responder rated 41.8. Money is not the lever — that ceiling survives every budget down to $27M/yr.

**What ships here** — the raw inputs (step 1), the canonical model they frame into, and pre-solved results:

- **`data.csv`**: the raw inputs a resilience authority would actually have — everything step 1 pastes.
- **`problem.json`**: 4 objectives (Readiness maximize **`min`**, SurgeCapacity maximize `sum`, Cost minimize `sum`, ActivationTime minimize `avg`), binary approach, 45 constraints (the $58M budget, the 4,600-hour surge floor, the 20–24 portfolio, per-line floors of 2 and caps of 7, per-subregion floors of 3 and caps of 8, ≥2 standing swift-water teams, ≥1 hazmat unit per subregion, 8 shared-crew exclusions, 20 mobile-unit → cache dependencies), and two scenarios (`concurrent_season`, `dual_hazmat_rule`).
- **`scores.json`**: the 96 assets scored per objective. Readiness is the certified rating of the hazard line *with that asset fielded* — the number the line is judged on when that unit is the weakest thing rolling.
- **`solutions.json`**: the exploratory NSGA `run` and the per-scenario `scenario_run`. No exact overlay — see the scope note below.

## The runbook

1. **Frame it from the raw inputs** — paste this ask, together with `data.csv`, into a fresh session:

   > We're renewing the region's standing response portfolio (`data.csv`): 96 assets — standing
   > teams, equipment caches, mobile units, and standby contracts — each with its hazard line
   > (flood, wildfire, quake, storm, hazmat), its subregion, a readiness rating (0–100), the
   > response-hours it can deliver in the first 72 hours, its annual cost ($M), and how long it
   > takes to activate (hours). The sheet also carries the crew a unit shares with another unit,
   > the cache a mobile unit deploys on, and the readiness and activation figures for the
   > affected assets under a concurrent flood-and-hazmat season.
   >
   > The decision is which assets to fund — each is in or out.
   >
   > Readiness is NOT a total. When a hazard hits, every asset we hold on that line is tasked,
   > and the line is only as good as the least-ready unit it fields — so the portfolio's
   > readiness is its WEAKEST hazard line. Maximize that weakest line. Maximize total surge
   > hours. Minimize total annual cost, and minimize the average activation time.
   >
   > Hard rules:
   > - Total annual cost at or below $58M.
   > - Total surge capacity at or above 4600 response-hours in the first 72 hours.
   > - Fund between 20 and 24 assets.
   > - Every hazard line gets at least 2 assets and at most 7.
   > - Every subregion gets at least 3 assets and at most 8.
   > - At least 2 standing swift-water teams (the flood-line teams).
   > - At least 1 hazmat unit in every subregion.
   > - Two units that share a station crew can't both be stood up.
   > - A mobile unit only deploys with its line's cache in the same subregion — fund the unit,
   >   fund the cache.
   >
   > Two futures to stress-test:
   > - **Concurrent season** — floodwater inundates tank farms and rail sidings, so the flood
   >   and hazmat lines answer the same event: flood assets rate 66% of their single-event
   >   readiness, hazmat assets 78%, and both lines activate 40% slower (the affected assets'
   >   figures are in the `..._under_concurrent_season` columns). Every rule stays as-is.
   > - **Dual hazmat rule** — after the regional spill review, every subregion must field two
   >   hazmat units instead of one (which lifts the hazmat line cap from 7 to 8 to fit them).
   >   Every other rule stays as-is.

   Framing that input (`model create` + `model update`) lands on exactly this problem — the ask plus the data reconstruct `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load source="emergency_readiness_portfolio"` is the shortcut: it skips framing and restores the pre-solved runs too.

2. *“Which assets should we fund? Show me the real choices — how ready our weakest line can be, how many surge hours we carry, what it costs, and how fast we get moving.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: 40 plans over 2^96 portfolios — the weakest line spans **27.2 to 41.8**, surge 4,768–9,834 hours, cost $30.2–57.8M, activation 2.9–7.6 hours. The balanced plan funds 22 assets at 37.6 / 7,729h / $41.7M / 5.9h. The trade that matters is **surge against readiness** (corr −0.40): contracted hours are how a portfolio buys capacity, and they are what drags the weakest line down. Cost barely moves readiness at all (corr −0.01).
3. *“Before we sign: can you guarantee the coverage commitments hold whatever we fund — and can we count on a third swift-water team whatever we pick?”*
   `explore audit` with the ten commitments as ONE conjunctive property list (≥2 swift-water teams, a hazmat unit in each of the four subregions, ≥2 assets on each of the five lines): verdict `holds` across every feasible portfolio, in one call. Tighten swift-water to 3 and it flips to `violated`, with a concrete legal 20-asset portfolio carrying exactly 2. Ask for the minimum spend and audit answers that too: **no feasible portfolio costs less than $25.5M/yr** (`holds`; at $25.6M it returns a counterexample) — a floor nobody wrote down, falling out of the surge floor, the 20-asset minimum, and the line/subregion/hazmat floors.
4. *“Give me the exact solve — I want these certified.”*
   `solve solver="highs"` **declines, by design** — see the scope note. The engine says why (a redefine hint naming the out-of-scope aggregations); the walkthrough routes the proof to `explore audit`, whose feasibility solves reason through the constraint set alone.
5. *“What is actually holding readiness down, and what would move it?”*
   `explore composition` + `explore sensitivity` on the ceiling plans, read against the bottleneck: **every plan on the frontier that reaches 41.8 is pinned by the same asset, HZM-DL-04** — the Delta's contracted spill responder. The ceiling is arithmetic: a floor of *k* over a group forces *k* of its members in, so the weakest link can never beat that group's *k*-th best asset; across all fifteen mandated groups the tightest bound is `hazmat in Delta` at **41.8**, and the next-tightest is 78.1, so the binder is unique. Two HiGHS feasibility probes settle it — a portfolio with every asset at or above 41.8 is *feasible*; above 41.85 it is *no feasible plan*. **The lever is not in this decision.** Procuring one credible Delta hazmat unit moves the ceiling; another $25M of budget leaves it exactly where it is.
6. *“And if the season goes bad, or the spill review changes the rule?”*
   `explore scenario_results` + `explore compare_runs`: under `concurrent_season` the ceiling falls **41.8 → 32.6**, still pinned by the Delta hazmat line — the concurrency lands on the line that was already thinnest. Under `dual_hazmat_rule` it falls **41.8 → 38.2**: requiring a *second* hazmat unit per subregion *lowers* measured readiness, because the second unit the Delta can field is worse than the first. A governance tightening that reads as strengthening is worth knowing before it is signed.
7. *“Keep the ceiling plan and the balanced one, and write it up for the board.”*
   `explore curate` per pick → `explore curated format="markdown"`: the handoff table, led by the audited guarantee and the 41.8 ceiling — the two things that hold whichever finalist the board picks.

**Exact-scope note.** Ask for `solve solver="highs"` here and the engine *declines, by design*: the exact MILP optimizes additive (`sum`) objectives, and on this binary selection `Readiness` is `min` (nonlinear) and `ActivationTime` is `avg` (fractional over a variable-size pick) — outside that scope, so the response is a redefine hint, not a silent wrong answer. The proof layer is still exact: `explore audit` is a *feasibility* solve, so it reasons over the whole feasible region without ever touching the objective (steps 3 and 5). One boundary is honest to name — `explore audit`'s `objective_bound` property needs a `sum` objective, so a *bound* on `Readiness` is out of scope; the 41.8 ceiling is proven instead by force-exclude feasibility probes, which is the same MILP doing the same job through the constraint set. For the same decline-then-audit arc with a *quadratic* objective, see [`charging_network_siting`](../charging_network_siting/); for the audit flagship on an all-`sum` shape, [`claims_investigation_triage`](../claims_investigation_triage/); for the emergent-guarantee flavor on a staffing roster, [`shift_coverage_staffing`](../shift_coverage_staffing/).
