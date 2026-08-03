# Charging network siting

**The decision.** Pick which of 72 candidate EV fast-charging sites to build across seven metros and three highway corridors, maximizing drivers served while holding down catchment overlap and build cost, under a $34M budget, a 16–24 site program, per-metro queue caps, competing-lot exclusions, a committed corridor flagship, and a public commitment that every metro gets a site.

**Why Frontier.** The first example with an **interaction matrix on a binary problem**: linear reach overstates coverage where catchments overlap — two downtown garages share the same drivers — so a quadratic overlap term carries the correction, and a top-N-by-reach ranking fails twice (8 Harborview picks against a cap of 4, two metros missed entirely; repaired, it lands at 137.3 k-drivers/day where the frontier reaches 151.2 at *less* overlap and *less* cost).

**What ships here** — the raw inputs (step 1), the canonical model they frame into, and pre-solved results:

- **`data.csv` + `catchment_overlap.csv`**: the raw inputs a decision owner would actually have — everything step 1 pastes.
- **`problem.json`**: 3 objectives (DriversServed maximize `sum`, Overlap minimize **`quadratic`**, Cost minimize `sum`), binary approach, constraints (the $34M budget, 16–24 cardinality, per-metro `group_limit` **floors of 1** and caps of 4, corridor caps, the flagship `force_include`, 5 exclusion pairs), and two scenarios (`adoption_surge`, `grid_cost_inflation`).
- **`scores.json`**: the 72 sites scored per objective, plus the 72×72 `Overlap` interaction matrix (strong within a catchment — Harborview pairs most — near-zero across regions). Pairwise-only: triple-overlap is understated, stated here honestly.
- **`solutions.json`**: the exploratory NSGA `run` and the per-scenario `scenario_run`. No exact overlay — see the scope note below.

## The runbook

1. **Frame it from the raw inputs** — paste this ask, together with `data.csv` + `catchment_overlap.csv`, into a fresh session:

   > We're picking which of 72 candidate EV fast-charging sites to build (`data.csv`):
   > drivers served (k/day) and build cost ($M) per site, plus its metro or corridor, any
   > competing lot it's mutually exclusive with, and whether it's already committed (the
   > North-corridor flagship is). Two nearby sites share drivers — linear reach overstates
   > coverage — so `catchment_overlap.csv` carries the pairwise catchment-overlap matrix
   > (strong within a catchment, near zero across regions). For the sites affected, the
   > sheet also carries drivers-served under a corridor adoption surge and build cost if
   > metro grid quotes come back high.
   >
   > The decision is which sites to build — each is in or out. Maximize total drivers
   > served; minimize total cost and the overlap correction (through the matrix).
   >
   > Hard rules:
   > - Total build cost at or below $34M.
   > - Build between 16 and 24 sites.
   > - Every metro (HAR, CED, EAS, BRK, KNG, NOR, WES) gets at least 1 site and at most 4.
   > - Each highway corridor (NCR, CCR, VCR) gets at most 5.
   > - For each mutually-exclusive pair, build at most one.
   > - The committed flagship (NCR-01) must be built.
   >
   > Two futures to stress-test:
   > - **Adoption surge** — corridor demand jumps: the affected sites' drivers-served move
   >   to the `drivers_under_adoption_surge` column values.
   > - **Grid cost inflation** — metro quotes come back high: the affected sites' costs
   >   move to the `cost_under_grid_inflation` column values.

   Framing that input (`model create` + `model update`) lands on exactly this problem — the ask plus the data reconstruct `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load source="charging_network_siting"` is the shortcut: it skips framing and restores the pre-solved runs too.

2. *“Which of these 72 sites should we build? Show me the real choices between coverage, overlap, and cost under the $34M budget — and how the picks shift if corridor adoption surges or metro grid quotes come back 30% over.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the reach/overlap/cost frontier plus the `adoption_surge` and `grid_cost_inflation` scenario frontiers — the overlap axis is where clustered high-reach programs pay, the correction a ranking can't see.
3. *“Before this goes public: is every metro guaranteed a site, and the North-corridor flagship built, whichever program we pick?”*
   `explore audit` with the eight commitments as ONE conjunctive property list: verdict `holds` across every feasible program.
4. *“What's the least coverage any legal program delivers — does 53 k-drivers/day hold, and does 59?”*
   `explore audit` with an `objective_bound` min property: 53 `holds` — a floor written nowhere in the model, falling out of the floors + budget + cardinality arithmetic — while 59 is `violated` with a witness program serving 58.0.
5. *“Keep the balanced program and the max-coverage one, and write it up for the infrastructure committee.”*
   `explore curate` per pick → `explore curated format="markdown"`. (Asking for an exact solve here *declines by design* — see the scope note below.)

**Exact-scope note.** Ask for `solve solver="highs"` here and the engine *declines, by design*: the exact MILP optimizes additive (`sum`) objectives, and `Overlap` is `quadratic` on a binary selection — outside that scope — so the response is a redefine hint, not a silent wrong answer. The audit beats above still run: audit is a *feasibility* solve, and a quadratic objective that no bound touches never enters the constraint region. For the certifiable quadratic sibling (minimize-`quadratic` on a *proportional* mean-variance shape), see [`supplier_selection`](../supplier_selection/); for the binary certify + audit arc on an all-`sum` shape, see [`claims_investigation_triage`](../claims_investigation_triage/); for the same decline-then-audit arc on a `min`-aggregated (weakest-link) objective, see [`emergency_readiness_portfolio`](../emergency_readiness_portfolio/).
