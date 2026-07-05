# Charging network siting

Pick which of 72 candidate EV fast-charging sites to build across seven metros and three highway corridors, maximizing drivers served while holding down catchment overlap and build cost, under a $34M budget, a 16–24 site program, per-metro queue caps, competing-lot exclusions, a committed corridor flagship, and a public commitment that every metro gets a site. The first example with an **interaction matrix on a binary problem**: linear reach overstates coverage where catchments overlap — two downtown garages share the same drivers — so a quadratic overlap term carries the correction, and a top-N-by-reach ranking fails twice (8 Harborview picks against a cap of 4, two metros missed entirely; repaired, it lands at 137.3 k-drivers/day where the frontier reaches 151.2 at *less* overlap and *less* cost).

- **`problem.json`**: 3 objectives (DriversServed maximize `sum`, Overlap minimize **`quadratic`**, Cost minimize `sum`), binary approach, constraints (the $34M budget, 16–24 cardinality, per-metro `group_limit` **floors of 1** and caps of 4, corridor caps, the flagship `force_include`, 5 exclusion pairs), and two scenarios (`adoption_surge`, `grid_cost_inflation`).
- **`scores.json`**: the 72 sites scored per objective, plus the 72×72 `Overlap` interaction matrix (strong within a catchment — Harborview pairs most — near-zero across regions). Pairwise-only: triple-overlap is understated, stated here honestly.
- **`solutions.json`**: the exploratory NSGA `run` and the per-scenario `scenario_run`. No exact overlay — see the scope note below.

Load with `model load source="charging_network_siting"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) + [catchment_overlap.csv](catchment_overlap.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“Which of these 72 sites should we build? Show me the real choices between coverage, overlap, and cost under the $34M budget — and how the picks shift if corridor adoption surges or metro grid quotes come back 30% over.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the reach/overlap/cost frontier plus the `adoption_surge` and `grid_cost_inflation` scenario frontiers — the overlap axis is where clustered high-reach programs pay, the correction a ranking can't see.
2. *“Before this goes public: is every metro guaranteed a site, and the North-corridor flagship built, whichever program we pick?”*
   `explore audit` with the eight commitments as ONE conjunctive property list: verdict `holds` across every feasible program.
3. *“What's the least coverage any legal program delivers — does 53 k-drivers/day hold, and does 59?”*
   `explore audit` with an `objective_bound` min property: 53 `holds` — a floor written nowhere in the model, falling out of the floors + budget + cardinality arithmetic — while 59 is `violated` with a witness program serving 58.0.
4. *“Keep the balanced program and the max-coverage one, and write it up for the infrastructure committee.”*
   `explore curate` per pick → `explore curated format="markdown"`. (Asking for an exact solve here *declines by design* — see the scope note below.)

**Exact-scope note.** Ask for `solve solver="highs"` here and the engine *declines, by design*: the exact MILP optimizes additive (`sum`) objectives, and `Overlap` is `quadratic` on a binary selection — outside that scope — so the response is a redefine hint, not a silent wrong answer. The audit beats above still run: audit is a *feasibility* solve, and a quadratic objective that no bound touches never enters the constraint region. For the certifiable quadratic sibling (minimize-`quadratic` on a *proportional* mean-variance shape), see [`supplier_selection`](../supplier_selection/); for the binary certify + audit arc on an all-`sum` shape, see [`claims_investigation_triage`](../claims_investigation_triage/).
