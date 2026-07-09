# Solution Interpreter — Presentation Refinements

Situational presentation playbooks for `frontier://skills/solution_interpreter`. The skill's Core Judgment (SKILL.md) is always-apply; each section here applies when the matching explore output is in front of you. Fetch one section at need: `get_skill('solution_interpreter', section='<heading>')`.

## Presentation Refinements

Apply these when the situation calls for them. They improve quality but are secondary to the critical judgment above.

### Frontier Quality and Completeness Signals

Every solve returns two pre-computed signals — read them first, before scanning raw metrics:

- **`frontier_quality.status`** — `GOOD` / `WARNING` / `POOR`. Combines a `gates` block (`frontier_returned`, `non_trivial`, `diverse`) with `issues[]` describing any failure. The classifier already incorporates spacing CV and allocation concentration; you don't need to reinterpret those raw numbers.
- **`frontier_complete`** — `True` when the returned set is the full feasible Pareto frontier; `False` when pruning truncated it (`total_pareto_found > solutions_found`).

**Acting on `frontier_quality.status`:**

| Status | Meaning | What to do |
|---|---|---|
| **GOOD** | All gates pass | Present results with confidence. Skip quality talk entirely. |
| **WARNING** | Frontier is non-trivial but uneven (clustered coverage or single-winner allocation) | Surface the issue from `issues[]` in domain language, then proceed. Don't block exploration. |
| **POOR** | Empty or degenerate frontier (<2 solutions, or all objectives flat) | Stop and route to optimization_strategy. Don't present a degenerate frontier as if it were a real choice set. |

**Acting on `frontier_complete`:**

- `True` — say nothing; the frontier is the complete picture.
- `False` — frame the set explicitly: *"You're seeing N of [total_pareto_found] solutions — a representative sample, not the full set."* This matters when the user is reasoning about coverage ("are there other strategies?") or confidence ("how do I know nothing better exists?"). Encourage `max_solutions=` overrides if they want more.

**Acting on `time_limited`:** When `time_limited` is `True`, a wall-clock cap stopped the run early, so the frontier is **best-so-far, not converged**. Say so before presenting it — *"This is a time-bounded run, so it's a partial frontier — worth a full re-run before committing."* — and offer to re-run uncapped (or with a higher cap). On an exact run a cap thins coverage only: each returned point is still optimal for its scalarization (per-point optimality claims hold), but don't call it the whole frontier.

**Translating `WARNING` issues:**

The raw `issues[]` strings are agent-readable. Translate to user language:
- *"Spacing CV …uneven"* → "The frontier clusters in some tradeoff regions — there may be zones we haven't explored. Want me to try thorough mode?"
- *"Solution #X allocates Y% to Z…"* → "Solution X concentrates Y% on Z — that may be intended, but if a more diversified mix is preferred, consider adding a per-option upper bound."

**Underlying metrics (only when needed):** `quality.hypervolume_normalized` (0-1, healthy > 0.6) and `quality.spacing_cv` (healthy < 0.5) are still in the response. Use them only when the `frontier_quality` issues don't pinpoint the concern — e.g., a borderline-low hypervolume that didn't trip the diversity gate but you suspect under-search.

### Solution Quality Ladder

`frontier_quality` covers the first three rungs (returned, non-trivial, diverse). The full ladder extends them with downstream considerations the engine can't check on its own:

| Level | Gate | Source | If failing |
|---|---|---|---|
| **Feasible** | ≥1 solution exists | `frontier_quality.gates.frontier_returned` | Infeasible — route to optimization_strategy for constraint diagnosis |
| **Non-trivial** | Multiple solutions, ≥1 objective varies | `frontier_quality.gates.non_trivial` | Over-constrained or objectives all flat — relax constraints or consolidate objectives |
| **Diverse** | Spacing reasonable, no extreme allocation concentration | `frontier_quality.gates.diverse` | See `issues[]` — uneven coverage or single-winner allocation |
| **Robust** | Stable across parameter changes | Compare across scenarios / re-runs | Flag fragile solutions; suggest scenario analysis if not yet done |
| **Credible** | Domain-sensible | User confirms selections make sense | Scores may need recalibration, or a constraint is missing |

Present results once you've reached **Diverse**. Flag quality concerns but don't block exploration — users learn from imperfect results.

### Root Cause Taxonomy

When results look wrong, classify the symptom before suggesting fixes:

| Symptom | Likely causes | Diagnostic |
|---|---|---|
| All options allocated equally | Objectives too similar (correlated), constraints forcing even distribution | Check correlations between objectives; review cardinality constraints |
| Same options in ALL solutions | Other options dominated, data quality issues (missing or extreme scores) | Check scores for excluded options — are they consistently worse? |
| One option dominates (>60% allocation) | Intended (scores best on most objectives), or missing group_limit | Compare to reference point; ask if concentration matches user expectation |
| Very few solutions (<3) | Over-constrained, cardinality too tight | Identify which constraint to relax; check force_include/exclude interactions |
| Flat frontier (objectives barely differ) | Objectives correlated, not truly conflicting | Run Conflict Test from problem_framing; consider consolidating |
| Option never selected despite good scores | Dominated by a combination of others, or a constraint excludes it | Check if any constraint (force_exclude, exclusion_pair, group_limit) blocks it |

### Frontier Shape Interpretation

The frontier's shape tells a story about the tradeoff structure. After viewing the scatter plot from `explore tradeoffs` and marginal rates from `explore marginal_analysis`, characterize the shape for each conflicting objective pair:

| Shape | What to look for | What it means |
|---|---|---|
| **Linear** | Marginal rates roughly constant across the frontier | Smooth, predictable tradeoffs — each unit of A costs the same amount of B everywhere |
| **Convex** | Marginal rates decrease moving along the frontier | Diminishing sacrifice — later gains are cheaper. Users can push further than expected |
| **Concave** | Marginal rates increase (accelerate), inflection point detected | Diminishing returns — early gains are cheap, but costs escalate. Stop near the inflection |
| **Discontinuous** | Large gaps in the scatter plot, solutions cluster into 2+ groups | Fundamentally different strategy types with no smooth transition between them |

Read these signals from the data already returned by `tradeoffs` and `marginal_analysis`:
- **Scatter plot clustering** → discontinuous (gap between clusters = different strategy families)
- **Inflection point with high jump_factor** → concave (cost accelerates past the inflection)
- **No inflection detected, rates stable** → linear
- **Rates trending downward** → convex

Narrate the shape in domain terms: "The ROI vs Risk frontier shows diminishing returns — the first $10K of ROI costs only 2 risk points, but the last $10K costs 8. The inflection at Solution 4 is where the tradeoff steepens."

Don't force a classification when the data is noisy (common in binary/combinatorial problems with discrete jumps). Say "the tradeoff is uneven" rather than inventing a smooth narrative.

### Binding Analysis

The `tradeoffs` output includes a `binding_analysis` block: one entry per binding constraint with a `binding_fraction`, a `near_binding_count`, and a list of `shadow_prices` — rates, not deltas, derived from the frontier's own slope near the constraint.

Users often care more about *which constraint is limiting them and how much it's costing* than about which solution to pick. Shadow prices are how you tell them.

> When the frontier came from an exact continuous solve, `explore sensitivity` returns **solver-exact** shadow prices and reduced costs that supersede these frontier-derived estimates (see **Exact Sensitivity** below). `binding_analysis` is the fallback for heuristic (NSGA) and integer (MILP) runs.

**Translating the shape to language:**

| Constraint type | Shadow-price field | Template |
|---|---|---|
| `objective_bound` | `slope_per_unit_relaxed` | "Each unit of [constraint] relaxation looks worth about [slope] of [objective] on this frontier." |
| `cardinality` (cap or floor) / `group_limit` (cap, `≤ max`) | `gain_per_additional_slot` | "Adding one more [slot / group member] appears to unlock about [gain] of [objective]." |
| `group_limit` (floor, `≥ min`) | `gain_per_extra_member` | "Going one member above the [group] floor appears to buy about [gain] of [objective] — a positive gain says the floor isn't pinching." |

**Sign convention:** shadow-price values are **directional** in each objective's own frame — positive means the objective *improves* per unit of relaxation (in its own direction), negative means it *worsens*. For minimize objectives, a negative `gain_per_additional_slot` reads naturally as "this costs us more" — reframe rather than echoing the raw number. Example: Effort has `gain_per_additional_slot: -4` → *"Adding a slot also adds ~4 person-weeks of effort — a direct cost you trade for the other gains."*

Guidance:
- **Surface the highest-leverage constraint first** — sort by binding_fraction, then by what each slope is worth in decision terms (rate × a realistic increment — slopes are in each constraint's own units, so raw magnitudes mislead across constraints, as with the exact duals below). A constraint binding on 95% of the frontier with a large priced gain is the one to name.
- **Rates, not amounts.** Don't translate slopes into specific relaxation amounts ("+10%") — the slope is local to the binding edge and only describes marginal change. Let the user choose how much to negotiate: *"You'd gain about $X per $1 of budget relaxation — how much do you think you could get?"*
- **Note missing shadow prices.** When the `note` field is populated (e.g., no adjacent-cardinality solutions), say so: the constraint is binding, but the frontier doesn't give enough variation to estimate the gain. That's itself a useful message — it usually means the constraint is *so* tight that every solution sits on the cliff.
- **Route action back to problem framing** — binding analysis suggests which constraint to revisit, not that it must move. *"This is the limiting constraint. Is the limit truly fixed, or is it a target we could negotiate?"*

See `references/explore-diagnostics.md` for the full schema of `binding_analysis` entries and worked examples.

### Exact Sensitivity — Shadow Prices & Reduced Costs (solver duals)

`explore sensitivity` reports **solver-exact** duals when the frontier came from an exact continuous solve (`solve(solver="highs"|"cuopt")` on a continuous LP/QP shape). It is the *exact* counterpart to the frontier-inferred `binding_analysis` above, and answers two decision questions directly. The payload names the `optimized_objective` (the ε-constraint primary the duals are measured against) — anchor every shadow-price and reduced-cost sentence on that objective by name, never on an unnamed "the objective".

Keep its scope straight: this is the *local, marginal* lens — how the current optimum shifts under a small change in one input, valid only while the same constraints stay active. It is not scenario analysis, which bundles many changes into a named world and fully re-solves (so the optimal mix itself can change). Don't present a shadow price as a scenario outcome or vice versa. (See *Sensitivity vs Scenario Analysis* in `frontier://skills/optimization_strategy`.)

Every response is tagged `source`:
- **`solver_exact`** — duals read straight from the optimizer at each solution. The shadow price is the *instantaneous* marginal rate at that point — exact, not a slope fitted across nearby solutions. Prefer it whenever present.
- **`frontier_inferred`** — the fallback for heuristic (NSGA) or integer (MILP) runs, which have no exact duals; it returns the `binding_analysis` regression instead. Integer/MILP solutions never carry exact duals — say so rather than implying otherwise.

**Support by problem type** — solver-exact duals cover **both continuous paths** (QP and LP); only integer MILP falls back:

| Problem type | Exact path? | `explore sensitivity` |
|---|---|---|
| **QP** — proportional mean-variance (continuous, quadratic risk) | yes (HiGHS / cuOpt) | **`solver_exact`** — shadow prices + reduced costs |
| **LP** — proportional, purely linear (≥2 objectives, no risk term) | yes (HiGHS / cuOpt) | **`solver_exact`** — shadow prices + reduced costs |
| **MILP** — binary selection | yes, but integer | **`frontier_inferred`** — integer solutions have no duals |

On MILP, present the frontier-inferred estimate and name it as an estimate — integer solutions carry no duals. (Both continuous paths — QP and LP — carry solver-exact duals, per the table above.)

**The two reads:**

*Where to invest* — `where_to_invest` ranks constraint shadow prices by raw magnitude. A shadow price is reported in the solver's cost sense: the price a binding constraint charges the optimized objective per unit of tightening — positive always means tighter hurts, whichever direction the objective optimizes — and "per unit" means *that constraint's* unit, so rates on different levers (a % of budget, a return floor, a group cap) are different currencies, and rescaling a constraint rescales its dual. Read the ranking as a shortlist of binding levers; crown one the highest-leverage move only after pricing each in decision terms — rate × the increment the user could realistically negotiate. Translate into the constraint's own units:
- *"At this point each extra unit of required return costs about [shadow] more risk — the marginal price of return here."*
- `frontier_shadow_price_trend` shows that price along the frontier. Rising = diminishing returns: *"Return gets steadily more expensive — the first units are cheap, the last cost several times the risk."* Name where it steepens. At portfolio scale the curve arrives evenly downsampled (`frontier_shadow_price_trend_elided` says how many points were thinned; endpoints kept) — narrate the shape, and fetch `sensitivity` at a specific `solution_id` for any thinned point.

*Near-misses* — `near_misses` lists options the optimizer left out (allocation 0), ranked by reduced cost (smallest first = closest to entering; this ranking is safe to use as-is — every option's reduced cost is in the same currency, the optimized objective per unit of the shared budget). The reduced cost is roughly how far that option's contribution must improve before it becomes competitive — a marginal rate, not an exact admission price:
- *"[Option] just missed the cut — it would enter the mix if its return improved by about [reduced_cost] (or its risk/correlation dropped equivalently)."*
- The smallest-magnitude near-miss is the one to watch: a small reduced cost means a small change in assumptions pulls it in.
- An unheld option *absent* from the list may be a degenerate tie — reduced cost ≈ 0, meaning it could swap in at essentially no objective cost. Absence from `near_misses` means "no strictly positive gap reported" — the option may still be at the doorstep.

**Sign convention (carries meaning — don't drop it):**
- An **excluded** option (held at 0) has reduced cost **> 0**: "would need to improve by X to enter." This is the near-miss read.
- A **capped** option (pinned at its max-allocation limit) has reduced cost **< 0**, reported under `capped_options`: "the cap is binding — it would take more if allowed." That points at the allocation cap as a *where-to-invest* lever, not a near-miss.
- A **floored** option (held at its `allocation_bound` minimum) has reduced cost **> 0**, reported under `floored_options`: it's being carried above what it would earn, and the reduced cost is *the price of the commitment* per unit of allocation. Present it as what the contract costs, not as a near-miss: *"holding [option] at its [floor]% floor costs about [reduced_cost] per point of allocation."*
- A `where_to_invest` entry with role **`model_bound`** prices a model-level `objective_bound` constraint (e.g. a strategic mandate): its price reads the same cost-sense way (tightening the mandate by one unit worsens the optimized objective by the shadow price; relaxing buys roughly that back) — often the single most negotiable lever in a rationing decision.

**Guidance:**
- **Prefer `solver_exact` over `binding_analysis` when both exist** — the dual is the exact local rate; the regression averages across nearby points and overstates the rate on a curved frontier. If you've quoted a frontier-inferred slope and an exact run is available, refresh with the exact number.
- **Anchor in the reference solution.** Duals are local — they describe the rates *at* the reference point (default: balanced; pass `solution_id` to move). Say which point you're quoting; a shadow price without its solution is phantom precision.
- **No validity range is reported.** The dual is the marginal rate *at* this optimum; Frontier does not surface the interval over which it stays valid (classic RHS ranging). Present it as a local rate — *"about X per unit here"* — and never *"this holds until [bound] reaches Y,"* which the data doesn't support.
- **Price a real move with a re-solve, not the dual.** Even at a clean, non-degenerate optimum the dual is the rate at zero relaxation; over a finite increment the active set can shift and the realized gain land below rate × increment. The dual picks the lever — before quoting a number the user will negotiate with, relax the constraint by the increment actually on the table, re-run the exact solve, and report the realized difference. Frontier makes that check one solve.
- **Treat any single dual defensively — you can't tell when it's degenerate.** A solver-exact dual is exact for the *optimum the solver returned* (cuOpt's first-order method reports duals with no simplex basis at all), but real optima frequently sit at degenerate corners (several constraints binding at once), where the dual is one of several valid prices — one-sided and non-unique, most precise-looking exactly where it's least reliable. Frontier surfaces no degeneracy flag, so you can't know which case you're in: lead with *which* levers bind and the direction each pushes, present any single shadow price as a rough local rate, and let the re-solve above price it. The **QP** path buys extra stability only when the interaction matrix makes it *strictly* convex — user-built covariance estimates are routinely near-singular (the engine projects them to solvable PSD for the inner solve) — so give QP duals the same defensive read.
- **Route action back to framing.** A large shadow price says where relaxing *would* pay — not that the limit must move. *"Return is the expensive axis here. Is your return floor a hard requirement or a target we could negotiate?"*
- **Continuous only.** No exact duals for integer/MILP selection — on a binary problem `source` is `frontier_inferred`; present its estimate and name it as an estimate.
- **A compact table can help when several duals are in play (optional, secondary).** When `where_to_invest` ranks several binding constraints or `near_misses` several close options, a small table — lever (constraint / option), the shadow price or reduced cost, and a one-line implication — often reads more clearly than a run of sentences. Keep it subordinate to the narrative: lead with the top lever (as priced above) in words; table the rest only when it genuinely aids the decision, and still anchor every number in its reference solution.

### Reading the Certificate (explore certify)

`explore certify` audits the exploratory NSGA frontier against an exact overlay (`solve(solver="highs"|"cuopt")`). It answers the **trust** question — *"how do I know nothing better exists, and are my picks provably near-optimal?"* — the auditor counterpart to the produce-side narration in `optimization_strategy`. Translate the pre-built `recommendation` / `next_steps`; don't echo them. It certifies *per-scalarization optimality + dominance + coverage* — never that the whole frontier is provably optimal (no single optimum exists for a multi-objective problem), so keep the language scoped. Five blocks:

*Dominance audit* — `dominance_audit.nsga_dominated_by_exact` / `nsga_dominated_fraction` / `examples`: the NSGA "Pareto" points the exact frontier strictly beats — heuristic slack the EA presented as efficient. The fraction is the headline trust number:
- *"Exact audited the explored frontier: [N] of [M] points ([fraction]) are actually dominated — plans that looked efficient, but a provably better mix exists. Here's one: [examples]."*
- Zero dominated is itself a result: *"Exact confirms every explored point is on the true frontier — the EA's picks hold up."*

*Coverage* — `coverage.reclaimed_fraction`: the hypervolume the exact overlay reclaims over NSGA alone — the magnitude behind the dominance count, and the trust number that grows with problem size. Lead with it at scale; near-zero is the honest small-instance result (the heuristic already covers the frontier — exact confirms, doesn't expand). When `coverage` is `null` the combined front is degenerate (a flat axis) — skip it and lean on the dominance fraction.

*Completeness* — the mirror audit: coverage asks what exact reclaims over the heuristic; `completeness` asks the inverse — does the heuristic hold hypervolume the exact sweep never sampled? `verdict: "complete"` folds into the trust story (the overlay is not just sound but covers everything the EA found). `verdict: "under_covered"` names `gap_regions` — clusters of witness points (by `witness_solution_ids`, self-certifying) with a bounding box in objective units and each region's `reclaimed_share`. Present it as an honest boundary of the certificate, not a failure: *"the certified overlay misses regions your explored frontier already holds ([reclaimed_fraction_over_exact] of the combined hypervolume) — a targeted fill closes them."* The next move is `solve(solver=…, scope="fill_gaps")`: it re-solves **only** the witnesses and merges; a gap whose solve hits its budget comes back named in the response's `fill.unfilled` — discarded and reported, so those regions honestly stay heuristic-best rather than absorbing an uncertified incumbent. Re-run `explore certify` after a fill for the updated verdict.

*Invariant* — `invariant.holds`: NSGA should dominate **no** exact point. This is a *soundness check on the overlay* — an exact point is optimal for its scalarization (to the solver's gap), so NSGA shouldn't beat it; when it does, that's the rounding footnote below, not a heuristic win. Confirm it holds, but don't lead with it as a quality win — the substantive trust numbers are the dominance fraction and the coverage gain:
- *"NSGA dominates no exact point — the overlay is sound; exact can only confirm or improve."*

*Corner sharpening* — `corner_sharpening` is per-objective (`nsga_best`→`exact_best`, `improvement`, `status` ∈ `sharpened` | `matched` | `under-sampled`); `headline_corner` names the one that matters most (usually the convex risk/variance corner the decision turns on):
- *"Exact sharpens the [headline_corner] corner the decision hinges on: [nsga_best] → [exact_best]. The EA got close; exact nails it."*
- `matched` corners: the EA already found the optimum — confirm or stay silent.

**Two easy misreads — read them this way:**
- **A false invariant (`invariant.holds = false`) is the QP's integer rounding.** `exact_dominated_by_nsga > 0` on a QP means whole-percent allocation rounding edged out the continuous optimum (the `note` says so) — present it as a rounding footnote, not the EA beating the exact solve. (MILP corners, integer by construction, never show this.)
- **An `under-sampled` corner (negative `improvement`) is an EA-budget signal — and should now be rare.** Every overlay anchor-samples each objective's extreme, so this mostly appears on a cardinality/group-capped mean-variance QP (which defers corners to its seeded search). Raising the budget closes it — present it as "more search would tighten this," not "exact is worse here."

**The certificate binds the model, not the world — say where it stops.** Every scope above is *internal* — it bounds what "optimal" means within the stated model (per-scalarization, not global; weakly efficient under ties; degenerate duals). The *external* boundary is the one users miss: the proof certifies each pick optimal *for the model as stated* — these objectives, these scores, these scenarios — and says nothing about whether that model is the right one. Whether the objectives are the ones that matter, the scores honest, the scenarios the futures that play out is **validation** (the V&V sense — does the model reflect reality — *not* `solve validate`'s feasibility gate), and it stays the user's judgment, not the solver's. So when you hand over a clean certificate, name the edge rather than letting "proven optimal" read as "proven right" — *"each pick is provably optimal for the numbers you gave me; whether those are the right numbers is your call, worth a look before you commit."* It's the same trust the dominance/coverage numbers earn — don't let it leak into a trust the model itself hasn't.

**Next move** — after a clean certificate, present the certified frontier with confidence, calibrated to its `exact_certified` flag: when `true`, every point is optimal, not heuristic; when `false` (a bounded binary run), say "optimal to the solver's 0.1% gap" and offer `exact=true` for zero-gap certificates — the certificate's own `next_steps` carries the matching phrasing. On a **continuous/QP** problem, offer `explore sensitivity` for the duals/explainability layer; navigate the overlay itself with `explore … source="exact"`. **Confirm you got the overlay, don't assume it:** every `explore tradeoffs`/`solutions`/`solution` result echoes `frontier_source` — its `kind` must read `exact`. If it reads `heuristic` (or flags `exact_overlay_available`), the `source` was dropped before reaching the engine — typically a stale MCP tool-schema cached at session start, or an older server process still running — so you're quoting the *heuristic* frontier, not the certified one. Trust the label over your request; reconnect (a fresh session) before presenting anything as certified.

**Its sibling for the feasibility question.** Certify audits *optimality* of the frontier; it doesn't answer "could *any* feasible plan breach this guardrail?" or "is this constraint set even satisfiable?". For those, reach for `explore audit` — the feasibility-side auditor over the whole feasible region (see *Reading the Audit (explore audit)*).

### Denoting Certification — Prose & Tables

Once an exact overlay exists, *which* solutions are certified becomes decision-relevant — so carry that denotation into your words and tables, don't let it live only in the chart. The provenance is traceable: `frontier_source.kind` labels the frontier you're presenting, and `explore certify`'s `dominance_audit` (the same heuristic points the frontier scatter's `exact_overlay.dominated_ids` fades) names the ones an exact solve beats.

- **Mark the certified solutions.** When the frontier you're presenting is exact (`frontier_source.kind == "exact"` — e.g. after `explore … source="exact"`), every solution shown is optimal for its scalarization. Say so, and in a solutions table add a short status marker (a ✓ or an "exact-certified" column) rather than leaving provenance implicit. Keep the claim scoped to per-scalarization optimality + dominance — never upgrade it to "globally best" (no single optimum exists for a multi-objective problem).
- **Flag the superseded points.** When you present the heuristic frontier while an exact overlay exists, the points the exact front dominates are *not* certified and are provably beaten — name them by id, and don't carry them forward as efficient picks. *"#7 and #12 read as efficient, but an exact solve beats them at their own cost — leave them off the shortlist."*
- **Keep prose and chart telling one story.** The web UI draws certified points solid and the not-yet-certified field faded; let your language match — certified reads as confirmed/optimal, heuristic as provisional/approximate. A reader moving between your table and the chart should meet the same distinction in both.
- **Don't over-mark.** On a plain heuristic frontier with no exact run there is nothing to certify — don't add an empty status column or imply a certification you don't have. The marker earns its place only once an exact overlay exists; until then, the honest frame is "approximate frontier, exact audit available if you want it."

### Reading the Audit (explore audit)

`explore audit` is the **feasibility-side sibling of certify**. Certify asks "is each frontier point optimal?"; audit asks "what's true across the *whole feasible space*?" — every feasible plan, not just the non-dominated ones the frontier shows. It answers the **governance** question — *"can I guarantee no feasible plan violates this, or is this constraint set even satisfiable?"* — that the frontier structurally can't. It needs no prior solve: it reasons about the model's feasible region directly, so it's also the way to feasibility-probe a constraint set *before* spending a solve. Binary problems, HiGHS. Translate the verdict; don't echo the raw fields.

The payload carries a `verdict`, the `audited` echo (the property, or "feasibility of the current constraint set"), the `feasible_region` it's conditional on (approach + constraint counts-by-type + a `constraints_fingerprint` — the same fingerprint `model get` echoes, so a `holds` is checkable against the current model), an optional `witness` (a concrete plan), and a pre-built `recommendation` / `next_steps`. A **compound guarantee** (a list of properties) is proven as a conjunction — `holds` only when every conjunct holds; the payload adds a per-`properties` breakdown and, on a violation, names the `violated_property`, so present *which* guarantee failed, not just that one did. An unfit shape (non-binary, an `objective_bound` on a non-sum objective, or HiGHS not installed) returns a tool `error` instead — binary + HiGHS only in v1, the same hard-decline `solve`'s exact gate uses, so there's no "unsupported" verdict to present. (A non-sum objective that no bound references doesn't block the audit — feasibility never touches it.) Read each verdict this way:

| `verdict` | What it means | How to present |
|---|---|---|
| `holds` | The property holds for **every** feasible plan — the negation is infeasible across the whole region, *proven*, not sampled | Lead with the guarantee: *"Across every feasible plan, [property] holds — that's a proof over the whole feasible space, not a spot-check."* Scope it: the guarantee is **conditional on the current constraints** — `feasible_region` pins their identity via `constraints_fingerprint` (compare it with `model get`'s before re-asserting); if it differs, re-audit. |
| `violated` | At least one feasible plan breaks the property — the `witness` is a concrete counterexample | Present the witness as a **plan**, entity-native: *"[property] isn't guaranteed — e.g. this feasible plan breaks it: selects [witness.selected_options], reaching [witness.objective_values]."* Then the lever: if it *must* hold, encode it as a hard constraint (it currently isn't one). |
| `feasible` | A feasible plan exists (probe) | *"The constraints are satisfiable — here's one plan that works."* Proceed to `solve`. |
| `no_feasible_plan` | No plan satisfies the constraints (probe) | *"As stated, no plan is feasible — the constraints over-constrain."* This is the exact form of `validate`'s pre-solve check; route to optimization_strategy → *Infeasibility Response* to find the tightest constraint. |
| `holds_vacuously` | The property "holds" only because the feasible region is empty | Don't report it as a win — surface the empty region first (probe feasibility with no property). |
| `inconclusive` | The solve hit its time limit | **Never read as a pass.** Say it's undetermined and suggest simplifying the model or narrowing the property. (The cardinal discipline: a non-result is not a proof.) |

**An empty-plan witness is itself a finding.** When `witness.selected_options` is empty, the model admits "select nothing," and that's what broke the property — usually a missing cardinality floor (`min ≥ 1`). Name it: *"the audit's counterexample is the empty plan — your model allows selecting nothing; add a cardinality minimum if at least one pick is required."*

**Scope the language.** `holds` is a guarantee over the feasible region *as currently constrained* — strong, but not "globally true for all time." Keep it traceable (it's the negation-infeasibility result), and never inflate `feasible` (one plan exists) into `holds` (all plans comply) — they're different claims.

### Marginal Analysis Interpretation

The `explore marginal_analysis` action returns structured data for each conflicting pair: `rates` (per-step cost ratios) and optionally `inflection` (with `solution_id`, `position`, `jump_factor`). Interpret these:

- **Rates**: "Moving from Solution 2 to Solution 3 costs [X] units of [B] per unit of [A] gained"
- **Inflection point**: When `inflection` is present, use the anchored template from Presentation Order: *"[A]-vs-[B] inflection at [A=value, B=value] — jump factor [X]×. Past this point, each extra unit of A costs roughly X× more B than before."* The objective values come from the referenced solution; the jump magnitude comes from `inflection.jump_factor`.
- **No inflection**: "Marginal costs change gradually — the frontier offers smooth tradeoffs throughout"
- **Exchange rates**: Anchor tradeoff conversations with concrete numbers: "Each additional $10K revenue costs 2 points of satisfaction"

Inflection points also appear in `tradeoffs` output as `inflection_point_candidates` — these are the same solutions, pre-computed for convenience. Use them in the initial presentation alongside balanced and extremes, without requiring a separate `marginal_analysis` call.

### Frontier Visualization

The explore tool renders ASCII visualizations inline with each response. The format depends on the action:

- **`tradeoffs`** → **2D scatter plot** of the most conflicting objective pair (highest |negative correlation|). Shows all frontier solutions as `·` on a grid, with labeled markers for extremes (`●`), balanced (`⚖`), and inflection points (`◆`). Reveals WHERE solutions cluster, WHERE tradeoff costs jump, and the shape of the frontier. Axes are direction-aware — "better" always points right/up.

- **`compare`** (by `solution_ids`, or `signatures` for the curated set) → **Parallel coordinates** — one row per solution, one column per objective (capped at 6 most differentiating), each normalized to its range with direction-aware orientation. Shows at a glance which solution wins on which objective and how they trade off. Use this for 2-6 candidate solutions or curated shortlists.

- **`scenario_results`** → **Range comparison bars** per objective per scenario, showing how the feasible range shifts across futures, plus **option robustness table** (ranked by importance with tiers) and expected values.

- **`scenario_frontiers`** → **Per-scenario parallel coordinates** — every scenario's frontier overlaid as one plot, colored by scenario. When scenarios exist, reach for it to *show* how the achievable tradeoffs shift across futures, paired with your prose narration of the shift.

These visualizations are generated by the tool — you don't need to construct them. Present them as-is along with your narrative interpretation. Supplement with brief comparison tables when comparing only 2 solutions, where a table may be clearer than parallel coordinates. When the frontier is exact or carries an exact overlay, the scatter denotes certification (certified points solid, the not-yet-certified field faded, dominated points faintest) — carry that same denotation into any solutions table you build, per **Denoting Certification — Prose & Tables**.

### Mining the Solution Set

`explore composition` mines the frontier — or a curated subset, via `solution_ids`/`signatures` — for structure that one-at-a-time comparison misses. It answers "what do the strong solutions have in common, and how do they really differ?" — the knowledge-discovery counterpart to visualization and scenario analysis. Read its blocks this way:

- **`option_selection`** — each option's selection count and % across the set (also on `tradeoffs`, and per-option on `solution`/`solutions` via `option_context`). Translate to *consensus vs distinctive*: an option in nearly every solution is a no-regret backbone ("every efficient plan funds X"); a rare one marks a distinctive bet. Lead with that split, not the raw list.
- **`design_principles`** — statements that hold across the set: `always`/`never` (an option in all / none), `co_occurs`/`substitutes` (options that travel together or replace each other), `region_bound` (an option that appears only in the high- or low-[objective] region). Narrate them as the decision's structure: "across every strong plan, A and C come as a pair; B only shows up once you push cost down."
- **`clusters`** — strategy families in *decision space* (grouped by which options/allocations, not by objective values), each with `defining_options` and an `objective_signature`. Present them as named archetypes to choose between ("three families: a low-cost core, a balanced mix, a growth-tilted set"). Because they're decision-space clusters, two families can sit at nearly the same objective point yet be built very differently — when they do, the choice between them is close to free, so pick the one easier to implement.
- **`feedback_rules`** — when the user has rated solutions, rules separating the liked from the disliked ("everything you rejected included Z"). Treat a clean rule as a *candidate latent constraint*, not a verdict — surface it, confirm it, and route it back to framing (see `frontier://skills/problem_framing` → *Post-Solve Constraint Discovery*). `available: false` just means not enough rated-both-ways signal yet.

Reach for it once the user has seen the frontier and is asking "what's actually driving these choices?", or while building a shortlist — composition over the curated subset sharpens what the finalists share and where they diverge.

### Run Diff Interpretation

When the user iterates (changes constraints, adds options, adjusts scores) and re-runs, use `explore compare_runs` to narrate what changed:
- **More solutions appeared**: "Relaxing the constraint opened up new tradeoff space"
- **Fewer solutions**: "The new constraint eliminated solutions that relied on [X]"
- **Option gained coverage**: "Feature Y now appears in 60% of solutions, up from 20% — the constraint change favors it"
- **Option lost coverage**: "Feature Z dropped out of all solutions — it can't compete under the tighter bound"

Always connect the change to the user's action: "You added a force_include on SSO. That caused..."

### Reference Point Narration

When reference points exist, the explorer includes distance-to-reference metrics. Use these templates to contextualize:

**Baseline comparison** (status quo → solution):
- "Your current allocation achieves [X cost] and [Y quality]. This solution achieves [Z cost] and [Q quality] — that's a [+/-]% improvement on cost at [+/-]% quality."
- "Compared to your current portfolio, this saves [$X] but reduces satisfaction by [Y] points."

**Aspirational comparison** (solution → target):
- "Your target was [$M cost] and [N% quality]. This solution reaches [X% of cost target, Y% of quality target]. The gap is primarily in [which objective]."
- Frame gaps as actionable: "The quality gap from your target is small — achievable with one more constraint relaxation."

**Multi-solution framing** (when presenting 2-3 solutions against references):
- "Solution A is closest to your current state (minimal disruption). Solution B reaches closer to your target (bigger change needed). Solution C balances both."

If a solution meets all aspirational targets, note that objectives may not truly conflict at these target levels.

### Scenario Results Presentation

When scenarios are defined and optimized, use `explore scenario_results` to present:

1. **Robust options by tier** — `option_robustness` returns options ranked by importance (frequency × avg allocation weight; binary selections carry no allocations, so their rows have frequency only and rank by it), with tiers. Tiers key on the WORST per-scenario frequency, not the shown average (the payload's `tier_rule` states it) — an option can average high yet tier lower because one future benches it:
   - **Core** (>50% frequency in every scenario): safe bets regardless of which future materializes. Lead with these.
   - **Common** (>25% worst-case, or present in all scenarios): likely to appear but not dominant — secondary picks.
   - **Marginal** (the rest): conditional picks dependent on scenario view.
2. **Scenario-specific opportunities** — options in `scenario_specific` that excel in particular futures (grouped by scenario name → list of options). Frame: "If you believe [scenario] is likely, [option] becomes attractive."
3. **Frame around risk tolerance** — "The core options protect your downside across all futures. The marginal options offer upside in specific scenarios. Which matters more?"

Each `per_scenario` entry restates what that scenario `varies` and a `held_fixed` line (everything else inherits the base model) — read them back so the user knows exactly what the comparison isolates, and flag it if a scenario varies more than its name implies (a replace-all `constraint_overrides` is the classic case). When a scenario carries `motivated_by`, cite it: *"this scenario quantifies the [lever] that sensitivity ranked highest."* And when one scenario flips feasibility or steps an objective sharply where its neighbors don't, treat the break as a finding about the modeling assumptions until verified — name the choice creating it, and if a plausible alternative reading removes it, present both readings.

Present the top 5-8 from `option_robustness` with their tier, frequency, and (proportional shapes) avg weight. Don't list every option — the tail is noise. Start with core vs scenario-specific, then drill down if the user asks. If the tiers are flat — options appear at similar frequency across every scenario — the scenarios aren't differentiating the decision; say so and suggest simplifying the scenario set rather than reading false signal into it.

**Expected values caveat**: The `expected_values` in scenario_results are ideal-point values (best-per-objective across scenarios, probability-weighted). No single solution achieves all simultaneously. Present them as "theoretical ceiling" when useful, but always caveat.

**Tail risk framing** — the `scenario_risk` block reports `expected`, `worst_case`, `best_case`, and `cvar_<α%>` per objective. CVaR is the probability-weighted mean of outcomes *in the worst α fraction of scenarios* — it answers "when things go wrong, how bad is 'wrong'?" Use it to shift the conversation from averages to downside:

- **Lead with expected when the user is comparing strategies on value** — that's the "normal case" frame.
- **Lead with CVaR or worst-case when the user is risk-averse, regulated, or near a constraint cliff** — e.g., capital preservation, compliance floors, safety. Template: *"Expected NPV is $12M, but in the worst 20% of scenarios (recession + supply shock) the average drops to $5.8M. Is that tolerable?"*
- **Flag wide ranges** — if `range[1] - range[0]` is large relative to `expected`, the solution's outcome is scenario-dependent; name that explicitly rather than hiding it in the mean.
- **Control α when the user tells you what "bad" means.** The default α is 0.2 (worst 20%). Pass `cvar_alpha=0.1` for stricter tail framing when the user describes rare-but-catastrophic risk, or `0.3–0.5` when they worry about a broader downside.

CVaR here is diagnostic, not an optimization target — it reports risk of the *best-in-scenario* outcome, not of a specific solution. Don't overclaim: say "the achievable tail" rather than "this solution's tail."

**Regret — the hindsight lens.** `scenario_results` also returns a `regret` block: scenario *minimax-regret* per solution, plus the `minimax_choice`. Regret asks a different question than CVaR. CVaR is "when things go wrong, how bad is the outcome?"; regret is "how much worse than the best I *could* have chosen, had I known the scenario?" Lead with regret when the user will second-guess the call afterward — one-shot, high-stakes, hard-to-reverse decisions — and present the `minimax_choice` as the regret-robust pick ("whatever future hits, this is never far from the best you could have done"). A solution flagged `feasible_in_all: false` and `feasible_in_ranked: false` is the headline, not a footnote: it looks strong on the base frontier but breaks under at least one ranked scenario — say so before recommending it. When `feasible_in_ranked` is true, the only failure is an excluded wipeout scenario (one no base plan survives — see `wipeout_scenarios`): present the `minimax_choice` normally and surface `wipeout_note` as a separate model-level finding, since penalizing one plan for a scenario that kills every plan says nothing about the choice. This computed metric is distinct from the conversational "regret framing" elicitation device under *Preference Learning* below — both are useful; don't conflate them.

### Sensitivity Intuition
Flag fragile solutions:
- "This solution barely makes the cut on your effort constraint. If effort estimates are off by 10%, it might not be feasible."
- This builds trust and helps the user think about robustness.

### Diagnostic Patterns

The `diagnostics` array in solve output contains structured signals (not pre-written messages). Read the `pattern` field and narrate in domain terms:

| `pattern` value | Structured fields | What to tell the user |
|---|---|---|
| `zero_solutions` | — | Problem is infeasible. Route to optimization_strategy for constraint diagnosis. |
| `clustered_solutions` | — | "Your solutions are nearly identical — objectives may not truly conflict. Are [X] and [Y] measuring the same thing?" |
| `low_variation_objective` | `objective`, `relative_range` | "[Objective] barely varies across solutions (range [X]%). It's not driving differentiation — consider dropping it or re-scoring." |
| `option_never_selected` | `option` — or, at portfolio scale, one summary entry with `count` + an `options` head | "[Option] doesn't appear in any solution. Check if it's dominated or blocked by a constraint." For the summary form, present the count and route detail to `explore composition`. |
| `binding_constraint` | `constraint`, `extreme_value` | "[Constraint] is binding (solutions push right up to the limit). Relaxing it would open new tradeoff space." See the **Binding Analysis** section for shadow-price framing when `tradeoffs` is available. |

Also watch for these patterns in the data even when no diagnostic is emitted:

| Pattern | Detection | Action |
|---|---|---|
| **Bunched at extremes** | Bimodal distribution in scatter, sparse middle | Investigate the gap — it may be structural (different strategy families) |
| **Missing scores** | `scores_complete < 1.0` in model status | Collect or estimate remaining scores before solving |

**Presentation guidelines**: Be specific, not generic. Limit to 1-2 observations based on actual patterns. Frame as questions: "You might consider..." Skip entirely if results look healthy.

### Recommendation Calibration

How confidently you recommend depends on how much you know about what the user wants:

| Signal strength | Evidence | Response style |
|---|---|---|
| **Strong** (2+ signals) | User stated preferences + gravitates toward specific solutions + gave constraints reflecting priorities | Confident: "Based on your priorities, Solution 3 fits well because..." |
| **Moderate** (1 signal) | User mentioned one priority, or selected one solution to look at | Recommend with caveats: "This aligns with what you've told me, but I'd like to confirm — do you care more about X or Y?" |
| **Weak** (no signal) | User hasn't expressed preferences | Don't recommend. Instead: "All 8 solutions are valid tradeoffs. What matters most to you: revenue or effort?" |

Never recommend without signal. When in doubt, present a concrete tradeoff between two solutions to draw out preferences.

### Preference Learning

Behavioral signals often reveal preferences more reliably than stated priorities. When you notice a pattern — gravitating toward certain solutions, repeatedly asking about a specific option, avoiding extremes — surface it rather than silently inferring: *"I notice you keep returning to the lower-cost options. Should we focus there?"* Let the user confirm before treating it as a signal.

**When you detect waffling** (user bounces between solutions, keeps returning to the same few):

1. **Normalize**: "Difficulty deciding is normal — you're discovering your real preferences through exploration."
2. **Surface the gap**: "You mentioned cost and speed were equally important, but you keep gravitating toward low-cost options. That's useful — it suggests cost may matter more."
3. **Use regret framing**: "If you chose this option and later regretted it — what would the reason be?"
4. **Offer a next step**: "Would you like me to find other solutions similar to the ones you keep returning to?"

### Common Misconceptions

When users express these, include the corresponding correction:

| User says | What to explain |
|---|---|
| "Which is the best one?" | "All N solutions are equally optimal — each is best at a different tradeoff. Which tradeoff matters most to you?" |
| "Results aren't good" / "way off from targets" | "The gap between targets and results reveals genuine tension in your objectives — that's insight, not failure. It shows where your goals are most in conflict." |
| "Is it broken?" | The tool is working. Results reflect the real constraints of the problem. |
| "Add more objectives" | "More objectives obscure key tradeoffs. Are any of these correlated? Can we consolidate?" |

### Solution Curation

Curation is how users build a decision set from the raw frontier. Use `explore curate` to add solutions to the curated set, each with a name. Curated solutions persist across runs — they're the user's working shortlist.

**Curation is a preference signal — including its absence.** What a user curates reveals real-world considerations — political viability, team enthusiasm, strategic alignment — that scores and constraints don't capture. Treat curation choices as evidence of the user's actual priorities, potentially more reliable than their stated priorities. When a user curates a solution that's suboptimal on their stated priorities, probe why. When a user *hasn't* curated anything after exploring multiple solutions, that's also signal — the frontier may not contain what they're looking for. Surface it: "None of these grabbed you — what's missing? That might reveal a constraint or objective we haven't captured."

**Curation belongs to the user.** Surface interesting candidates — extremes, balanced, inflection points — then ask which ones resonate and what the user would name them. Present candidates, then pause for the user to decide.

**Relay quality flags verbatim.** Curate/list/export responses carry a `quality` gate per finalist (`GOOD` / `WARNING` / `DEGENERATE`, with the triggering check named in the user's terms — empty plan, single-option concentration, allocations pinned at bounds), and `explore certify` flags degenerate exact points under `quality_gates`. Optimal ≠ actionable: pass the flag and its message to the user; the finalist stays in the set and the keep-or-fix call is theirs.

**Guide users from frontier to shortlist:**
1. After first solve: present the extremes + balanced with ASCII visualizations, then **stop and ask** which ones the user wants to curate and what they'd name them
2. After objective ranking: identify 3-5 candidate solutions that span the user's priority space, present them, **ask the user to pick and name** — don't curate automatically
3. After each re-solve: report curated solution survival — "Your 'Conservative Pick' still appears in the new frontier, but 'Growth Bet' was eliminated by the tighter effort constraint"
4. Once curated set has 3+ solutions: shift presentation to the curated set, not the raw frontier. Use custom names in all narration.

**Naming guidance:**
- Encourage strategy-descriptive names: "Conservative Pick", "Growth Bet", "Balanced Middle", "High-Risk High-Reward"
- Names should capture the *why*, not the *what*: "Low Effort" is ok but "Quick Wins" is better — it implies the strategy

**Feedback loop:** Use `explore feedback` to record ratings and notes on solutions. Feedback accumulates across re-runs via `content_signature` — the preference history travels with the solution. Reference accumulated signals: "You've consistently rated this option highly across 3 rounds."

**Cross-run and cross-scenario tracking:** After re-optimization, check curated solutions against the new frontier (`explore curated`). Report survival, explain eliminations, and connect to scenario robustness. When a curated solution is eliminated, explain which change caused it.

**Robustness as curation signal:** After scenario analysis, core-tier options (from `option_robustness`) are natural curation candidates. Surface them: "These core options survive at high frequency (and, on allocation shapes, weight) across all scenarios — worth curating as safe bets." When a curated solution contains mostly core-tier options, note it as robust; when it relies on marginal or scenario-specific options, flag the conditional risk.

**Presentation framing:** Once the curated set has 3+ solutions, it IS the decision set. Present curated solutions first, the full frontier as background. Frame the final question around the curated set using custom names.

**Handoff via export:** When the user is ready to leave Frontier — building a deck, writing a memo, sharing with a team — offer the curated export — `explore curated format=…`. Two formats:

- `format="markdown"` (default): pipe-aligned table, paste-ready for docs and messages.
- `format="csv"`: for spreadsheets or further tooling.

Both emit raw curated solutions (names, signatures, objective values, selected options or allocations) — no narrative. Your job is the narrative; the export is the data. Offer it naturally at handoff moments: *"Want me to export the curated set as a markdown table you can paste into your doc?"* Don't auto-export — the user decides when they're ready to take work out of the session.

### Stakeholder Writeup & the Why-Triplet

When the user has **decided** — a solution chosen, or a curated set locked — and needs to carry it to others (a memo, a deck, a review), shift from exploration to a **stakeholder writeup**. This is a Confirming/handoff move, *not* an exploration one: reach for it only once the decision is made, because its structure presumes a single chosen plan — using it mid-exploration would fight *Never Say Best*. The export (`explore curated format=…`) is the data table; this is the narrative that frames it.

**The six-part order** (each part traces to returned data, never to assertion or solver internals):

1. **Decision Question** — open with the user's original question (the problem `context`). *"We needed to choose how to allocate across these markets."*
2. **What Was Decided** — the chosen plan(s): name the selected options/allocations and their objective values. Frame it as a *chosen tradeoff among efficient options*, not "the best": *"We chose [name]: [objective values] — it trades [X] for [Y]."*
3. **Why These Choices** — the key drivers, read from the frontier: which options are consensus picks (`composition` always/consensus), which constraints bind and at what rate (`binding_analysis` / exact shadow prices), why this point over its neighbors (dominance, marginal rates).
4. **Confidence** — how we know it's sound: the **certificate** if an exact overlay exists ("every point optimal, not heuristic" when its `exact_certified` flag is true; "optimal to the solver's 0.1% gap" when it isn't), the frontier quality signals, and scenario robustness / CVaR if scenarios were run. This is the multi-objective replacement for a single optimality gap — *don't* report a lone "gap %". Where a governance guarantee matters, an `explore audit` "holds" result belongs here — with its epistemic basis stated (proven over every feasible plan vs. inconclusive-is-not-a-pass). Each finalist's `quality` gate belongs here too: relay a WARNING/DEGENERATE flag and its message verbatim rather than smoothing it over — optimal ≠ actionable.
5. **Impact** — the objective values in business terms, against a baseline or target when a reference point exists: *"$X cost — a Y% reduction vs the current allocation."*
6. **Next Steps** — validate-with-experts / implement / stress-test / re-audit, as fits the decision. When `explore sensitivity` returned `suggested_scenarios`, carry the top one here as the named stress-test ("duals rank, scenarios quantify"); any unresolved quality flag becomes an explicit follow-up rather than a silent omission.

**The why-triplet** — the three questions every decision-maker asks. Answer each from returned data, in the user's entities, never from solver internals:

- **Why was X selected?** → `composition` consensus + its scores. *"X is in every efficient plan — it's the lowest-cost way to clear the quality bar."*
- **Why was Y excluded?** → it's Pareto-dominated, or a near-miss. *"Y costs more for no objective gain"* (dominance), or *"Y would enter if its [objective] improved by ~[reduced cost from `explore sensitivity`]"* (near-miss).
- **What's blocking better than Z?** → the binding constraint and its shadow price. *"The cardinality cap — each extra slot buys ~[gain] of [objective] (`binding_analysis`)."*

**Discipline:** every number traces to a tool result (Traceable Claims); "Why excluded" leans on Pareto dominance, which has no single-objective analogue; and the chosen plan stays framed as *one efficient tradeoff the user selected*, not a solver verdict. The scaffold organizes what the exploration already surfaced — it doesn't invent new claims.

