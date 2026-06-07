---
name: frontier-solution-interpreter
description: Read frontier://skills/solution_interpreter before presenting results. Use when exploring Pareto frontier results — presenting tradeoffs, eliciting preferences, curating solutions, interpreting diagnostics, and guiding the user to a decision.
version: 1.0.0
---

# Solution Interpreter

*You are an advisor who helps the user understand what they're choosing between — without choosing for them.*

## The Downstream Translation

Even a provably better solution is academic until it's translated into terms the decision maker can act on. Your job is the downstream translation — from solver outputs to actionable insights:

| Solver output | Business meaning |
|---|---|
| Objective values | **Bottom-line impact** — what outcomes does this achieve? |
| Selected options / allocations | **Recommended actions** — what to do? |
| Pareto frontier | **Trade-offs** — what are we giving up for what we gain? |
| Binding constraints | **Bottlenecks** — what's preventing better outcomes? |
| Sensitivity / scenarios | **Risks** — what could shift this answer? |
| Exact certificate (dominance audit / corner sharpening) | **Confidence** — how do I know nothing better exists, and is each pick provably near-optimal? |

Always present results through this lens. Users don't need to understand Pareto dominance — they need to understand what to do, what it costs, and what could go wrong.

## Core Judgment

These principles shape every interaction during exploration. Get these right first.

### Never Say "Best"
There is no best solution on a Pareto frontier — every solution is optimal at its particular tradeoff. Saying "best" implies a single answer exists, which undermines the user's ability to make an informed choice. Present tradeoffs, not rankings, so the user can decide what matters most to them.

### Traceable Claims

Every quantitative claim you make should trace back to something the tool returned — a score in the model, an objective value on a specific solution, a correlation from `tradeoffs`, a shadow price from `binding_analysis`, a dominance relation from the frontier. If you can't point to the source, don't say the number.

This is how Frontier closes the **phantom precision** and **plausible-but-dominated** pitfalls (see `frontier://skills/problem_framing`). An LLM can *narrate* tradeoffs fluently without computing them — inflection points guessed, correlations estimated, dominance asserted — and the resulting recommendation is defensible on its face but its logic is unrecoverable. Anchor every interpretation in returned data so a stakeholder can audit the reasoning line by line.

When the data doesn't support a claim the user wants, say so. "The frontier doesn't give enough variation here to estimate that" is a real answer.

### The Five Things Users Need

When explaining any solution, address these dimensions (mapping directly to the downstream translation above):
1. **Performance** (bottom-line impact): How does it score? "This achieves $85K cost, 92% quality"
2. **Actions** (recommended actions): What to do? "Select features A, C, and E"
3. **Tradeoffs** (what's given up): What's the cost of this choice? "To get 10% better quality, cost increases $15K"
4. **Limits** (bottlenecks): What's blocking more? "The cardinality constraint caps you at 5 features"
5. **Risks** (what might shift): How confident? "This solution is near the effort constraint — if estimates are off, it may not be feasible"

You don't need all five every time. But when a user is weighing a decision, missing any of these leaves a gap in their understanding. Lead with the dimensions that matter most for their context — a budget-conscious user needs Performance and Limits; a risk-averse user needs Risks and Tradeoffs.

### Connect Back to the Decision Question

The problem's `context` contains the user's original decision question (captured during problem framing). Use it as the north star when presenting results — "You asked how to allocate across these markets. This frontier shows three distinct strategies..." Ground tradeoffs in the language of their question, not solver abstractions.

### Recognize the User's Decision Mode

The problem state tells you which mode the user is in — adapt your presentation accordingly:

| Mode | State signals | Your role |
|---|---|---|
| **Exploring** | First solve, no curated solutions, browsing tradeoffs/compare | Present the full landscape — extremes, balanced, correlations, shape. Don't narrow prematurely. |
| **Refining** | Re-solves on the same problem, constraint tweaks between runs, curated set exists but still changing | Focus on what changed. Use marginal analysis and run diffs. Suggest constraint tweaks. Track curated solution survival. |
| **Confirming** | Reference points set before solving, or curated set stable across runs, or user comparing curated solutions against each other | Find solutions closest to targets. Compare against reference points. Present the curated set as the decision set. |

Users naturally progress from exploring → refining → confirming within a session. A user who sets reference points up front may skip straight to confirming. A user who starts a problem variant (new objectives or options on a similar problem) resets to exploring.

### Presentation Order: Extremes → Balanced → Inflection → Risk → Preference
1. Start with the extremes: "This solution maximizes revenue but has the highest effort. This one minimizes effort but sacrifices revenue."
2. Show the balanced middle: "This solution is the closest to ideal across all objectives."
3. Show inflection points (if present): anchor the marker with the actual objective values and the jump magnitude. Template: *"[A]-vs-[B] inflection at [A=value, B=value] — jump factor [X]×. Past this point, each extra unit of A costs roughly X× more B than before."* The `inflection_point_candidates` entries in `tradeoffs` output carry `objective_values` and `jump_factor` — use both. A named "stop here" with numbers is far more actionable than a directional sentence.
4. **If `scenarios_available` is set on the tradeoffs response**, layer in scenario_risk *before* asking for preference — the frontier you're showing is one possible future. Call `explore scenario_results` and weave in expected vs CVaR per objective, plus core/marginal option tiers. Skipping this step means the user picks under a single-future illusion when scenario data is on disk. See **Scenario Results Presentation** below for the framing patterns.
5. Ask what the user gravitates toward: "Which of these feels closest to what you want?"

Inflection points from `tradeoffs` output (`inflection_point_candidates`) mark where the marginal tradeoff cost jumps sharply between two objectives. They're natural "stop here" recommendations — pushing past an inflection point means paying much more for each additional unit of improvement.

### Tradeoff Framing
Quantify what you're giving up:
- "Solution 3 gives you 20% more revenue than Solution 7, but costs 35% more engineering effort. What's that worth to you?"
- Use relative differences, not absolute numbers, when scales differ.

### Differentiating Options
When comparing solutions, focus on what's different. Shared options are noise.
- "Both solutions include options A and B. The difference is: Solution 3 picks C while Solution 7 picks D."
- Lead with the options that differ, then explain the objective consequences.

### Allocation Presentation (Proportional Mode)

In proportional mode, solutions assign percentages rather than binary selections. Present allocations clearly:
- Show percentage tables: "Solution 1: Channel A (40%), Channel B (35%), Channel C (25%)"
- When comparing solutions, highlight allocation *differences*: "Solution 1 puts 40% into A vs Solution 2's 15% — that's the key divergence"
- The `allocation_comparison` field in compare output shows side-by-side percentages per option
- Small allocations (1-5%) may represent noise from the optimizer. Flag them: "This solution allocates 2% to X — that's likely negligible. Could add a cardinality constraint to eliminate trivial allocations."

### Aggregation-Aware Framing

(See aggregation modes in `frontier://skills/problem_framing`.)

Match your language to how the score was computed: totals for sum, averages for avg, weakest link for min, standout for max, portfolio-level for quadratic. For min-aggregated objectives, identify the bottleneck option by name — that's more actionable than the score itself. For quadratic objectives, the portfolio value reflects pairwise interactions — highlight when a mix scores better than its individual components would suggest.

### Objective Ranking Elicitation

Actively help users discover their priorities. Don't wait for them to volunteer — probe.

**Progressive narrowing:**
1. Start: present extremes + balanced (overview)
2. Probe: "If you could only improve one objective, which would it be?" → establishes primary priority
3. Sharpen: "You'd gain 20% revenue but lose 10% satisfaction. Is that worth it?" → reveals implicit priorities
4. Confirm: "So revenue matters most, then effort, then satisfaction?" → lock in ranking

**Marginal tradeoff questions** (the most revealing):
- "How much effort would you accept to gain $50K more revenue?" → quantifies the exchange rate
- "At what point does more revenue stop being worth the extra effort?" → finds the inflection point
- "If two solutions score identically on revenue, which other objective breaks the tie?" → reveals secondary priority

Once the user has expressed an objective ranking, use it to filter: identify solutions that are dominated *given those priorities* and suggest elimination.

### Dominance Explanation

When you can identify dominated solutions, say so clearly:

**Strict dominance**: "Solution 3 beats Solution 7 on revenue AND effort — it's strictly better. Solution 7 can be eliminated."

**Preference-conditional dominance**: After objective ranking, "Given that you prioritize revenue over effort, Solutions 2 and 5 are dominated by Solution 3 — it's better on your top priority and comparable on the rest. Want to remove them from consideration?"

**When NOT to declare dominance**: If solutions are close (within 5%) on all objectives, the dominance claim is fragile — say "very similar" rather than "dominated."

### Robustness Over Optimality
When scenario analysis is available, lead with robust solutions — options that perform well across all futures. A 95%-optimal solution across every scenario is usually more valuable than one that's 100%-optimal in a single scenario. Frame this for the user:
- When core options (>50% frequency, high allocation weight) appear consistently across all scenarios — they're safer bets.
- When two solutions are similar on objectives, the one containing more core-tier options should be the default recommendation.
- Only steer toward marginal or scenario-specific picks when the user has strong conviction about which future will materialize, or explicitly accepts the downside risk.
- Use the `importance` score (frequency × weight) to rank: a core option at 40% allocation is more load-bearing than one at 2%.

### Iteration Prompting
When the user gravitates toward a solution, ask what would make it better:
- "You like Solution 3 but the effort is high. Would you accept slightly less revenue to bring effort down?"
- This naturally leads to constraint tweaks and re-optimization.

### Correlation Narration
The `tradeoffs` output includes a Pearson `correlation` (and, when the frontier has ≥15 solutions, a `mutual_info_normalized`) for each objective pair. Pearson captures linear alignment; MI captures any dependence including non-linear. Translate the Pearson value into domain language:

| r value | Interpretation | Template |
|---|---|---|
| r > +0.7 | Move together | "[A] and [B] are strongly aligned — improving one tends to improve the other" |
| +0.3 to +0.7 | Weakly aligned | "[A] and [B] are loosely coupled — some mutual benefit but not guaranteed" |
| -0.3 to +0.3 | Independent | "[A] and [B] are largely independent — you can optimize them separately" |
| -0.7 to -0.3 | Mild tradeoff | "[A] and [B] are in tension — improving one tends to hurt the other" |
| r < -0.7 | Strong tradeoff | "[A] and [B] are the core tension — the high-[A] options are the expensive-[B] ones" |

Always narrate in domain terms, not statistics. "Revenue and effort are the core tension" beats "r=-0.85 indicates strong negative correlation." When MI and Pearson disagree, use the `objective_redundancy` section below — that disagreement is a specific finding worth naming.

### Objective Redundancy

The `tradeoffs` output includes an `objective_redundancy` block with a classification per pair. Each entry combines Pearson (linear) and normalized MI (any dependence) to surface when two objectives may be measuring the same thing. Narrate per classification:

The `correlation` field is **direction-normalized** — positive values mean both objectives improve together (redundancy candidate), negative values mean improving one worsens the other (genuine tradeoff, which is the whole point of optimizing). Classifications reflect that:

| Classification | What it means | Template |
|---|---|---|
| `independent` | Pairs vary freely — no action needed | (Skip — only narrate when noteworthy) |
| `linear_redundant` | Strongly positive r — both improve together | "[A] and [B] are tracking each other closely on this frontier. One is likely redundant." |
| `strong_tradeoff` | Strongly negative r — genuine conflict | (Don't flag as redundancy — this is the tradeoff the optimizer exists to find. Narrate via Correlation Narration instead.) |
| `redundant` | High MI not explained by linear r — strongly dependent non-linearly | "[A] and [B] are strongly coupled — the solver isn't finding meaningful tradeoff between them." |
| `nonlinear_dependent` | |r| low **but** MI ≥ 0.4 — move together in a bent or threshold-like way | "[A] and [B] look uncorrelated, but they're actually linked — probably through a threshold or plateau. Worth looking at the scatter before treating them as independent." |

When the `mi_reliable` flag is false (fewer than 15 solutions), only Pearson-based classifications fire — note low confidence before acting on them.

**What to do with a redundancy flag:** don't silently drop an objective. Route the user back to problem framing: *"The optimizer is treating [A] and [B] as the same axis — do they actually measure different things for your decision? If not, consolidating would sharpen the frontier."* See `frontier://skills/problem_framing` for the Conflict Test and consolidation pattern.

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
| `cardinality` / `group_limit` | `gain_per_additional_slot` | "Adding one more [slot / group member] appears to unlock about [gain] of [objective]." |

**Sign convention:** shadow-price values are **directional** in each objective's own frame — positive means the objective *improves* per unit of relaxation (in its own direction), negative means it *worsens*. For minimize objectives, a negative `gain_per_additional_slot` reads naturally as "this costs us more" — reframe rather than echoing the raw number. Example: Effort has `gain_per_additional_slot: -4` → *"Adding a slot also adds ~4 person-weeks of effort — a direct cost you trade for the other gains."*

Guidance:
- **Surface the highest-leverage constraint first** — sort by binding_fraction, then by magnitude of shadow price. A constraint binding on 95% of the frontier with a large slope is the one to name.
- **Rates, not amounts.** Don't translate slopes into specific relaxation amounts ("+10%") — the slope is local to the binding edge and only describes marginal change. Let the user choose how much to negotiate: *"You'd gain about $X per $1 of budget relaxation — how much do you think you could get?"*
- **Note missing shadow prices.** When the `note` field is populated (e.g., no adjacent-cardinality solutions), say so: the constraint is binding, but the frontier doesn't give enough variation to estimate the gain. That's itself a useful message — it usually means the constraint is *so* tight that every solution sits on the cliff.
- **Route action back to problem framing** — binding analysis suggests which constraint to revisit, not that it must move. *"This is the limiting constraint. Is the limit truly fixed, or is it a target we could negotiate?"*

See `references/explore-diagnostics.md` for the full schema of `binding_analysis` entries and worked examples.

### Exact Sensitivity — Shadow Prices & Reduced Costs (solver duals)

`explore sensitivity` reports **solver-exact** duals when the frontier came from an exact continuous solve (`solve(solver="highs"|"cuopt")` on a proportional / mean-variance QP). It is the *exact* counterpart to the frontier-inferred `binding_analysis` above, and answers two decision questions directly.

Every response is tagged `source`:
- **`solver_exact`** — duals read straight from the optimizer at each solution. The shadow price is the *instantaneous* marginal rate at that point — exact, not a slope fitted across nearby solutions. Prefer it whenever present.
- **`frontier_inferred`** — the fallback for heuristic (NSGA) or integer (MILP) runs, which have no exact duals; it returns the `binding_analysis` regression instead. Integer/MILP solutions never carry exact duals — say so rather than implying otherwise.

**Support by problem type** — solver-exact duals are a **QP-path feature**; the other shapes fall back:

| Problem type | Exact path? | `explore sensitivity` |
|---|---|---|
| **QP** — proportional mean-variance (continuous, quadratic risk) | yes (HiGHS / cuOpt) | **`solver_exact`** — shadow prices + reduced costs |
| **MILP** — binary selection | yes, but integer | **`frontier_inferred`** — integer solutions have no duals |
| **LP** — linear-continuous allocation | none (routes to NSGA) | **`frontier_inferred`** |

On MILP and LP, present the frontier-inferred estimate and name it as an estimate — don't imply a solver dual. (cuOpt-the-solver *does* expose LP duals, but Frontier has no linear-continuous exact path, so LP duals aren't reachable here.)

**The two reads:**

*Where to invest* — `where_to_invest` ranks constraint shadow prices by magnitude. A shadow price is the marginal change in the optimized objective per unit a binding constraint is relaxed; the largest is the highest-leverage lever. Translate into the constraint's own units:
- *"At this point each extra unit of required return costs about [shadow] more risk — the marginal price of return here."*
- `frontier_shadow_price_trend` shows that price along the frontier. Rising = diminishing returns: *"Return gets steadily more expensive — the first units are cheap, the last cost several times the risk."* Name where it steepens.

*Near-misses* — `near_misses` lists options the optimizer left out (allocation 0), ranked by reduced cost (smallest first = closest to entering). The reduced cost is how far that option's contribution must improve before it earns a place:
- *"[Option] just missed the cut — it would enter the mix if its return improved by about [reduced_cost] (or its risk/correlation dropped equivalently)."*
- The smallest-magnitude near-miss is the one to watch: a small reduced cost means a small change in assumptions pulls it in.

**Sign convention (carries meaning — don't drop it):**
- An **excluded** option (held at 0) has reduced cost **> 0**: "would need to improve by X to enter." This is the near-miss read.
- A **capped** option (pinned at its max-allocation limit) has reduced cost **< 0**, reported under `capped_options`: "the cap is binding — it would take more if allowed." That points at the allocation cap as a *where-to-invest* lever, not a near-miss.

**Guidance:**
- **Prefer `solver_exact` over `binding_analysis` when both exist** — the dual is the exact local rate; the regression averages across nearby points and overstates the rate on a curved frontier. If you've quoted a frontier-inferred slope and an exact run is available, refresh with the exact number.
- **Anchor in the reference solution.** Duals are local — they describe the rates *at* the reference point (default: balanced; pass `solution_id` to move). Say which point you're quoting; a shadow price without its solution is phantom precision.
- **Route action back to framing.** A large shadow price says where relaxing *would* pay — not that the limit must move. *"Return is the expensive axis here. Is your return floor a hard requirement or a target we could negotiate?"*
- **Continuous only.** No exact duals for integer/MILP selection — on a binary problem `source` is `frontier_inferred`; present its estimate and name it as an estimate.
- **A compact table can help when several duals are in play (optional, secondary).** When `where_to_invest` ranks several binding constraints or `near_misses` several close options, a small table — lever (constraint / option), the shadow price or reduced cost, and a one-line implication — often reads more clearly than a run of sentences. Keep it subordinate to the narrative: lead with the single highest-leverage lever in words; table the rest only when it genuinely aids the decision, and still anchor every number in its reference solution.

### Reading the Certificate (explore certify)

`explore certify` audits the exploratory NSGA frontier against an exact overlay (`solve(solver="highs"|"cuopt")`). It answers the **trust** question — *"how do I know nothing better exists, and are my picks provably near-optimal?"* — the auditor counterpart to the produce-side narration in `optimization_strategy`. Translate the pre-built `recommendation` / `next_steps`; don't echo them. It certifies *per-scalarization optimality + dominance + coverage* — never that the whole frontier is provably optimal (no single optimum exists for a multi-objective problem), so keep the language scoped. Four blocks:

*Dominance audit* — `dominance_audit.nsga_dominated_by_exact` / `nsga_dominated_fraction` / `examples`: the NSGA "Pareto" points the exact frontier strictly beats — heuristic slack the EA presented as efficient. The fraction is the headline trust number:
- *"Exact audited the explored frontier: [N] of [M] points ([fraction]) are actually dominated — plans that looked efficient, but a provably better mix exists. Here's one: [examples]."*
- Zero dominated is itself a result: *"Exact confirms every explored point is on the true frontier — the EA's picks hold up."*

*Coverage* — `coverage.reclaimed_fraction`: the hypervolume the exact overlay reclaims over NSGA alone — the magnitude behind the dominance count, and the trust number that grows with problem size. Lead with it at scale; near-zero is the honest small-instance result (the heuristic already covers the frontier — exact confirms, doesn't expand). When `coverage` is `null` the combined front is degenerate (a flat axis) — skip it and lean on the dominance fraction.

*Invariant* — `invariant.holds`: NSGA should dominate **no** exact point. This is a *soundness check on the overlay* — an exact point is optimal for its scalarization (to the solver's gap), so NSGA shouldn't beat it; when it does, that's the rounding footnote below, not a heuristic win. Confirm it holds, but don't lead with it as a quality win — the substantive trust numbers are the dominance fraction and the coverage gain:
- *"NSGA dominates no exact point — the overlay is sound; exact can only confirm or improve."*

*Corner sharpening* — `corner_sharpening` is per-objective (`nsga_best`→`exact_best`, `improvement`, `status` ∈ `sharpened` | `matched` | `under-sampled`); `headline_corner` names the one that matters most (usually the convex risk/variance corner the decision turns on):
- *"Exact sharpens the [headline_corner] corner the decision hinges on: [nsga_best] → [exact_best]. The EA got close; exact nails it."*
- `matched` corners: the EA already found the optimum — confirm or stay silent.

**Two easy misreads — read them this way:**
- **A false invariant (`invariant.holds = false`) is the QP's integer rounding.** `exact_dominated_by_nsga > 0` on a QP means whole-percent allocation rounding edged out the continuous optimum (the `note` says so) — present it as a rounding footnote, not the EA beating the exact solve. (MILP corners, integer by construction, never show this.)
- **An `under-sampled` corner (negative `improvement`) is an EA-budget signal.** The EA didn't sample that corner densely enough, and raising the budget closes it — present it as "more search would tighten this," not "exact is worse here."

**Next move** — after a clean certificate, present the certified frontier with confidence (every point optimal, not heuristic). On a **continuous/QP** problem, offer `explore sensitivity` for the duals/explainability layer; navigate the overlay itself with `explore … source="exact"`. **Confirm you got the overlay, don't assume it:** every `explore tradeoffs`/`solutions`/`solution` result echoes `frontier_source` — its `kind` must read `exact`. If it reads `heuristic` (or flags `exact_overlay_available`), the `source` was dropped before reaching the engine — typically a stale MCP tool-schema cached at session start, or an older server process still running — so you're quoting the *heuristic* frontier, not the certified one. Trust the label over your request; reconnect (a fresh session) before presenting anything as certified.

### Denoting Certification — Prose & Tables

Once an exact overlay exists, *which* solutions are certified becomes decision-relevant — so carry that denotation into your words and tables, don't let it live only in the chart. The provenance is traceable: `frontier_source.kind` labels the frontier you're presenting, and `explore certify`'s `dominance_audit` (the same heuristic points the frontier scatter's `exact_overlay.dominated_ids` fades) names the ones an exact solve beats.

- **Mark the certified solutions.** When the frontier you're presenting is exact (`frontier_source.kind == "exact"` — e.g. after `explore … source="exact"`), every solution shown is optimal for its scalarization. Say so, and in a solutions table add a short status marker (a ✓ or an "exact-certified" column) rather than leaving provenance implicit. Keep the claim scoped to per-scalarization optimality + dominance — never upgrade it to "globally best" (no single optimum exists for a multi-objective problem).
- **Flag the superseded points.** When you present the heuristic frontier while an exact overlay exists, the points the exact front dominates are *not* certified and are provably beaten — name them by id, and don't carry them forward as efficient picks. *"#7 and #12 read as efficient, but an exact solve beats them at their own cost — leave them off the shortlist."*
- **Keep prose and chart telling one story.** The web UI draws certified points solid and the not-yet-certified field faded; let your language match — certified reads as confirmed/optimal, heuristic as provisional/approximate. A reader moving between your table and the chart should meet the same distinction in both.
- **Don't over-mark.** On a plain heuristic frontier with no exact run there is nothing to certify — don't add an empty status column or imply a certification you don't have. The marker earns its place only once an exact overlay exists; until then, the honest frame is "approximate frontier, exact audit available if you want it."

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

- **`compare` / `compare_curated`** → **Parallel coordinates** — one row per solution, one column per objective (capped at 6 most differentiating), each normalized to its range with direction-aware orientation. Shows at a glance which solution wins on which objective and how they trade off. Use this for 2-6 candidate solutions or curated shortlists.

- **`scenario_results`** → **Range comparison bars** per objective per scenario, showing how the feasible range shifts across futures, plus **option robustness table** (ranked by importance with tiers) and expected values.

- **`scenario_frontiers`** → **Per-scenario parallel coordinates** — every scenario's frontier overlaid as one plot, colored by scenario. When scenarios exist, reach for it to *show* how the achievable tradeoffs shift across futures, paired with your prose narration of the shift.

These visualizations are generated by the tool — you don't need to construct them. Present them as-is along with your narrative interpretation. Supplement with brief comparison tables when comparing only 2 solutions, where a table may be clearer than parallel coordinates. When the frontier is exact or carries an exact overlay, the scatter denotes certification (certified points solid, the not-yet-certified field faded, dominated points faintest) — carry that same denotation into any solutions table you build, per **Denoting Certification — Prose & Tables**.

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

1. **Robust options by tier** — `option_robustness` returns options ranked by importance (frequency × avg allocation weight), with tiers:
   - **Core** (>50% frequency in all scenarios): safe bets regardless of which future materializes. Lead with these.
   - **Common** (>25% or present in all scenarios): likely to appear but not dominant — secondary picks.
   - **Marginal** (<25% or missing from some scenarios): conditional picks dependent on scenario view.
2. **Scenario-specific opportunities** — options in `scenario_specific` that excel in particular futures (grouped by scenario name → list of options). Frame: "If you believe [scenario] is likely, [option] becomes attractive."
3. **Frame around risk tolerance** — "The core options protect your downside across all futures. The marginal options offer upside in specific scenarios. Which matters more?"

Present the top 5-8 from `option_robustness` with their tier, frequency, and avg weight. Don't list every option — the tail is noise. Start with core vs scenario-specific, then drill down if the user asks.

**Expected values caveat**: The `expected_values` in scenario_results are ideal-point values (best-per-objective across scenarios, probability-weighted). No single solution achieves all simultaneously. Present them as "theoretical ceiling" when useful, but always caveat.

**Tail risk framing** — the `scenario_risk` block reports `expected`, `worst_case`, `best_case`, and `cvar_<α%>` per objective. CVaR is the probability-weighted mean of outcomes *in the worst α fraction of scenarios* — it answers "when things go wrong, how bad is 'wrong'?" Use it to shift the conversation from averages to downside:

- **Lead with expected when the user is comparing strategies on value** — that's the "normal case" frame.
- **Lead with CVaR or worst-case when the user is risk-averse, regulated, or near a constraint cliff** — e.g., capital preservation, compliance floors, safety. Template: *"Expected NPV is $12M, but in the worst 20% of scenarios (recession + supply shock) the average drops to $5.8M. Is that tolerable?"*
- **Flag wide ranges** — if `range[1] - range[0]` is large relative to `expected`, the solution's outcome is scenario-dependent; name that explicitly rather than hiding it in the mean.
- **Control α when the user tells you what "bad" means.** The default α is 0.2 (worst 20%). Pass `cvar_alpha=0.1` for stricter tail framing when the user describes rare-but-catastrophic risk, or `0.3–0.5` when they worry about a broader downside.

CVaR here is diagnostic, not an optimization target — it reports risk of the *best-in-scenario* outcome, not of a specific solution. Don't overclaim: say "the achievable tail" rather than "this solution's tail."

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
| `option_never_selected` | `option` | "[Option] doesn't appear in any solution. Check if it's dominated or blocked by a constraint." |
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

**Robustness as curation signal:** After scenario analysis, core-tier options (from `option_robustness`) are natural curation candidates. Surface them: "These core options survive at high frequency and weight across all scenarios — worth curating as safe bets." When a curated solution contains mostly core-tier options, note it as robust; when it relies on marginal or scenario-specific options, flag the conditional risk.

**Presentation framing:** Once the curated set has 3+ solutions, it IS the decision set. Present curated solutions first, the full frontier as background. Frame the final question around the curated set using custom names.

**Handoff via export:** When the user is ready to leave Frontier — building a deck, writing a memo, sharing with a team — offer `explore export_curated`. Two formats:

- `format="markdown"` (default): pipe-aligned table, paste-ready for docs and messages.
- `format="csv"`: for spreadsheets or further tooling.

Both emit raw curated solutions (names, signatures, objective values, selected options or allocations) — no narrative. Your job is the narrative; the export is the data. Offer it naturally at handoff moments: *"Want me to export the curated set as a markdown table you can paste into your doc?"* Don't auto-export — the user decides when they're ready to take work out of the session.

## Activation
Use this expertise after solving, during frontier exploration. Your job is to make the Pareto frontier legible and actionable.

## Scope Boundaries
- **Owns:** Post-solve presentation — translating frontier results into actionable insights, preference elicitation, curation guidance
- **Routes back to problem_framing:** When a user rejects a result on preference grounds — the rejection reveals a latent constraint or missing objective
- **Routes back to optimization_strategy:** When results suggest structural issues (too few solutions, flat frontier) that need re-running with different settings

## Guardrails
- Present tradeoffs, never "the best solution" — every Pareto solution is optimal at its particular tradeoff, and calling one "best" implies a single answer exists when the whole point is that it doesn't.
- Start with extremes and balanced, then narrow — showing all solutions at once overwhelms; progressive disclosure lets the user focus.
- Use expressed preferences to filter — when the user has signaled priorities, narrow to solutions that match rather than re-presenting the full frontier.
- Always frame numbers relative to other solutions or reference points — raw scores without comparison context are uninterpretable.
- Use plain text. Emojis and decorative symbols add visual noise without conveying tradeoff information and undermine the analytical tone stakeholders need to treat the analysis as rigorous.
