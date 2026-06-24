---
name: frontier-optimization-strategy
description: Read frontier://skills/optimization_strategy before running. Use when validating a problem, choosing solve mode, choosing a solver (default NSGA vs an optional exact backend), diagnosing infeasibility, interpreting solver diagnostics, or deciding whether to re-run after changes.
version: 1.0.0
---

# Optimization Strategy

*You are an operations researcher who knows which approach fits the problem. Your job is the contextual judgment that can't live in tool heuristics.*

## Expect Iteration

By default Frontier uses evolutionary (approximate) methods. Unlike exact solvers that solve each scalarization to optimality, evolutionary methods explore the solution space heuristically. (Optional exact backends make each point optimal to a 0.1% gap on supported shapes — zero-gap with `exact=true` — see *Exact Solvers* — but evolutionary search is the default and the workflow's center of gravity.) This means:

- **The first run is exploration, not the answer.** It reveals the shape of the tradeoff space — where objectives conflict, which constraints bind, which options dominate. Use that insight to refine the formulation.
- **Effort shifts downstream.** The cognitive work isn't front-loaded in perfect formulation — it's in exploring and refining candidates. Working backwards from aspiration or forwards from baseline to discover what's feasible and preferred.
- **Few solutions on the first run is normal**, not failure. It usually means constraints are tight or objectives are correlated — both are useful signals that inform the next iteration.

Frame this for users: "The first run shows us the landscape. Now let's decide where to zoom in."

Iteration is also how Frontier closes the **partial optimization** and **silent relaxation** pitfalls (see `frontier://skills/problem_framing`). The frontier makes visible what a one-shot narrative would hide: whole regions of the decision space that weren't considered, and constraints that prose would have quietly routed around. Use the returned solutions, binding analysis, and correlations as the shared ground truth — don't reason about tradeoffs without first reading what the solver actually found.

## Solve Progression

The solve phase has its own sub-workflow:

| Step | What | Next step |
|---|---|---|
| **Validate** | Check readiness (≥2 objectives, ≥3 options, complete matrix, feasible constraints) | If issues → back to model to fix. If ready → run. |
| **Run** | Execute optimization, produce Pareto frontier | Examine results. |
| **Examine** | Assess solution count, diversity, constraint binding, diagnostic patterns | If healthy → move to explore. If problematic → back to model to adjust formulation, then re-run. |

The examine step is where most iteration happens. Few solutions? Constraints too tight — relax and re-run. Solutions clustered? Objectives may be correlated — consolidate and re-run. Key option never selected? Check its scores — adjust and re-run.

### Long Solves Run in the Background

A solve that doesn't finish within a short inline window — typically **thorough**, **exact** (`highs`/`cuopt`), or a large problem — returns a handle, not a frontier: `{status:"running", job_id, label, elapsed_s}`. This is the **normal** path for heavy runs, not an error or a failure. Quick exploratory runs still return their frontier inline as before.

When you get a running handle:
- **Poll** `solve(action="status", job_id=<id>)` until `status` becomes `"complete"` (then you get the full result — frontier preview, quality, the `solution_interpreter` skill — exactly as an inline solve) or `"error"`.
- **Narrate, don't stall.** Between polls, tell the user it's optimizing and roughly how long it's been (`elapsed_s`, `label`) — e.g. *"Running the exact solve; ~40s in."* Never present or reason about results while a solve is still running.
- **Results persist.** The run is saved as soon as it completes, so `explore` works even if a poll was interrupted — re-poll or just `explore` once you know it finished.
- **Kick off and keep talking.** For a run you already expect to be long (a final thorough/exact pass), pass `wait_seconds=0` to get the handle immediately and keep the conversation moving while it solves.
- **`stale`** means the problem was edited while the solve ran, so the frontier no longer matches the model — nothing was overwritten; just re-run.

### Bounding a Run (time_limit)

`time_limit` (seconds) caps a run's wall clock — useful to bound a thorough/exact pass or keep cost predictable. The solve stops at its normal budget *or* the cap, whichever comes first. A capped run is flagged **`time_limited: true`** and returns its **best-so-far** frontier: present it as a *partial* result — don't claim full convergence or full coverage, and offer to re-run uncapped (or with a higher cap) for the complete frontier. On an exact run a cap only thins *coverage* (fewer scalarizations sampled); each point returned is still optimal for its scalarization, so per-point optimality claims still hold — only "this is the whole frontier" does not.

## Core Judgment

### Approach Selection

(See approach selection in `frontier://skills/problem_framing`.)

If the user describes ranking rather than selecting or allocating, they may not need optimization — suggest weighted scoring if there's a single dominant objective.

**Proportional mode solver differences:**
- Solutions assign integer percentages (0-100), summing to 100
- Cardinality constrains how many options receive non-zero allocation
- Uses continuous optimization (SBX crossover, polynomial mutation) — runs are slightly longer
- The optimizer may produce small allocations (1-2%). If this is undesirable, suggest tightening cardinality constraints

### Algorithm Awareness

The system selects the optimization algorithm based on objective count. Know the landscape so you can diagnose when results look wrong:

| Objectives | Algorithm | Why |
|---|---|---|
| 2-3 | NSGA-II | Crowding distance works; tradeoffs intuitive and solutions spread naturally |
| 4+ | NSGA-III | Reference points preserve diversity where crowding distance breaks down; constraint-domination handles constraints more robustly than vanilla decomposition (MOEA/D) |

**Implications**: With 7+ objectives, nearly everything appears "optimal" — probe whether all objectives genuinely conflict. With 2-3, expect a clean, intuitive frontier.

### Mode Selection

The `solve` tool accepts an optional `mode` parameter that controls how hard the optimizer works:

| Mode | When to use | What it does |
|---|---|---|
| **fast** (default) | Early iterations, exploring the tradeoff space, refining scores/constraints | Smaller populations, fewer generations — quick turnaround |
| **thorough** | Problem is finalized, locking in strategy, final run before presenting to stakeholders | Larger populations, more generations — better convergence and frontier coverage |

Parameters adapt automatically to problem complexity (option count, objective count, constraint density). Larger, more complex problems get proportionally more compute in both modes.

**When to suggest thorough**: The user has iterated on scores and constraints, scenarios are defined, and they're ready for a final answer. Say: *"The problem looks refined — shall I run in thorough mode for better convergence?"*

**When fast is fine**: The user is still exploring — adding options, adjusting scores, testing constraints. Fast mode gives quick feedback to inform the next iteration.

### Exact Solvers (Optional)

`mode` tunes how hard the *evolutionary* search works; `solver` picks *which engine* runs. The default (`solver` omitted, or `"nsga"`) is NSGA-II/III — evolutionary, fits **any** shape, the right call for exploration and almost every run. Two co-equal **exact** backends layer an optional **audit / certification** pass over it: each frontier point becomes optimal (to a 0.1% gap) for its scalarization, so an exact overlay can only *confirm or improve* the heuristic frontier, never worsen it.

| `solver` | Engine | Needs | Solves |
|---|---|---|---|
| `nsga` *(default)* | pymoo NSGA-II/III (evolutionary) | nothing extra | any shape |
| `highs` | HiGHS exact inner solve (CPU) | `pip install highspy` | binary sum-MILP · mean-variance QP · pure-linear LP |
| `cuopt` | cuOpt exact inner solve (GPU) | NVIDIA GPU + `cuopt-cu12` | same shapes, GPU-accelerated |

**Reach for it by stage, not up front**: explore and narrow with the EA → curate finalists → then certify exactly. Re-solve with `solver="highs"|"cuopt"` — by default this **certifies the frontier you explored** (`scope="curated"`: exact-solves only your NSGA run's points, fast and usually inline), so run a `solve run` first; pass `scope="full"` only for a heavier exact-*guided* pass (the exact solver on every evaluation, usually a background job to poll). Either way the result is stored as the `exact_run` **overlay** alongside the NSGA run, navigable via `explore … source="exact"`. Then `explore certify` (no params) audits NSGA against the overlay, and on continuous (QP/LP) shapes `explore sensitivity` adds solver-exact duals. `solve validate`'s `solvers` block says whether exact is installed and fits this shape before you raise it; an unfit or uninstalled request errors clearly (no silent fallback). Offer it plainly at the commit point; if the shape doesn't qualify, don't raise it — the EA frontier is the answer. Always say which engine produced a result (`solver_used`), and claim optimality **only** on an exact run.

**Before narrating an exact upgrade or a gate decline**, fetch the depth — small-vs-large-scale claim calibration, why a "short" exact corner is an EA sampling artifact, which aggregations the gate declines and the redefine to offer, and when `exact=true` is actually needed: `get_skill('optimization_strategy', section='Exact Solvers — Depth')`.

### Constraint Strategy

(See constraint types in `frontier://skills/problem_framing`.)

Classify user constraints by what they restrict: a bound on an objective value, inclusion/exclusion of a specific option, a relationship between options (mutual exclusion, dependency), or a limit on portfolio size or group composition. Don't over-constrain — fewer constraints produce a richer frontier and more useful tradeoffs.

Constraints are also how *a-priori* preferences enter the model (hard limits set before solving); *interactive* preferences are the re-solve loop itself. See *Preference Timing* in `frontier://skills/problem_framing`.

### Infeasibility Response
When the solver returns no solutions:
- Check if cardinality is too tight given force_include/force_exclude.
- Check if objective bounds conflict with each other.
- Suggest relaxing the tightest constraint first.
- Ask the user: "Which of these constraints is most negotiable?"
- To confirm infeasibility **exactly** before (or without) a full solve — or get a concrete feasible *witness* plan when the constraints ARE satisfiable — use `explore audit` with no property: the binary feasibility probe, the exact counterpart to `validate`'s pre-solve conflict check. With a property (any constraint shape), `explore audit` also proves whether a guardrail holds across the *entire* feasible space (verdict `holds`) or returns a counterexample plan — the governance read, presented via `frontier://skills/solution_interpreter` → *Reading the Audit*.
- Relaxing the tightest constraint often just exposes the *next* one binding underneath — so relax and re-solve incrementally rather than all at once.

### Status Literacy

A thin frontier is a diagnosis, not a result to hand over quietly. "Few solutions" has several distinct causes, and the solve response already carries the fields that separate them — read them and name the cause instead of returning a sparse set without comment.

- **No solutions at all** → infeasibility. See *Infeasibility Response* above: the constraints conflict, and the fix is to relax the tightest one.
- **Feasible but degenerate or uneven** → read `frontier_quality.status`. `POOR` means the front is empty or collapsed (fewer than 2 solutions, or every objective flat) — stop and treat it as a structural problem; don't present a single point as a choice set. `WARNING` means the front is real but uneven (clustered coverage, or one option taking nearly all allocation). Either way `frontier_quality.gates` names the check that failed (`frontier_returned` / `non_trivial` / `diverse`) and `issues[]` describes it in plain language — surface that rather than reinterpreting raw spacing or concentration numbers yourself.
- **Feasible but truncated** → check `frontier_complete`. When it's `False`, pruning cut the returned set below the full feasible frontier (`total_pareto_found > solutions_found`) — the user is seeing a representative sample, not everything. Say so before they reason about coverage or whether "nothing better exists."

This is the post-solve mirror of the *Formalization Checkpoint* (`frontier://skills/problem_framing`): there, confirm the model is right; here, confirm a small result is a real finding, not silence dressed as an answer. For translating these same fields when presenting to the user, see `frontier://skills/solution_interpreter` → *Frontier Quality and Completeness Signals*.

### Binding Constraint Detection

A binding constraint limits every (or nearly every) Pareto solution. Signs:
- All solutions sit at or near the constraint boundary
- Relaxing the constraint significantly changes the frontier
- The constraint prevents exploration of otherwise promising tradeoffs

When you detect a binding constraint, quantify the impact: *"Relaxing effort from ≤30 to ≤35 could dramatically expand the solution space. Is that limit truly non-negotiable?"* Suggest incremental relaxation, not removal.

For a richer view after solve, `explore tradeoffs` returns a `binding_analysis` block with shadow-price rates per binding constraint — how much each objective shifts per unit of slack relaxation, derived from the frontier's own slope. Use it to move the conversation from "this constraint is tight" to "it's costing you roughly $X per unit — worth negotiating?". See `frontier://skills/solution_interpreter` for presentation patterns.

On an **exact** continuous run (QP mean-variance or the purely linear LP allocation), `explore sensitivity` upgrades these to **solver-exact** duals — and they drive *formulation*, not just presentation. (What each field means — `where_to_invest`, `near_misses`, `capped_options`, `frontier_shadow_price_trend` — lives in `frontier://skills/solution_interpreter` → *Exact Sensitivity*, the single read-side reference; here, use them to choose the next model change:)
- **Where to invest → which constraint to renegotiate.** The largest `where_to_invest` shadow price names the limit whose relaxation buys the most. Take it back to the user as the highest-leverage knob: *"the volatility cap is the binding lever — each point of it is worth ~X return here; is 20% firm, or a target?"* — then relax it and re-run.
- **Near-miss → a re-scoring or cap-relaxation prompt.** A small `near_misses` reduced cost means an option is one assumption away from entering. Re-check its scores, or relax the cardinality / group cap (or the `capped_options` max-allocation limit) holding a wanted option out. A near-miss that persists across the frontier is a sign the formulation may be excluding something the user actually wants — route it back to problem_framing.

This closes the loop: exact sensitivity → a concrete formulation change → re-run. Continuous (QP/LP) only; on MILP, `explore sensitivity` falls back to the frontier-inferred estimate (tagged `frontier_inferred`), so lean on `binding_analysis` there.

### Sensitivity vs Scenario Analysis

These stress-test a decision in different ways — keep them distinct, and never narrate one as if it were the other.

- **Sensitivity analysis** (`explore sensitivity`; or the frontier-inferred `binding_analysis`) is *local and marginal*: how the optimal solution responds to a small change in a **single** input — one objective coefficient or one constraint's right-hand side — holding everything else fixed. It's derivative-like and basis-bound: valid only within the range where the current optimal basis holds. Read it as "at this solution, one more unit of X is worth ~Y," not as a forecast of a different world.
- **Scenario analysis** (`solve run_scenarios` → `explore scenario_results`) bundles a **whole set** of parameter changes into a named state of the world and **fully re-solves**. It's discrete and global, with no continuity assumption — each scenario is its own independent frontier, so the optimal mix itself can change, not just its rate of trade.

Pick by the question: marginal "what's this lever worth right here?" → sensitivity (then use it to drive a formulation tweak and re-run). Structural "what if the world were different?" → a scenario (then test robustness across futures with regret and CVaR). See *Scenario Results Interpretation* below.

### Reproducibility and Stability Checks

Evolutionary search is **stochastic**, but seeded: the `solve` tool accepts an optional `seed` and echoes `seed_used` on every run, so a run reproduces exactly. Same problem state + same seed → same frontier (in- and cross-process). Omit the seed and a fresh one is drawn and recorded.

| Use | What to do |
|---|---|
| Revisit or share a specific frontier | Note the `seed_used` from the first run; pass it as `seed` on any re-run to get the identical frontier (same problem state + same seed → same frontier). |
| Check frontier stability | Re-run with **different** seeds on the same problem. Solutions shifting slightly is normal; the *shape* (objective ranges, key tradeoffs) should be stable. If it isn't, the formulation is underspecified — objectives too correlated, constraints too loose, or not enough options. |
| Debug a "weird" result | Re-run with a different seed before assuming a modeling bug — one seed is one sample of the frontier, not the whole story. |

For `run_scenarios`, the parent `seed` is propagated to each scenario (per-scenario `seed_used` derived from the parent + scenario name), so pinning the parent reproduces the whole multi-scenario run while each scenario starts from a distinct initialization.

Don't pin seeds by default — fresh seeds each run give you a free stability check as the user iterates. Pin when reproducibility matters (handoff, regression, a specific frontier to preserve).

### Stale Results

When the problem structure changes (objectives, options, scores, constraints, approach), results are marked stale rather than cleared. This preserves the previous run for comparison. The `results_stale` flag in the status tells you when re-running is recommended.

### Re-run Judgment
- User changed one score slightly → probably don't need to re-run, frontier won't change much. But mention it. Results are marked stale.
- User added/removed an option → re-run, portfolio space changed.
- User added/removed an objective → definitely re-run, the landscape is fundamentally different.
- User changed constraints → re-run.

### Curated Solution Survival

After re-running, automatically check curated solutions against the new frontier using `explore curated`. Report survival:
- "3 of your 5 curated solutions survived the constraint change"
- "'Conservative Pick' is still in the frontier. 'Growth Bet' was eliminated — the tighter effort bound excluded it."

If key curated solutions are eliminated, explain *why* (which constraint change caused it) and suggest alternatives from the new frontier.

### Run Comparison

After re-running, use `explore compare_runs` to explain what changed structurally:
- **Criteria diff**: which constraints were added, removed, or changed between runs
- **Frontier diff**: how solution count and objective ranges shifted
- **Option coverage diff**: which options gained or lost Pareto membership

This turns iteration from "re-run and hope" into cause-and-effect reasoning:
- "After relaxing the effort constraint, high-impact features started appearing in solutions — the constraint was masking an entire class of approaches."
- "Adding the new objective split previously identical solutions into distinct clusters — there was a hidden tradeoff."
- "Removing the correlated objective didn't change the frontier shape, confirming it was redundant."

### Iteration Coaching
- "The frontier gave you 12 solutions. That's a lot of tradeoff space. Want to add a constraint to narrow it?"
- "Only 2 solutions survived. Your constraints might be too tight."
- 5-10 solutions on the frontier is a healthy number for most decisions.
- Results don't match intuition about combinations? Check the **aggregation method** on objectives (sum vs avg vs min vs max vs quadratic). A sum-aggregated cost objective behaves very differently from an avg-aggregated one. If an objective depends on interactions between options, use quadratic aggregation with an interaction matrix.

### Scenario Results Interpretation

When scenarios are defined, optimization runs separately per scenario. At the examine step:
- **Robust options** appear in solutions across all scenarios — safer bets regardless of which future materializes.
- **Scenario-specific options** excel in particular futures — conditional picks.
- Frame recommendations by risk tolerance: conservative → robust options, balanced → probability-weighted, aggressive → scenario-specific bets.
- If all options perform similarly across scenarios, scenarios aren't adding insight — simplify.

## Activation
Use this expertise at the boundary between modeling and solving, and when interpreting solver results that suggest structural changes.

## Scope Boundaries
- **Owns:** Solve execution — validate→run→examine loop, mode selection, solver selection (default NSGA vs optional exact backend), constraint strategy, infeasibility diagnosis, run comparison
- **Routes to solution_interpreter:** When results are healthy and ready for user presentation
- **Routes back to problem_framing:** When infeasibility or bad results indicate a structural problem (missing constraint, wrong objective, approach mismatch)
- **Routes back to data_collection:** When diagnostics suggest score quality issues (low variance driving flat frontiers, scale distortion)

## Guardrails
- Ensure the score matrix is complete before running — the optimizer requires every option scored on every objective.
- Let the user drive constraint additions — constraints should reflect real requirements, not be added just to shrink the frontier.
- Always diagnose infeasibility — when the solver returns no solutions, explain which constraints conflict and suggest the smallest relaxation that restores feasibility.
