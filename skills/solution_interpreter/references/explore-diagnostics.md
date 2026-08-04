# Explore Diagnostics — Field Reference

Detailed schemas for the richer diagnostic blocks returned by `explore tradeoffs`, `explore scenario_results`, and `explore composition`. Read this when you need to pin down exact field semantics — the SKILL.md narration patterns cover the usual cases.

## `binding_analysis` (from `tradeoffs`)

List of entries, one per binding constraint.

| Field | Type | Meaning |
|---|---|---|
| `constraint` | str | Human-readable description, e.g. `"budget ≤ 500000"`, `"cardinality ≤ 5"`, `"group_limit(A, B, C) ≤ 2"` |
| `constraint_type` | str | `"objective_bound"` \| `"cardinality"` \| `"group_limit"` |
| `binding_fraction` | float [0-1] | Fraction of frontier solutions at or very near the bound |
| `near_binding_count` | int | Count of those solutions |
| `shadow_prices` | list | Per-other-objective rate of improvement from relaxing this constraint |
| `note` | str\|null | Populated when shadow prices couldn't be estimated (e.g., no adjacent-cardinality solutions) |

### `shadow_prices` entry shape

For `objective_bound`:

```json
{"objective": "NPV", "slope_per_unit_relaxed": 0.014}
```

Slope is rate of change of the objective against the constrained quantity, estimated by linear regression among near-binding solutions. Sign is *direction of improvement*: positive means relaxing helps the objective, negative means it hurts. Rare for it to be negative (would indicate the constraint is currently protecting the objective on the frontier).

For `cardinality` (cap `≤ max` or floor `≥ min` — both use this key) and `group_limit` (cap side):

```json
{"objective": "NPV", "gain_per_additional_slot": 1.2}
```

Discrete shadow price: expected improvement of the best-on-that-objective solution from +1 slot of capacity. Estimated as (best at binding level) − (best at adjacent level).

A `group_limit` with a **floor** (`min > 0`) emits a second entry, labeled `group_limit(...) ≥ min` with key `gain_per_extra_member`: what one member above the floor achieves vs. pinned-at-floor — positive means the floor isn't pinching. Both sides can bind across one frontier (some plans pinned at the cap, others held up at the floor — or `min == max` pins every plan at both), so a single group can surface both a cap entry and a floor entry; read each by its label.

### Worked example

Budget binding on 18 of 20 solutions, with slope 0.014 for NPV and −0.002 for Risk:

> Budget is the tightest constraint — 90% of frontier solutions are pressed against it. On this frontier, each additional $1 of budget relaxation trades for about $0.014 of NPV and a 0.002 increase in Risk. Worth asking whether the $500k cap is a true limit or a planning target.

## `certify` (from `explore certify`)

The exact-overlay audit (`explorer.certify_against_exact`). Top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `nsga_run_id` / `exact_run_id` | str | The two runs compared (exploratory NSGA, exact overlay) |
| `exact_solver` | str | `"highs"` \| `"cuopt"` — the engine behind the overlay |
| `exact_certified` | bool | Every overlay point carries an optimality certificate — always true on the continuous LP/QP path (exact by construction); on MILP only at a zero gap (`exact=true`) |
| `rounding_bound_dips` | obj? | Proportional only: exact points whose whole-percent rounding dips a hair outside an `objective_bound` — a display artifact (the LP/QP honors the bound); present as a footnote, never as infeasibility |
| `nsga_count` / `exact_count` | int | Frontier sizes |
| `dominance_audit` | obj | The heuristic slack exact catches (below) |
| `coverage` | obj\|null | Hypervolume the overlay reclaims over NSGA alone (below); `null` on a degenerate flat-axis front |
| `invariant` | obj | The NSGA-never-dominates check (below) |
| `corner_sharpening` | dict | Per-objective exact-vs-NSGA best (below) |
| `headline_corner` | str\|null | The objective whose corner exact sharpened most (usually the risk corner) |
| `recommendation` | str | One-line synthesis — translate, don't echo |
| `next_steps` | str | Shape-branched onward guidance (sensitivity for QP, binding_analysis for MILP) |

### `dominance_audit`

```json
{"nsga_dominated_by_exact": 7, "nsga_dominated_fraction": 0.15, "examples": [...]}
```

Count + fraction of NSGA "Pareto" points the exact frontier strictly dominates — points that looked efficient but a provably better mix exists. The fraction is the headline trust number; `examples` are concrete dominated→dominating pairs.

### `coverage`

```json
{"nsga_hypervolume": 0.62, "combined_hypervolume": 0.74, "exact_reclaims": 0.12, "reclaimed_fraction": 0.16}
```

The hypervolume the exact overlay reclaims over the NSGA frontier alone — the *magnitude* behind the dominance count. All four values are normalized to [0,1] (the box, the same convention as a run's `hypervolume_normalized`), computed against a shared reference over the combined front so it is fair to both. `reclaimed_fraction` is the headline: it grows with problem size, and is near-zero on a small instance the heuristic already covers (exact confirms, doesn't expand). The whole block is `null` when the combined front is degenerate (a flat axis).

### `invariant`

```json
{"holds": true, "exact_dominated_by_nsga": 0, "note": "..."}
```

`holds` is the soundness check: NSGA should dominate **no** exact point. A nonzero `exact_dominated_by_nsga` is **not** a heuristic win — on a QP it's integer rounding of the continuous optimum to whole-percent allocations (read the `note`); MILP corners (integer by construction) never show it. Present a violation as a rounding footnote.

### `corner_sharpening`

Dict keyed by objective name:

```json
{"Volatility": {"nsga_best": 4.29, "exact_best": 4.07, "improvement": 0.22, "status": "sharpened", "is_risk_corner": true}}
```

`status` ∈ `sharpened` (exact beat the EA's best), `matched` (EA already optimal), `under-sampled` (negative improvement — rare now that every overlay anchor-samples each objective's extreme; chiefly a cardinality/group-capped QP, where anchors defer to the seeded search. An EA-budget artifact, **not** an exact-solver limit — raise the budget to close it). `is_risk_corner` flags the convex variance corner the decision usually hinges on.

## `scenario_risk` (from `scenario_results`)

Dict keyed by objective name.

| Field | Type | Meaning |
|---|---|---|
| `expected` | float | Probability-weighted (or equal-weight) mean of per-scenario best value |
| `worst_case` | float | The worst single scenario's best achievable value |
| `best_case` | float | The best single scenario's best achievable value |
| `cvar_<α%>` | float | Probability-weighted mean of values in the worst α-fraction of scenarios |
| `range` | [min, max] | Envelope across scenarios |

The top-level `cvar_alpha` echoes the α the user requested (default 0.2). The CVaR key name includes α as a percent (`cvar_20`, `cvar_10`), so your narration can stay adaptive: always pull the CVaR value via `scenario_risk[obj][f"cvar_{int(alpha*100)}"]` rather than hardcoding `cvar_20`.

Note the convention: **α here is the *tail fraction*, not a confidence level.** `cvar_alpha=0.2` is the worst 20% of scenarios — the inverse of the finance/textbook CVaR$_\alpha$ convention where α is the confidence (e.g. 0.95) and the tail is 1−α. In Frontier, larger α = a broader, less extreme downside.

### Interpretation notes

- "Worst α-fraction by probability mass": probabilities are summed ascending (worst-first for maximize, best-first for minimize) until the cumulative mass reaches α. CVaR is the weighted mean of those scenarios' values. With 4 scenarios at 0.4/0.2/0.3/0.1 and α=0.2, the worst one (0.1) plus half of the next-worst (0.1 of 0.2) contribute.
- Equal-weight scenarios make this straightforward — each scenario's probability is `1/n`.
- `scenario_risk` uses *best-in-scenario* values. It's a ceiling analysis, not a specific-solution analysis. For solution-level risk, a user must specify which solution to track (future extension).

## `regret` (from `scenario_results`)

Scenario minimax-regret. For each base-frontier solution, its value under each scenario is recomputed via `optimizer.score_slate` (scenarios are independent frontiers, so a base solution carries no cross-scenario values), then regret = best-achievable-in-scenario − this solution, normalized per objective range and clamped to [0,1].

| Field | Type | Meaning |
|---|---|---|
| `available` | bool | False when there are no scenarios, or no base `run` to evaluate |
| `method` | str | `"scenario_minimax"` |
| `normalization` | str | `"per_objective_range"` — regret as a fraction of each scenario objective's achievable spread |
| `per_objective` | dict | Per objective: `{min_max_regret, achieved_by_solution_id}` — the lowest worst-case regret on that objective over the **ranked** scenarios, and which solution achieves it |
| `per_solution` | list | Solutions by lowest `max_regret` (compact, ≤20): `{solution_id, content_signature, max_regret, mean_regret, by_scenario, feasible_in_all, feasible_in_ranked}` |
| `per_solution_total` | int | Full base-frontier count behind the compact `per_solution` list |
| `survivors_by_scenario` | dict | Per scenario: how many base plans stay feasible there |
| `minimax_choice` | obj\|null | The regret-robust pick: `{solution_id, content_signature, max_regret}` |
| `saturated` + `saturation_note` | bool + str | Every base plan hits total regret in some ranked scenario — the ranking is an all-1.0 tie, `minimax_choice` is omitted |
| `wipeout_scenarios` + `wipeout_note` | list + str | Scenarios with zero surviving base plans, excluded from the ranking |

- `max_regret` is the solution's worst regret across the **ranked** (scenario, objective) pairs; `minimax_choice` minimizes it. A total-wipeout scenario (no feasible base plan — see `survivors_by_scenario`) is excluded from the ranking and named in `wipeout_scenarios` + `wipeout_note`; its uniform 1.0 stays visible in `by_scenario`, and `feasible_in_ranked` distinguishes failing only the excluded wipeout from failing a ranked scenario. A 1.0 in `by_scenario` next to a sub-1.0 `max_regret` is the wipeout exclusion, not an inconsistency — a displayed 1.0 always means total regret (infeasible or fully dominated), never rounding.
- `feasible_in_all: false` means the solution breaks under at least one scenario's constraints — a 1.0 lands in `by_scenario` there, but it drives `max_regret` only when that scenario is ranked. Headline it when `feasible_in_ranked` is also false; when only a wipeout scenario fails, present the row normally and let `wipeout_note` carry the model-level finding.
- In `scenario_results`, each `per_scenario` entry also carries the lift: `base_plans_feasible` / `base_plans_total` (the same survivor counts, next to that scenario's own frontier).
- Distinct from `scenario_risk`/CVaR: CVaR is "how bad when things go wrong"; regret is "how much worse than the best I could have chosen, in hindsight." See *Regret — the hindsight lens* under `get_skill('solution_interpreter', section='Scenario Results Presentation')`.

## `objective_redundancy` (from `tradeoffs`)

List of entries, one per objective pair.

| Field | Type | Meaning |
|---|---|---|
| `objectives` | [str, str] | The pair |
| `classification` | str | `independent` \| `linear_redundant` \| `strong_tradeoff` \| `redundant` \| `nonlinear_dependent` |
| `pearson` | float [-1, 1] | Direction-normalized r: +1 = both improve together, -1 = one improves as other worsens |
| `mutual_info_normalized` | float\|null [0-1] | Dependence strength (0 = independent, 1 = deterministic). Null when n<15. |
| `flags` | list[str] | Diagnostic flags: `high_pearson_aligned`, `high_pearson_opposed`, `high_mi`, `pearson_mi_disagreement` |
| `mi_reliable` | bool | False when n<15 — MI wasn't computed, only Pearson-based classification |

### Classification logic (for reference, do not re-derive)

Pearson `correlation` is direction-normalized — each objective's raw values are sign-flipped when its direction is `minimize`, so r ≥ +0.7 means both improve together and r ≤ -0.7 means the objectives are in genuine tension. Classification:

- `linear_redundant`: r ≥ +0.7 (both improving together)
- `strong_tradeoff`: r ≤ -0.7 (genuine conflict — NOT redundancy; this is the tradeoff users hired the optimizer to find)
- `redundant`: MI ≥ 0.7 when neither strong_tradeoff nor linear_redundant applies
- `nonlinear_dependent`: |r| < 0.3 **and** MI ≥ 0.4 (the informative Pearson/MI disagreement case)
- `independent`: none of the above

When `mi_reliable` is false, only Pearson-based classifications (`linear_redundant`, `strong_tradeoff`) can be detected reliably — treat MI-driven classifications with caution and say so to the user.

### Worked examples

High Pearson, high MI (both agree):

> Revenue and Profit move together almost perfectly on this frontier. The solver is essentially treating them as one axis — consolidating them would make the tradeoffs sharper.

Low |r|, high MI (disagreement):

> User Satisfaction and Retention look uncorrelated (r=0.12), but they're actually strongly dependent (MI=0.62). That usually means a threshold or plateau — e.g., satisfaction above some level drives retention but below that it doesn't move it. Worth looking at the scatter before deciding whether to keep both.

## `composition` (from `explore composition`)

Mining the solution set — operates on the active frontier, or a curated subset via `solution_ids` / `signatures`.

| Field | Type | Meaning |
|---|---|---|
| `scope` | obj | `{set: "frontier"\|"curated", n_solutions, approach}` |
| `option_selection` | list | Per option, sorted by `selection_pct` desc (also on `tradeoffs`, and per-option on `solution` via `option_context`) |
| `co_occurrence` | list | Option pairs ranked by departure from independence (top 8; all when `detail=true` — past the detail cap the top-ranked pairs stay inline, `truncated` echoes total vs shown, and the full list is on disk at `full_result_path`) |
| `design_principles` | list | Statements that hold across the set |
| `clusters` | list | Decision-space strategy families (omitted when too few solutions) |
| `clusters_summary` | obj | `{n_families, largest_family_share}`, plus a `note` when the share exceeds 0.9 — rides with `clusters` |
| `feedback_rules` | obj | Liked/disliked separators, when feedback exists |

### `option_selection` entry

Binary: `{option, selection_count, selection_pct}`. Proportional adds `mean_weight` (over all solutions) and `mean_weight_if_included` (over solutions holding it). Read as consensus (≈1.0) vs distinctive (low).

### `co_occurrence` entry

```json
{"options": ["A", "B"], "lift": 2.8, "relation": "complement"}
```

`lift = P(A∧B) / (P(A)·P(B))`: >1 complements (travel together), <1 substitutes. Ranked by `|lift − 1|` — no value cutoff, salience *is* the ranking.

### `design_principles` entry

```json
{"type": "always", "options": ["A"], "support": 1.0, "detail": "Every solution on this set includes A."}
```

`type` ∈ `always` / `never` (selected in all / none), `co_occurs` / `substitutes` (strong lift), `region_bound` (presence tracks an objective — point-biserial |r| ≥ 0.6). `support` is the strength (a fraction for always/never; |r| for region_bound; lift-derived otherwise).

### `clusters` entry

```json
{"cluster_id": 1, "size": 7, "representative_solution_id": 3,
 "defining_options": ["Mobile", "API"],
 "objective_signature": {"Impact": {"min": 6, "max": 9, "mean": 7.5}}}
```

Clustering is in **decision space** (option/allocation composition), not objective space — so two families can share an `objective_signature` yet differ in `defining_options` (different plans, same outcome → the choice between them is low-regret). `k` comes from the largest gap in merge distance, capped at a legible handful; `defining_options` are options far more common inside the family than outside (empty = a weakly-separated family). `clusters_summary.largest_family_share` prices the split: near 1.0 it is one strategy with corner variants (a 104/1/1/1), so present the family rather than a menu.

### `feedback_rules`

```json
{"available": true,
 "rules": [{"condition": "solution includes A", "separates": "liked", "separation": 1.0, "coverage": 1.0}],
 "note": "..."}
```

`available: false` (with a `note`) when there isn't rated-both-ways signal yet. A rule surfaces only when it *measurably* separates liked from disliked (`separation` × `coverage`), never on a rating count. Treat a clean rule as a candidate latent constraint → route to `problem_framing`.
