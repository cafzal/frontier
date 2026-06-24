# Frontier – Architecture Reference

**Related docs:** [`best-practices.md`](best-practices.md) – skill, prompt & MCP design guidelines | [`README.md`](README.md) – user setup and usage guide

## Design Principle: Deterministic Guardrails + Human Judgment

Frontier's architecture separates what must be **deterministic** from what requires **human judgment**: the split that makes its optimization explainable and governable. Every component below sits on one side of this line.

- **Deterministic guardrails (engine layer).** Computed from problem state, not model output, so the same inputs always yield the same result. Hard-constraint enforcement during search (`optimizer.py`, 8 constraint types – never violated); Pareto dominance filtering (NSGA-II/III); seeded reproducibility (`seed` / `seed_used` – same inputs + seed reproduce the exact frontier, in- and cross-process; `optimizer._seeded_rng_fallback` makes pymoo's otherwise-unseeded survival/selection RNG deterministic); pre-solve constraint-conflict validation (`optimizer._check_constraint_conflicts`); infeasibility analysis with relaxation suggestions; and frontier quality gates (`metrics.frontier_quality` → GOOD/WARNING/POOR). `metrics.py` states the contract directly: "deterministic signals from problem state … not LLM-generated."
- **Human judgment (agent + skills layer).** The skills steer the two calls that stay with the human – *framing* (objectives vs. hard constraints) and *selection* (which non-dominated solution to commit to). `solution_interpreter` enforces "never say best" and **traceable claims**: every quantitative statement must trace to returned data (a score, an objective value, a shadow price, a dominance relation), so a stakeholder can audit the reasoning line by line. Curation is the human's act of choosing; the engine records it but never makes it.

**Tighter optimality is an optional tier of the same determinism.** The default NSGA path is deterministic and reproducible but *heuristic* – it approximates the frontier. On supported problem shapes the agent can elect an exact-solver backend ([§ Solver Backends](#solver-backends-pluggable)) that makes each frontier point *optimal to a 0.1% gap* (a certified zero gap with `exact=true`) rather than approximate – a stronger form of the engine-side guarantee. Choosing *when* to pay for that **exact pass** is itself the human/skill judgment (the `optimization_strategy` skill guides it); the exact run, once elected, is deterministic. So exact solvers don't add a third layer – they deepen the deterministic guardrail and hand the agent one more judgment call.

The layers in §2 realize this split: the **Engine Layer** is the deterministic core, the **Skills** are the judgment scaffold, and the **MCP Tools** are the interface between them. Read the rest of this document through that lens.

## Tools & Skills Reference

### MCP Tools

Frontier exposes 4 tools – 3 domain tools with multiple actions, plus a skill delivery tool:

| Tool | Action | Purpose |
|------|--------|---------|
| **model** | `create` | Start a new optimization problem (name, domain, context, approach) |
| | `update` | Add/modify objectives (≥2 enforced; 2–7 is the designed envelope), options, scores, constraints, reference points, scenarios. Merge semantics for scores; full replacement for everything else. Marks results stale on structural changes. |
| | `get` | Return problem state. Defaults to the `summary` slice (counts + status flags — always small). Optional `section` for targeted slices: summary, objectives, options, scores, constraints, matrices, scenarios, run, runs, exact_run, curated, references — or `full` for the complete dump (opt-in; can exceed token caps on large models). |
| | `list` | List all problems with metadata snapshots |
| | `delete` | Remove a problem and its data file |
| | `save` | Save a problem to the user's library (`saved/<name>/`) in the portable examples format. Params: `problem_id`, `save_as` (name; defaults to a slug of the problem name). Always includes `solutions.json` when the problem has a run. |
| | `load` | Load a problem by name in the portable format, resolving `saved/` first then bundled `examples/` (a save shadows a like-named example). Mints a fresh `problem_id`, registers it in the store, restores scenarios + prior results. Omit `source` to list available names. |
| **solve** | `validate` | Pre-flight check: ≥2 objectives, ≥3 options, complete score matrix, feasible constraints. Also returns a `solvers` block – which engines are `available` in this environment and whether the optional exact backends `exact_fits_shape` for this problem – so the agent can make an informed `solver` choice. |
| | `run` | Validate then optimize. Returns compact result – `objective_ranges`, `preview` (per-objective extremes + balanced solution by id/objective_values only), `quality`, `seed_used`, `total_pareto_found` (pre-pruning count), `frontier_complete` (bool – true when returned set is the full Pareto frontier, false when pruning truncated it), `frontier_quality` (status GOOD/WARNING/POOR with progressive gates and issues), `metrics`, `full_result_path` (disk path to complete JSON with all solutions – preferred for bulk export or artifact assembly), and `next_steps` pointer. Full solution detail also retrievable via `explore solutions` (`solution_id=` for one). Optional `mode`: "fast" (default, quick iterations) or "thorough" (final convergence). Optional `max_solutions` caps Pareto set size (default 100). Optional `seed` (int) for reproducibility (same inputs + seed → same frontier; echoed as `seed_used`); omitted = fresh random seed drawn and echoed. Auto-selects NSGA-II (2-3 obj) or NSGA-III (4+ obj). Parameters adapt to solution space size and objective count. Optional `solver` selects the engine – default NSGA (evolutionary, any shape) or an opt-in exact backend `"highs"`/`"cuopt"` (each frontier point optimal to a 0.1% gap on supported shapes, zero-gap with `exact`; see *Solver Backends*); the engine that ran is echoed as `solver_used`. Optional `exact` (bool) certifies each **MILP** inner solve to a zero gap (no-op on the always-exact QP/LP and on the default NSGA path). Optional `scope` (exact backends, any supported shape) picks how the overlay is produced: `"curated"` (default) progressively certifies the existing NSGA `run`'s frontier points (the lean explore-then-certify path; falls back to a full pass when there's no run yet), `"full"` runs the exact solver on every NSGA evaluation (an exact-*guided* search); the mode that ran is echoed as `overlay_scope`. A requested exact solver that is uninstalled or ill-fitting returns a clear error rather than silently falling back. **Long runs execute in the background** (see *Long-running solves*): the response is either the compact frontier result (fast solves, unchanged) or a `{status:"running", job_id}` handle to poll via `solve status`. Optional `time_limit` (s) caps wall-clock – the run stops at its budget or the cap, whichever first, returning a best-so-far frontier flagged `time_limited` (echoed on the result). Optional `wait_seconds` tunes the inline wait before a handle is returned (0 = return a handle immediately). |
| | `run_scenarios` | Independently optimize each scenario with score overrides/adjustments. Accepts optional `mode`, `max_solutions`, and `seed` (per-scenario seeds are deterministically derived from the parent so each scenario reproduces while starting from a distinct initialization), plus `time_limit` / `wait_seconds`. Also background-dispatched, same as `run`. |
| | `status` | Poll a background solve by `job_id` (from a `run`/`run_scenarios` that returned `status="running"`). Returns `status="running"` (with `elapsed_s` + a `label`) until the solve finishes, then the full solve result (with the `solution_interpreter` skill, injected once per problem at delivery). A `status="stale"` result means the problem's solve-inputs were edited mid-solve – nothing was overwritten; re-run. |
| **explore** | `tradeoffs` | Frontier overview: total solution count, objective ranges, correlations + normalized MI per pair (MI computed when n≥15), extremes, balanced solution (ideal-point closest), inflection-point candidates (diminishing-returns boundaries), frontier shape classification per conflicting pair (linear / concave / convex / discontinuous, with confidence), binding_analysis (shadow-price rates per binding constraint – how much each objective shifts per unit of slack relaxation, derived from frontier; objective_bound / cardinality / group_limit), objective_redundancy (classification per pair using Pearson + MI, flags Pearson/MI disagreement = non-linear dependence), vs references. Optional `scenario` param targets a specific scenario's frontier; optional `source="exact"` retargets analysis at the `exact_run` overlay instead of the default NSGA `run` (threaded through compare / solutions / curate / marginal_analysis too). Every analytics result (tradeoffs / compare / solutions / marginal_analysis) echoes `frontier_source` (`{run_id, solver, kind: heuristic\|exact}`) so a dropped or omitted `source` can't pass a heuristic frontier off as exact; a heuristic frontier served while an exact overlay exists also flags `exact_overlay_available`. |
| | `sensitivity` | Solver-exact duals from an exact continuous run (QP mean-variance or LP allocation): `where_to_invest` (constraint shadow prices, ranked – marginal objective change per unit relaxed) + `near_misses` (unheld options by reduced cost, closest first) + `capped_options` + `frontier_shadow_price_trend`, anchored on a reference solution (default balanced; `solution_id` overrides). The *exact* counterpart to `binding_analysis`; falls back to the frontier-inferred binding analysis on heuristic/MILP runs. Tagged `source=solver_exact｜frontier_inferred`, echoes `frontier_source` (run provenance, like the other analytics actions), and names the `optimized_objective` (the ε-constraint primary the duals are measured against — interpretations anchor on it by name); continuous (QP/LP) only (integer/MILP solutions have no exact duals). |
| | `compare` | Side-by-side comparison of 2+ solutions (shared/differentiating options, tradeoff summary) — by `solution_ids`, or by `signatures` to compare the curated set (absorbs the old `compare_curated`). Optional `scenario`, `detail`. |
| | `solutions` | Pareto frontier listing — or one solution's detail (with reference-point analysis) via `solution_id` (absorbs the old `solution`). Default: compact list (solution_id + objective_values + content_signature); `detail=true` for the full dump. Optional `scenario`. |
| | `feedback` | Record user feedback: solution_id or content_signature, rating (1-5), notes, stage. Links to content_signature (stable across runs) and attaches to matching curated solution. |
| | `compare_runs` | Diff two runs: criteria changes, frontier diffs, option coverage. Defaults to current-vs-previous run; `run_ids` (2+) overrides with explicit history. |
| | `certify` | Audit the exploratory NSGA frontier against the exact overlay. **No params**: audits the current `run` against the `exact_run` overlay (solve, then solve with an exact solver, then certify). Optional `run_ids` = exactly two (one NSGA + one exact, order-free – exact auto-detected by its solver) override with explicit historical runs. Returns a solver-agnostic certificate: `dominance_audit` (which NSGA points an exact solve dominates + fraction), `coverage` (hypervolume the overlay reclaims over NSGA – `reclaimed_fraction`), `invariant` (NSGA never dominates an exact point), `corner_sharpening` per objective (exact vs NSGA best, status sharpened/matched/under-sampled, is_risk_corner), `headline_corner`, and a `recommendation`. See *Solver Backends*. |
| | `audit` | Witness / feasibility auditor over the **whole feasible space** (binary problems, HiGHS) – the feasibility-side sibling of `certify`, reasoning about every feasible plan rather than the non-dominated frontier. **No prior solve required**: reads the model's constraints directly. Optional `audit_property` (a constraint-shaped dict, same vocabulary as model constraints). With no property → feasibility probe (`verdict` feasible + a witness plan / no_feasible_plan – the exact form of `validate`'s pre-solve check). With a property → proves a guarantee: it holds for **every** feasible plan (`verdict` holds, the negation infeasible across the whole region) or returns a concrete counterexample (`verdict` violated + `witness`), plus holds_vacuously / inconclusive, the `feasible_region` the verdict is conditional on, and a `recommendation`/`next_steps`. An unfit shape (non-binary / non-sum / HiGHS missing) returns a tool error, matching `solve`'s exact gate – no "unsupported" verdict. See *Solver Backends*. |
| | `scenario_results` | Per-scenario analysis with frequency-weighted option importance. Returns option_robustness sorted by importance (avg_frequency x avg_weight) with tiers: core (>50% in all scenarios), common (>25%), marginal (<25%). Also: scenario-specific options, expected values (ideal-point, probability-weighted), scenario_risk per objective (expected / worst_case / best_case / cvar_<alpha>%), and `regret` (scenario minimax-regret per solution + `minimax_choice`, with scenario-infeasible solutions flagged – a distinct lens from CVaR: "how much worse than the best achievable, in hindsight"; each base solution re-evaluated under every scenario via `optimizer.score_slate`, since scenarios are independent frontiers; when every base solution saturates at max regret the block sets `saturated` + a saturation note and omits `minimax_choice` rather than nominate an all-tied pick, and `per_solution_total` counts beyond the listed top 20). Optional `cvar_alpha` (float in (0,1), default 0.2) sets the CVaR tail fraction. |
| | `scenario_frontiers` | Per-scenario Pareto frontiers overlaid for visualization. `viz_data` colors each scenario's frontier for the web UI's parallel-coordinates overlay and emphasizes curated picks (bold, colored by the scenarios they're feasible in); ASCII surfaces a per-scenario objective-range table. Shows how the achievable tradeoffs shift across futures. |
| | `curate` | Pin a solution to the curated set with custom name and notes (optional `scenario` for scenario frontiers) — and manage existing pins: `remove=true` unpins, `rename="…"` renames, both by content signature (absorbs the old `uncurate` / `rename_curated`). |
| | `curated` | List all curated solutions with `in_current_frontier` survival flag — or render the handoff export with `format="markdown"\|"csv"` (columns: name, content_signature, objective values, selected_options; invalid format errors cleanly; absorbs the old `export_curated`). |
| | `marginal_analysis` | Marginal rate analysis: cost-per-unit between adjacent solutions, inflection point detection (where marginal cost jumps sharply). Default summary; `detail=true` for per-pair breakdown. Optional `scenario` param. |
| | `composition` | Mine the solution set (knowledge-discovery pillar): per-option `selection_count`/`selection_pct` (consensus vs distinctive – also surfaced on `tradeoffs`, and per-option on `solution` via `option_context`), `co_occurrence` lift (complements/substitutes), `design_principles` (always / never / region-bound), decision-space `clusters` (strategy families, each with defining options + objective signature – clustering in decision space, not objective space), and `feedback_rules` (rules separating liked from disliked solutions when feedback exists → candidate latent constraints). Operates on the active frontier, or a curated subset via `solution_ids` / `signatures`. |
| **get_skill** | *(single action)* | Retrieve workflow guidance by name. Without `section`: the skill's core (SKILL.md). With `section=<heading>`: exactly that section, resolved across the core **and** the skill's `references/` files — the cheap recovery path every decision response's `guidance_pointer` names. Unknown sections error with the available titles. Works with all MCP clients (unlike resources, which require client-side resource support). Available skills: `problem_framing`, `data_collection`, `optimization_strategy`, `solution_interpreter`. |

**Rendering surfaces.** `explore` actions (and `model get` summary) emit both an ASCII `visualization` for chat / coding-agent clients and a structured `viz_data` payload for chart-rendering surfaces. The web UI renders `viz_data` via Plotly: frontier solutions adapt to dimensionality (2 obj → 2D scatter, 3 → 3D scatter or a 2D scatter colored by a user-chosen third objective, ≥4 → parallel coordinates), with a per-chart toggle (3D / 2D-color / parallel-coordinates, as the dimensionality allows); scenarios overlay as colored parallel coordinates with curated picks emphasized, and the formulation renders as a typed card. Clicking a point (or brushing parallel coordinates) curates it through the agent.

### MCP Skills (Resources + Tool)

Skills are available two ways for maximum client compatibility:
1. **`get_skill` tool** (universal) – works with any MCP client. Call `get_skill('problem_framing')` etc.
2. **MCP resources** (backward compat) – `frontier://skills/*` URIs, for clients that support resource reads.

Skills provide domain guidance the agent consults at each workflow stage:

| Skill | Purpose |
|-------|---------|
| **problem_framing** | Translate decision language into objectives/options/constraints. Covers objective vs constraint classification (principle-based, not keyword matching), hidden objective detection, approach selection (binary vs proportional – "does quantity matter?"), aggregation modes (canonical definition – sum/avg/min/max/quadratic), interaction matrices for quadratic aggregation, reference points, scenario definition, question anchors for guiding problem exploration. Cross-referenced by other skills. |
| **data_collection** | Guide score elicitation. Covers data readiness levels, best-worst anchoring, batch efficiency, source evaluation, conflict resolution, score quality signals (variance, scale mismatch), aggregation implications on scoring (cross-references problem_framing), score provenance/attribution, completeness drive. |
| **optimization_strategy** | Drive solve progression. Core: iteration expectations, validate→run→examine flow, background-job literacy, mode + solver selection (incl. the staged exact-certify trigger and engine table), constraint strategy (cross-references problem_framing for types), infeasibility response, binding constraint detection, curated solution survival tracking, run comparison, stale results and re-run judgment. Deep exact-solver narration (claim calibration, gate declines + redefines, `exact=true`) lives in `references/exact-solvers.md`, fetched as the *Exact Solvers — Depth* section. |
| **solution_interpreter** | Present results without bias. Core Judgment (always apply, injected): "never say best", traceable claims, five explanation dimensions, presentation order (Extremes → Balanced → Inflection → Risk → Preference), tradeoff framing, objective ranking elicitation, dominance explanation, question anchor. The 22 situational *Presentation Refinements* (certificate reading, audit reading, exact sensitivity, scenario presentation, composition mining, curation, the stakeholder writeup + why-triplet, diagnostics, …) live in `references/presentation-refinements.md` — fetched per section via `get_skill(…, section=…)` at the moment each output is presented; `references/explore-diagnostics.md` holds raw field schemas. |

---

## 1. User Workflow

The user describes a decision in plain language; the agent translates it into Frontier's model and drives the loop: **frame** (objectives, options, constraints, scenarios — `model`) → **score** (`model update`) → **solve** (`solve validate/run/run_scenarios`, optionally per scenario or with an exact backend) → **explore** (tradeoffs, comparisons, scenario robustness, certification — `explore`) → **refine and re-solve** → **curate and decide**. Each phase transition auto-injects the matching skill (§ Skill Auto-Injection), so the agent gets framing judgment before modeling, scoring judgment before data entry, solve strategy before running, and presentation judgment before narrating results. The full per-action contract is the tools table above.

## 2. System Architecture

Three layers between the agent client and the data on disk:

- **MCP server** (`mcp_server/`, FastMCP, stdio/SSE) — `server.py` owns the 4 tools (`model`, `solve`, `explore`, `get_skill`), `frontier://skills/*` resources, response compaction, and the optional bearer gate; `guidance.py` owns skill loading/injection state and the decision guidance pointers; `jobs.py` owns the background solve jobs.
- **Engine** (`engine/`) — `models.py` (the canonical `Problem`: objectives, options, scores, 8 constraint types, interaction matrices, scenarios, runs, curation, feedback), `optimizer.py` (NSGA-II/III via pymoo; binary + proportional modes, constraint encoding, quadratic aggregation, scenario optimization, infeasibility analysis), `explorer.py` (tradeoff/scenario/curation analytics, certification, sensitivity, ASCII renderers), `viz.py` (the structured `viz_data` chart builders the web UI mirrors), `metrics.py` (deterministic framing/data/solve metrics, `frontier_quality` gates), `store.py` (atomic JSON persistence to `data/{problem_id}.json`), `problem_io.py` (portable save/load bundles).
- **Exact backends** (`solvers/`) — `highs_backend.py` (CPU) and `cuopt_backend.py` (GPU) behind one scalarization engine (`_scalarization.py`) and one shape/availability gate (`__init__.py`); opt-in per run via `solve(solver=…)` (§ Solver Backends).

The skills (`skills/*/SKILL.md` + per-skill `references/`) are data, not code: markdown the server injects and serves, carrying the judgment layer described above.

### Problem Persistence: Live Store vs Portable Bundles

A problem persists two ways: the **live store** (`store.py` → `data/{problem_id}.json`) holds the full canonical `Problem` keyed by UUID (session state – a stateful deployment just mounts a disk here; the bundled render.yaml blueprint deliberately doesn't yet, so hosted-beta state is ephemeral); writes go through an **atomic** `save` (temp file + `os.replace`), so a concurrent reader never sees a half-written file and a background solve worker can persist without tearing a foreground edit (see *Long-running Solves*). **Portable bundles** (`problem_io.py`, the `model save` / `load` format the [`examples/`](examples/) use) split it across `problem.json` (definition), `scores.json` (data), and `solutions.json` (results, written when solved). `save` writes to a gitignored `saved/` library; `load` resolves a name against `saved/` then `examples/`, minting a fresh `problem_id`. `problem_io` is also where the examples' top-level `scenarios` list is bridged to `scenario_config` (`enabled=True`) – a naive `Problem(**problem_json)` would drop scenarios, since the field is `scenario_config` and pydantic ignores unknown keys.

### Solver Backends (Pluggable)

The default solve path is pymoo's NSGA-II/III for all problem shapes. Two **opt-in exact-solver backends** sit alongside it as an **optional audit / certification layer** over that heuristic exploration, gated so the default path never imports either (each imports its solver lazily, so the modules load whether or not that solver is present). Both share **one NSGA-scalarization engine** ([`solvers/_scalarization.py`](solvers/_scalarization.py)): NSGA-II evolves epsilon-constraint scalarization targets – and, under a cardinality/group cap, an asset-selection priority vector – while an injected **inner solver** solves each scalarized single-objective problem to optimality, and the Pareto frontier is assembled from those exact inner optima. NSGA explores the scalarization space; the inner solver makes each evaluated point optimal to a 0.1% gap instead of heuristic. The two backends differ **only** in that injected inner solve, and both return the engine's exact `Run` / `Solution` shape – so explorer, metrics, and store need zero changes. Additive, gated, reversible. In OR terms this is the *sequential-collaborative* matheuristic pattern – the heuristic explores, the exact solver certifies – reusing the same optimality-gap-against-a-bound a MILP solver already reports for its own incumbent, here applied across a Pareto front rather than a single objective.

**The exact run is an auditor, not a replacement.** Because each inner solve is optimal for its scalarization, overlaying exact points on the heuristic frontier can only **confirm or improve** it, never worsen it – the invariant *NSGA never dominates an exact point* (strict for the MILP, which is integer by construction; for the QP, rare violations can only come from integer-rounding the continuous optimum to whole-percent allocations, and are reported as such, not as a heuristic win). In practice exact does three things: it **sharpens the convex risk/variance corner**: the flat convex bowl where heuristics wobble and a QP is exact, i.e. the minimum-risk portfolio a risk-averse decider hinges on (min-volatility, most-diversified sourcing, firmest grid); it **audits heuristic slack** by flagging NSGA points shown as efficient that an exact solve dominates; and on a binary MILP it returns the **optimal subset** for its targets (to a 0.1% gap, certified zero-gap with `exact=true`). Where exact instead looks "short" of NSGA at a linear extreme, that's the EA's epsilon-target sampling budget – a denser sweep or targeted solve closes it – not a capability limit.

**Agent-selectable per run (provable optionality).** Solver choice is a **per-run decision made by the agent**: there is one selection mechanism, the `solver` argument, no deployment-wide env toggle. Solver choice is problem-shape- and workflow-phase-dependent (explore with the EA, certify a finalized pick with an exact solver), so a global default would fight that flow rather than help it. `solve(solver="highs"|"cuopt", exact=…)` threads through `optimizer.optimize(..., solver=…)`; omitted (or `"nsga"`) runs the default evolutionary search. [`solvers/__init__.py`](solvers/__init__.py) owns the two pure checks both the tool and `optimize()` consult – `available_solvers()` (probes importability with `importlib.util.find_spec`, so it answers without importing the solver) and `exact_solver_fits(problem)` (the shared shape gate, one scope for both backends: **binary selection** with every objective `sum`-aggregated – the MILP is linear in the 0/1 vars, so `avg` (fractional over a variable-size pick), `min`/`max`, and `quadratic` (nonlinear) are each declined *with a redefine hint* rather than silently optimized as a sum; or **proportional mean-variance**: a **minimize**-direction `quadratic` *risk* objective backed by an interaction matrix alongside `sum`/`avg` linear objectives, with `min`/`max` (nonlinear) and a *maximize* quadratic (non-convex) declined; or **proportional purely-linear**: ≥2 `sum`/`avg` objectives and no quadratic term → an exact multi-objective LP). The `solve` tool guards on both before running and returns an actionable error if the requested solver is uninstalled or ill-fitting – it never silently degrades to NSGA, so a result that claims optimality always ran an exact engine. Every run records `Run.solver` / `Run.exact` and echoes `solver_used` in the response, keeping the frontier traceable to the engine that produced it (the same "traceable claims" contract the skills enforce). The `optimization_strategy` skill's *Exact Solvers* section carries the agent-facing workflow guidance: when to upgrade from EA exploration to an exact-audited frontier, and how to narrate it without over-claiming.

The two backends are **first-class and co-equal**: the same auditor engine over two inner solves, one GPU and one CPU; pick by the hardware on hand, not by tier.

- **NVIDIA cuOpt** (`solvers/cuopt_backend.py`, `solver="cuopt"`) – the **GPU exact backend**: a convex min-variance **QP** for proportional mean-variance portfolios, a 0/1 **MILP** for binary selection, and an exact **LP** for purely linear proportional allocation. The continuous duals (QP + LP shadow prices / reduced costs, read via `get_dual_solution` / `get_reduced_cost`) are surfaced through `explore sensitivity`, at parity with HiGHS. The QP has two equivalent builds – the term-by-term high-level `Problem` API (`_solve_qp_cuopt`, GPU-verified) and a scalable matrix `data_model` / CSR build (`_solve_qp_cuopt_matrix`) that drops the O(n²) Python-expression construction for a dense covariance to one vectorised conversion; both pack the same covariance, so they return the same optimum (the `_USE_MATRIX_QP` flag selects; the matrix build is the default, GPU-confirmed equivalent to the term-by-term path). Needs `cuopt-cu12` + a GPU, so it runs in Colab. cuOpt MILP solves run in **deterministic mode** (`CUOPT_MIP_DETERMINISM_MODE`) – the mode Frontier's seeded-reproducibility contract needs (cuOpt's default *opportunistic* mode can return a different optimum run-to-run), which as a side effect takes the *sequential* LP root relaxation: the epsilon sweep proposes infeasible corners the engine expects (scored dominated), and cuOpt's *concurrent* root solve aborts the process (`std::terminate`, an uncatchable kernel crash) on an infeasible relaxation, whereas the sequential path returns `Infeasible` as a clean `ok=False` like HiGHS – a configuration fix, not a workaround.
- **HiGHS** (`solvers/highs_backend.py`, `solver="highs"`) – the **CPU exact backend**: the same engine with HiGHS inner solves. Exact **MILP** for binary selection with additive (`sum`) objectives (every Frontier combinatorial constraint – cardinality, force in/out, dependency, exclusion, group limit, objective bound – is linear-integer, handled exactly; the [`capital_project_selection_120`](examples/capital_project_selection_120) showcase) and convex **QP** for mean-variance portfolios (linear `sum`/`avg` objectives plus the quadratic risk term); and exact **LP** for purely linear proportional allocation (a linear objective optimized, the rest ε-constrained, with shadow prices + reduced costs). HiGHS can't solve MIQP, but never has to: under a cardinality/group cap the EA picks *which* assets are eligible and HiGHS solves the continuous QP on that support exactly. The solver is a plain `pip install highspy` (CPU, cross-platform, no special index, no GPU), so the exact path – and the shared NSGA engine with it – runs and is tested on the same machine as the engine ([`tests/test_highs_backend.py`](tests/test_highs_backend.py)).

Both backends accept `optimize(..., exact=True)` to certify each MILP inner solve (zero gap) instead of the default speed-oriented bounded solve. The independent scalarizations can also run concurrently through a thread pool (`_parallel_solve`, solver-agnostic – cuOpt releases the GIL during its GPU solve, HiGHS during its CPU solve), the DIY `concurrent.futures` pattern that cuOpt's deprecated `BatchSolve` now points to. The matrix QP build and parallel solve are covered, on the CPU-checkable path, by [`tests/test_cuopt_matrix.py`](tests/test_cuopt_matrix.py).

**Certifying a heuristic frontier against exact.** `engine.explorer.certify_against_exact(problem, nsga_run, exact_run) -> dict` ([`engine/explorer.py`](engine/explorer.py)) makes the auditor framing a concrete, returnable artifact: it overlays an exact run on an NSGA run of the same problem and returns a certificate with four parts – a **`dominance_audit`** (`nsga_dominated_by_exact`, `nsga_dominated_fraction`, `examples` – the heuristic slack exact catches), a **`coverage`** block (`reclaimed_fraction` / `exact_reclaims` – the hypervolume the overlay reclaims over NSGA alone, the magnitude behind the dominance count, normalized against a shared reference so it is fair to both fronts and grows with problem size), the **`invariant`** (`holds`, `exact_dominated_by_nsga`, `note` – confirming NSGA never dominates an exact point, with the QP integer-rounding caveat surfaced in the note rather than asserted away), and **`corner_sharpening`** per objective (`nsga_best`, `exact_best`, `improvement`, `status` ∈ `sharpened` | `matched` | `under-sampled`, `is_risk_corner`), plus a `headline_corner` and a plain-language `recommendation`. The certificate is scoped to *per-scalarization optimality + dominance + coverage* – deliberately **not** a proof that the whole frontier is optimal (no single optimum exists for a true multi-objective problem); the invariant is a soundness check on the overlay, not an independent quality score. It is surfaced as the MCP **`explore certify`** action. An exact solve is stored as the **`exact_run` overlay** (`Problem.exact_run`) *alongside* the exploratory NSGA `run` – never replacing it, and separately explorable via `explore … source="exact"` – so the "explore with the EA, then certify" flow needs **no params**: `certify` audits the current `run` against `exact_run` (two explicit `run_ids`, order-free, override that). The certificate is **solver-agnostic** (identical on HiGHS or cuOpt). Covered by [`tests/test_certify.py`](tests/test_certify.py).

**Two ways to produce the `exact_run` overlay.** A full pass — `optimize(solver=…)` — runs the exact inner solver on *every* one of the ~pop×gen NSGA evaluations (the inner solve **is** NSGA's fitness): a separate exact-guided search. The lean alternative — `optimizer.certify_curated(problem, source_run, solver=…)` — exact-solves **only the points of an existing run** (the curated NSGA frontier), re-solving each for its own support + epsilon targets and assembling them into the same non-dominated, duals-carrying exact `Run`. This is the literal *explore-then-certify* path: the heuristic explores broadly, the exact solver certifies only what it kept, at ~|frontier| inner solves instead of ~pop×gen — so it confirms/sharpens the *explored* frontier at a fraction of a full pass's exact cost. Covers every supported shape — binary 0/1 MILP and proportional QP/LP; each point is still optimal for its scalarization, so the auditor guarantee is unchanged. Covered by [`tests/test_certify_curated.py`](tests/test_certify_curated.py).

**Witness / feasibility audit (the feasibility-side auditor).** Where `certify` audits *optimality* of the frontier, `engine.optimizer.audit(problem, prop=…)` audits *feasibility / property* claims over the **whole feasible region** — every feasible plan, not just the non-dominated ones. It reuses the exact MILP encoding (`solvers.highs_backend._add_milp_constraints`, shared with `_solve_milp_highs` so the audit's feasible region is provably the space `solve` optimizes over) via an objective-free feasibility solve: `_audit_milp_highs` returns the **raw** HiGHS status, because the verdict turns on Infeasible (the property holds) staying distinct from a time limit (inconclusive). The property is expressed in Frontier's existing `Constraint` vocabulary — no new grammar — and `_negate_property` encodes its negation as the witness condition (cardinality negating to a two-branch disjunction). `INFEASIBLE` on the negation across the region ⇒ the guarantee holds for every feasible plan; a feasible point ⇒ a concrete counterexample witness (evaluated back to objective values via `score_slate`); a `holds` over an empty region is reported as `holds_vacuously`, never silently as a pass. Gated to **binary** shapes that pass `exact_solver_fits` (HiGHS only in v1; cuOpt parity and a continuous path are open); an unfit shape or missing backend **raises `ValueError` → a tool error**, the same hard-decline as `solve`'s exact gate (never a silent or structured degradation). Every verdict echoes the `feasible_region` it is conditional on, so a `holds` is self-certifying. Backend-agnostic orchestration is `solvers._scalarization.audit_milp` (injected `inner_audit`, the same split as `optimize_milp`'s `inner_milp`); the MCP payload (verdict framing + property + region echo) is `explorer.audit_property`, surfaced as **`explore audit`**. Covered by [`tests/test_audit.py`](tests/test_audit.py) (engine verdict matrix + wire-level stdio).

### Skill Auto-Injection

Tool responses include the relevant skill's **core** for the *next* workflow phase, so agents receive guidance at the right time without manually calling `get_skill()`. Cores are deliberately lean (always-apply judgment); each skill's deep, situational sections live in its `references/` and are fetched per section at the moment of need (`get_skill(name, section=…)`) — never injected, so a full workflow carries ~12k tokens of guidance instead of ~24k while every section stays one cheap call away.

| Trigger | Injected Skill |
|---------|---------------|
| MCP connect (server instructions) | Condensed `problem_framing` + constraint schemas + scores schema |
| `model/create` response | `data_collection` |
| `model/update` (objectives/options) | `data_collection` (if not already injected) |
| `model/update` (scores hit 100%) | `optimization_strategy` |
| `solve/validate` (ready=true) | `optimization_strategy` (if not already injected) |
| `solve/run` response | `solution_interpreter` (first solve per problem; re-armed by structural edits — score-only updates mark results stale but never re-arm) |

Re-injection of `optimization_strategy` only fires on problem *shape* changes (objectives or options). Refinements (constraints, interaction_matrices, scenarios, reference_points, approach flip, score updates) don't re-trigger – the methodology guidance hasn't changed.

### Guidance Persistence

Auto-injected skill text lands mid-transcript (inside tool results), so on a long session a coding agent's compaction can summarize it away and the web UI (§5) can push it past the context limit. Two pinned layers keep the right guidance reachable regardless:

- **Durable index.** The MCP `instructions` (folded into every host's system prompt, so compaction-surviving) carry a *GUIDANCE MAP*: each of the four skills, the phase it governs, which three auto-inject their full text, and an explicit instruction to **re-fetch via `get_skill(<name>)` when a skill's body is no longer in recent context**. The always-on map is the compaction survivor; the full skill text is the at-the-moment-of-need payload.
- **Decision-step pointers.** Every explore/decision action's response carries a `guidance_pointer` (`mcp_server/guidance.py`: `_attach_guidance_pointer` + the `_DECISION_GUIDANCE` map) naming the exact section that governs reading *that* output (e.g. `tradeoffs` → *Presentation Order*, `certify` → *Reading the Certificate*, `audit` → *Reading the Audit*, `sensitivity` → *Exact Sensitivity*, or *Binding Analysis* on the frontier-inferred fallback) plus the exact one-call fetch: `get_skill('solution_interpreter', section='<that heading>')`. Since the cores stopped inlining the deep sections, this pointer is the *primary* delivery for situational depth — a ~1k-token section fetch at the moment of need, far cheaper than re-fetching a whole skill — and the same call recovers it after compaction. Centralized as one map + helper so the convention stays uniform; error and navigation actions get no pointer (the intended exclusion set is recorded beside the map).

Together with auto-injection (fresh full text at each phase transition) and web-UI compaction (§5), the agent always either *has* the active guidance or *knows the one call to recover it* – without pinning every skill's full text forever (token-prohibitive).

### Visualization & Compaction Layers

**Built-in visualizations**: `explorer.py` renders ASCII/text visualizations inline with explore results, avoiding a separate rendering step:
- `_render_tradeoffs_viz` – scatter plot of objective pairs with labeled extremes
- `_render_parallel_coords` – parallel coordinates chart for compare and compare_curated
- `_render_scenario_viz` – robust vs scenario-specific option summary
- `_render_marginal_rates` – cost-per-unit bar chart with knee marker

**Structured viz payloads (`viz_data`)**: alongside the ASCII, each explore view also attaches a JSON `viz_data` block (built by `engine/viz.py`'s `_viz_data_*` builders) for hosts that can render charts. Six types: `scatter` (tradeoffs), `parallel_coords` (solutions / compare), `marginal_rates` (marginal_analysis – one per objective pair, under `pairs[]`), `scenario_summary` (scenario_results), `scenario_parcoords` (scenario_frontiers – per-scenario overlay with curated picks emphasized), and `formulation` (`model get` summary). Chat and coding-agent surfaces ignore the field and render the ASCII; the web UI (§5) consumes it and renders charts – Plotly for the frontier scatter/3D and the scenario overlay, D3 for the compare parallel-coords and marginal rates. The Python builders and the web UI's TypeScript types (`ui/lib/viz-data.ts`) are kept field-for-field in sync.

The `scatter` payload also carries **exact-certification** so the chart can denote it (built by `_scatter_exact_layer`, mirroring the `frontier_source` provenance + dominance logic): a `provenance` block (`kind` heuristic vs exact, `solver`, `exact_certified`) on every frontier, and – on a heuristic base-case frontier that has an `exact_run` overlay – an optional `exact_overlay` block (the certified `points` to draw on top, plus the heuristic `dominated_ids` the exact front strictly beats), with the objective ranges widened to keep a sharpened exact corner in-frame. The overlay is gated by the same scenario / run_id guards as `_frontier_provenance`, so it never attaches to a scenario frontier. `FrontierPlot.tsx` renders certification by **solidity** (certified points solid, the not-yet-certified field faded, dominated points faintest – opacity orthogonal to the role hue), with a provenance chip and a solid emerald-diamond overlay trace in the 2D/3D scatter; the ≥4-objective parallel-coordinates view can't overlay a second trace, so there certification rides on the chip plus greying the dominated lines. The `solution_interpreter` skill ("Denoting Certification – Prose & Tables") carries the same denotation into the agent's prose and tables.

**Response compaction**: `server.py/_format_explore` trims redundant bulk from explore output (e.g. strips option lists from extreme_solutions, restructures scenario_specific_options) to prevent MCP truncation while keeping the dict shape stable for consumers. `solve/run` responses are also compacted: full solution bodies are not returned inline – only objective ranges plus a preview (extremes + balanced by id/values), with a `next_steps` pointer to `explore` for full detail. The complete run payload (all solutions with allocations) is persisted to disk at `full_result_path` for bulk export and artifact assembly without token overhead.

### Long-running Solves (background jobs)

FastMCP runs a **synchronous** tool function *directly on the asyncio event loop* (`mcp.server.fastmcp` calls `fn(...)` for a sync tool, only `await`ing async ones). Frontier's tools are sync `def`s, so a long `optimize()` – a `thorough` NSGA run, an exact scalarization sweep, or a large problem – would block the whole server (no pings/keepalive, no concurrent calls) *and* overrun the client's per-tool-call timeout, killing the turn before the solve returns. Making the tools `async` is not an option: the function is the same object the test suite (~117 call sites) invokes synchronously, and `@mcp.tool()` returns it unchanged.

So `solve run` / `run_scenarios` **dispatch the heavy optimize+persist to a background worker thread** (`mcp_server/jobs.py`: a module-level `ThreadPoolExecutor` + a `{job_id → SolveJob}` registry; the dispatcher takes the solve as a `work_fn` closure) and wait inline only a short budget (`FRONTIER_SOLVE_WAIT_SECONDS`, ~10 s). If the solve finishes in time it returns the full result – fast solves stay synchronous, so existing behavior and tests are unchanged. Otherwise it returns a `{status:"running", job_id}` handle and the worker runs on; the agent polls **`solve status`** until `complete`. Validation and solver-resolution stay inline (immediate errors); only the optimize runs in the worker. The `solution_interpreter` injection moves to *delivery* time (the inline-complete return or the `status` return), always on a request thread, so the once-per-problem throttle stays single-threaded.

Two correctness properties make backgrounding safe:
- **Persisted by the worker.** The worker writes the run to the store before signalling done, so the frontier survives even an interrupted poll – `explore` works the moment `status` reports `complete`.
- **Safe persistence under concurrency.** `Store.save` is **atomic** (temp file + `os.replace`), so a reader never sees a torn file. Before attaching its run, the worker re-loads the problem **under a process write lock** and compares a **structural fingerprint** (objectives, options, scores, constraints, approach, interaction matrices, scenarios) against the dispatch snapshot: if the solve-inputs changed *during* the solve it reports `stale` and writes nothing rather than clobbering the edit (non-solve edits – curation, a rename – don't trip it, and the same gate covers the infeasible result so a mid-solve edit isn't reported as a stale "infeasible"). The lock serializes concurrent background solves on the *same* problem – the normal *run NSGA, then run an exact overlay* flow – so the two runs can't lose each other. The one residual window is a foreground edit committing inside the worker's brief locked finalize; it isn't reachable under today's single-client serial-call model, and the full hardening is to take the same lock in the foreground mutators.

Determinism is unaffected: the optimizer's seeded-RNG shim is thread-local (`optimize_scenarios` already runs solves off-thread the same way), so a seeded run reproduces identically whether it completes inline or in a worker. `time_limit` (above) composes with this – a long *bounded* run still backgrounds, then returns its `time_limited` best-so-far. *Deferred:* fine-grained live progress (gen *i/N*) and a `solve cancel` action (both would thread a progress/stop callback through the optimizer and both exact backends); coarse progress (`elapsed_s` + a phase `label`) ships today. A future option for multi-client/SSE deployments is to register an async tool-wrapper so even the brief inline wait never occupies the loop.

## 3. Problem State Lifecycle

The state machine that bridges user actions (diagram 1) to data mutations (diagram 4). Shows when results go stale, runs get archived, and curated solutions are tested for survival.

```mermaid
stateDiagram-v2
    [*] --> Empty: model/create

    Empty --> Framing: add objectives & options
    Framing --> Framing: update objectives/options

    Framing --> Scoring: first score added
    Scoring --> Scoring: update scores\n(merge semantics)

    Scoring --> Solvable: score matrix complete\n+ ≥2 objectives + ≥3 options
    Solvable --> Solvable: add/update constraints,\nreference points, scenarios\n(non-structural, no stale flag)

    Solvable --> Solved: solve/run\n→ NSGA-II/III produces Run
    Solvable --> ScenarioSolved: solve/run_scenarios\n→ per-scenario Runs
    Solvable --> Infeasible: solve/run\n→ no feasible solutions

    Infeasible --> Solvable: relax constraints\nor remove scenario constraint overrides

    ScenarioSolved --> Exploring: explore/scenario_results\n→ robust vs specific options,\nexpected values

    Solved --> Exploring: explore/* calls\n(tradeoffs, compare, dominance)
    Exploring --> Exploring: navigate frontier,\ncompare solutions,\nreference point analysis

    Exploring --> Curating: explore/curate\n→ snapshot solution with\ncontent_signature
    Curating --> Curating: curate more,\nrename, uncurate

    Curating --> Feedback: explore/feedback\n→ rating, notes, stage\n→ linked via content_signature\n→ attaches to curated solution

    state stale_fork <<fork>>
    Solved --> stale_fork: update objectives, options,\nscores, constraints, or approach
    ScenarioSolved --> stale_fork
    Exploring --> stale_fork
    Curating --> stale_fork
    Feedback --> stale_fork

    state stale_join <<join>>
    stale_fork --> stale_join

    stale_join --> Stale: results_stale = true\ncurrent Run archived to runs[]

    Stale --> Solved: re-solve\n→ new Run replaces current\n→ results_stale = false
    Stale --> ScenarioSolved: re-solve scenarios\n→ new ScenarioRun

    note right of Stale
        Curated solutions persist across re-runs.
        content_signature used to check if
        a curated pick survived in the new frontier
        (in_current_frontier flag).
        Feedback accumulates on curated solutions
        via content_signature – survives re-runs
        and problem variations.
    end note

    note right of Solved
        Run: solutions[], quality (hypervolume,
        spacing CV), constraints_snapshot,
        solver / exact (engine provenance).
        Each solution: selected_options,
        objective_values, allocations,
        content_signature (MD5).
    end note

    note left of ScenarioSolved
        ScenarioRun: dict of scenario_name → Run.
        Each scenario independently optimized
        with score overrides/adjustments
        and constraint overrides applied.
    end note
```

## 4. Data Flow

What data is user-provided, system-collected, and engine-generated at each phase.

```mermaid
flowchart LR
    subgraph PROVIDED["User-Provided Data"]
        direction TB
        P1["Problem framing<br/><i>name, domain, context,<br/>approach (binary/proportional)</i>"]
        P2["Objectives<br/><i>name, direction, unit,<br/>aggregation (sum/avg/min/max/quadratic)</i>"]
        P3["Options<br/><i>name, description</i>"]
        P4["Score matrix<br/><i>(option, objective) → value</i>"]
        P5["Constraints<br/><i>cardinality, force include/exclude,<br/>objective bounds, exclusion pairs,<br/>dependencies, group limits,<br/>max allocation (8 types)</i>"]
        P6["Reference points<br/><i>baseline (current portfolio +<br/>objective values) &<br/>aspirational (target outcomes)</i>"]
        P7["Scenarios<br/><i>name, probability, description,<br/>score overrides, score adjustments<br/>(multiply/add), constraint overrides</i>"]
        P8["Feedback<br/><i>rating (1-5), notes, stage,<br/>linked via content_signature<br/>(stable across runs)</i>"]
        P9["Curation<br/><i>solution picks, custom names, notes,<br/>accumulated feedback history</i>"]
    end

    subgraph COLLECTED["System-Collected Metrics"]
        direction TB
        C1["Framing metrics<br/><i>objective/option/constraint counts,<br/>directions completeness</i>"]
        C2["Data metrics<br/><i>score completeness %,<br/>missing scores list,<br/>variance by objective,<br/>dominated options</i>"]
        C3["Solve metrics<br/><i>solution count, hypervolume,<br/>spacing CV, objective variation,<br/>option coverage</i>"]
        C4["Outcome metrics<br/><i>selected solution, rating,<br/>feedback count</i>"]
        C5["Diagnostics<br/><i>zero solutions, clustering,<br/>low variation, never-selected options,<br/>binding constraints</i>"]
    end

    subgraph GENERATED["Engine-Generated Results"]
        direction TB
        G1["Pareto frontier<br/><i>set of non-dominated solutions<br/>(dominance filtering via NSGA-II/III)</i>"]
        G2["Solutions<br/><i>selected options, objective values,<br/>allocations (proportional),<br/>content signatures (MD5)</i>"]
        G3["Quality & diversity<br/><i>normalized hypervolume (coverage),<br/>spacing CV (spread uniformity)</i>"]
        G4["Tradeoff analysis<br/><i>correlations, extremes,<br/>balanced solution (min distance<br/>to ideal point),<br/>frontier shape per pair<br/>(linear/concave/convex/discontinuous),<br/>reference point comparisons</i>"]
        G5["Scenario results<br/><i>per-scenario frontiers,<br/>robust options (all scenarios),<br/>scenario-specific options,<br/>expected values (prob-weighted)</i>"]
        G6["Run history<br/><i>archived runs, constraint snapshots,<br/>frontier diffs, option coverage</i>"]
        G7["Infeasibility analysis<br/><i>binding constraints,<br/>relaxation suggestions</i>"]
        G8["Curation tracking<br/><i>in_current_frontier survival flag,<br/>cross-run signature matching,<br/>feedback history + avg rating</i>"]
    end

    P1 & P2 & P3 --> C1
    P4 --> C2
    P2 & P3 & P4 & P5 -->|"validate & optimize<br/>(NSGA-II/III)"| G1
    G1 --> G2
    G1 --> G3
    G2 --> G4
    P7 & P4 & P5 -->|"per-scenario optimization<br/>(adjustments → overrides applied)"| G5
    P6 --> G4
    G2 --> C3
    P8 --> C4
    P5 -->|"infeasible?"| G7
    P9 -->|"snapshot + signature"| G8
    G8 -.->|"survival check<br/>against new frontier"| G2

    C1 & C2 -->|"returned with<br/>model updates"| AGENT1([Agent])
    C3 & C5 -->|"returned with<br/>solve results"| AGENT1
    G4 & G5 & G6 & G8 -->|"returned with<br/>explore results"| AGENT1
```

## 5. Web UI & Hosting

An optional web app ([`ui/`](ui/)) is the **visualize** surface – a thin chat shell over the same MCP engine, for users not in a coding-agent MCP client. It adds no domain logic: workflow behavior still flows from the engine's tool descriptions, `instructions`, auto-injected skills, and the `viz_data` payloads above.

**Pluggable agent runtime**: `ui/lib/agent-runtime.ts` selects a backend via the `AGENT_BACKEND` env var. All adapters normalize to one Anthropic-shaped SSE wire format (`mcp_tool_use` / `mcp_tool_result` content blocks), so the UI is identical across them and provider lock-in is bounded to that single file.

| `AGENT_BACKEND` | Status | MCP loop | Engine reachability |
|---|---|---|---|
| `messages-api` (prod default) | working | Anthropic server-side MCP connector | needs a public https engine (Anthropic calls it) |
| `anthropic-local` (local default) | working | client-side, inside the web app | works with localhost or a gated engine |
| `openai-compatible` | working | client-side | any OpenAI Chat-Completions provider |

**Design invariants:**
- **Thin system prompt, behavior fetched from the engine** (`ui/lib/system-prompt.ts`) – no skill/workflow content is *hardcoded* in the UI. Each adapter fetches the engine's MCP `instructions` (the canonical workflow + framing guidance) and folds them into the system prompt: `anthropic-local` / `openai-compatible` get them from their MCP client, and `messages-api` fetches them over a short-lived authenticated client (cached) because the Anthropic connector does **not** surface the `instructions` field. The prompt file itself holds only identity + web-UI-specific *presentation* directives (e.g. render charts, don't echo the engine's ASCII visualizations) that by definition can't live in the shared engine. Domain/workflow fixes still belong in the engine, never the prompt.
- **Direct SSE, no AI SDK**: each `data: {…}` line is a raw Anthropic event; `ui/lib/stream-reducer.ts` folds them into message state incrementally. It addresses content blocks by the server-assigned `index` (which counts thinking blocks the UI doesn't store), never by array position – position-based addressing misroutes every delta after a thinking block, leaving empty text blocks the API rejects on the next turn (`text content blocks must be non-empty`). The same module strips internal fields and drops empty text blocks when building the round-trip payload.
- **Clean assistant rendering** (`ui/app/page.tsx`, `ui/components/Viz/`) – the chat surfaces only the model's prose, markdown tables, and charts drawn from each `tool_result`'s `viz_data` (frontier scatter/3D and the per-scenario overlay via Plotly, compare parallel-coords and marginal rates via D3, scenario summary and formulation as cards). Raw tool-call input/result JSON is **not** rendered, and the engine's ASCII visualizations are stripped if the model echoes them – the value-add over the coding-agent MCP surface is a clean, chart-first view.
- **Token-budget-aware history compaction** (`ui/lib/compaction.ts`) – the UI resends the full transcript every turn and, unlike a coding agent, does no compaction of its own, so a long session would otherwise grow until it hits the model's context window and hard-fails.

  When (system + history) approaches a budget, `compactHistory` **pins** the durable layers – the system prompt (durable index), the first user turn (decision anchor), and the last K messages – and drops the older middle turns behind one breadcrumb naming how many turns were elided and the active skill's `get_skill()` re-fetch path. Dropping whole turns is safe because each turn bundles its own `tool_use`+`tool_result`, and the breadcrumb (assistant) plus a user-turn tail boundary preserve the user/assistant alternation the APIs require. Budget knobs are env-tunable (`AGENT_CONTEXT_WINDOW` / `AGENT_OUTPUT_RESERVE` / `AGENT_KEEP_RECENT_MESSAGES`); defaults target the prod Anthropic path. This is the web-UI half of *Guidance Persistence* (§ Skill Auto-Injection): the session survives the context wall and keeps a breadcrumb back to the skills the durable index names.

  This client-side trim is the provider-agnostic **floor** (it also covers the `openai-compatible` adapter, which has no native equivalent) and bounds growth **between** turns – the transcript the UI resends. On the Anthropic backends it pairs with the engine's **native context management** (`anthropic-request.ts` `contextManagement()`): `clear_tool_uses` clears the oldest tool results – Frontier's large explore/solve JSON – server-side with a placeholder, or `compact` summarizes; *intended to* bound growth **within** a single agentic turn (the server-side connector loop), not only between turns. Env-gated: `AGENT_CONTEXT_MANAGEMENT=clear_tool_uses｜compact｜off` (default `clear_tool_uses`); the connector + context-management beta combination isn't yet confirmed end-to-end, so smoke-test before relying on the within-turn bound, and `off` falls back to the client-side floor.
- **Prompt caching** (`anthropic-request.ts` `cachedSystem` / `cachedTools` / `withCacheBreakpoint`) – the dominant cost lever. `cache_control` breakpoints mark the stable prefix – the system prompt, the four tool definitions (on the `mcp_toolset` entry), and the conversation prefix including the injected skill cores (~4.7k tokens for `solution_interpreter`; deep sections are fetched per-section on demand rather than injected) – so every later turn and every server-side tool-call iteration re-reads them at ~10% of input price instead of re-processing them in full.

  Caching itself **removes nothing**: skill guidance stays in context, just cheap. Context management *does* trim: `clear_tool_uses` clears old tool results, which on a long session can include the auto-injected skill cores (only the explicit `get_skill` re-fetch path is excluded). Guidance that ages out is recovered via the durable index's `get_skill` re-fetch (§ Skill Auto-Injection) – so skill content stays *available*: present and cheap while cached, recoverable once cleared. `AGENT_PROMPT_CACHE=off` disables; cache hit/write token counts are logged server-side for verification.
- **Ephemeral sessions**: a random `problem_id` per session (pre-D.1 multi-tenancy); refreshing the page starts a new one.

**Hosting & auth**: [`render.yaml`](render.yaml) provisions both services (engine + web UI) and is the OSS deploy path. There are **two independent shared-secret gates**. Each is open when its secret is unset *for local / loopback use*, but **fails closed once the service is actually exposed** — a misconfigured public deploy refuses to serve rather than run unauthenticated:

- **Engine gate**: a shared `FRONTIER_MCP_TOKEN` (auto-generated as a Render env group, injected into both services) gates the public engine: the web app forwards it on the connector call (`authorization_token`), and direct MCP-client users send it as an `Authorization: Bearer` header. Enforced by an ASGI bearer check in `mcp_server/server.py` over the `/sse`, `/messages`, and `/mcp` paths. With the token unset, an HTTP transport serves only on a loopback bind; bound to a routable interface (`MCP_HOST=0.0.0.0`) without a token, `main()` refuses to start.
- **Web UI gate**: a shared `UI_ACCESS_PASSWORD` gates the web app itself via HTTP Basic Auth (`ui/middleware.ts`), covering the page *and* the API routes. This is necessary because the engine token authenticates the *app→engine* call and is never exposed to UI users – so without its own gate the hosted UI would be open to anyone with the URL (and would spend the operator's `ANTHROPIC_API_KEY`). Unset, the gate is open in development but a **production** build (`NODE_ENV=production`) refuses to serve (503). Behind the password, a per-instance request rate limiter (`ui/lib/rate-limit.ts`) caps the chat/render request rate as a floor against runaway spend.

Per-user identity (`owner_id` scoping, real per-user accounts) is the next step (D.1); today every session shares one problem namespace, which is acceptable for a trusted closed beta.
