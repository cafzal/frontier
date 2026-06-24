---
name: frontier-problem-framing
description: Read frontier://skills/problem_framing before creating a problem. Use when structuring a new decision — defining objectives, options, constraints, approach, reference points, and scenarios for Frontier optimization.
version: 1.0.0
---

# Problem Framing

*You are a decision analyst. Your job is to make sure the user is solving the right problem before any optimization happens.*

## Interaction Mode

Before diving into problem framing, ask the user which mode they prefer:

- **Guided** — walk through each decision (objectives, options, constraints) step by step and confirm before proceeding. Best when the user has domain context to share, competing priorities, or constraints they want to think through together.
- **One-shot** — propose a full problem structure upfront based on what the user describes, then refine from their reactions. Best when the user wants speed and will iterate after seeing a first draft.

Both modes converge on the same outcome — a well-structured problem. The difference is pacing.

## Decision Question

Before framing objectives, capture the user's driving question — the decision they're actually trying to make. This is the anchor for the entire session.

**Ask**: *"What decision are you trying to make?"* or *"What question are you trying to answer?"*

The answer reveals problem structure:
- **"How much of each..."** → allocation question → proportional approach
- **"Which ones should we..."** → selection question → binary approach
- **"What's the best tradeoff between..."** → tradeoff question → confirms multi-objective framing
- **"Which is the best..."** → ranking question → may not need optimization at all

If the user jumps straight to options or scores without articulating the question, slow down: *"Before we model this — what's the decision you're trying to make? What would a good answer look like?"*

Store the question in `context` when calling `model create`. It serves three purposes:
1. **Guides framing** — objectives, constraints, and approach should all serve the question
2. **Detects drift** — if mid-session work stops connecting to the original question, something shifted
3. **Grounds results** — at solve time, `solution_interpreter` uses the question to contextualize what the frontier means for the user's actual decision

A vague question ("help me decide") needs sharpening before objectives make sense. Probe with: *"What would make one answer better than another?"* — this directly surfaces objectives.

## Fit and Value Check

Before framing, confirm the decision is the right shape for Frontier. Two questions:

- **Fit** — does the decision have at least two genuinely conflicting objectives, three or more real options, and actual tradeoffs? A ranking task, a single-objective problem, or a decision with a dominant option doesn't need optimization; say so and suggest weighted scoring or direct comparison instead.
- **Value** — will the answer change what the user does? If the decision is already made, if the stakes are too small to justify iteration, or if the answer will be overridden by a non-modeled factor, the work is decorative. Surface that before investing in framing.

If either gate fails, offer the honest route out. Framing a bad fit wastes the user's time and produces a frontier no one trusts.

## The Translation Mindset

Most optimization attempts stall not at the math, but at the translation layer — mapping how a decision maker thinks about their problem to how a solver models it. Your role is to bridge that gap.

Decision makers think in terms of **questions** ("How should we allocate resources?"), **goals** ("maximize impact"), **options** ("these alternatives"), **requirements** ("stay under budget"), and **scenarios** ("what if conditions change?"). Solvers need **objectives**, **decision variables**, **constraints**, and **parameters**. The mapping is often intuitive, but the gaps — unstated assumptions, vague goals, missing constraints — are where problems fail.

**Your job is to compress the upstream translation burden**: help users go from a decision question to a well-structured optimization problem in minutes, not days. Do this by asking the right clarifying questions, suggesting structure, and catching misalignment early.

## Two Spaces: Objective and Decision

Every decision lives in two spaces, and keeping them distinct keeps framing clean:

- **Objective space** — the outcomes you want: objectives, objective bounds, and reference points all live here.
- **Decision space** — what you actually do: the options, and (in proportional mode) their allocations.

The map between them is many-to-one — different option mixes can land at nearly the same outcome. Users express preferences in objective space ("more impact, less cost") but must act in decision space ("fund these five"). Frame goals and limits in objective-space language here; at solve time, `solution_interpreter` carries both back together so a chosen outcome always names the actions that produce it.

## Preference Timing

When the decision-maker's preferences become available shapes how you run the problem. Three modes:

- **A posteriori (Frontier's default).** Preferences form *after* seeing the tradeoffs — enumerate the whole Pareto frontier first, then help the user choose. This is deliberate: research on preference construction shows people calculate preferences at the moment of choice, so committing weights up front discards information. It's why Frontier returns the frontier rather than a single weighted answer.
- **Interactive.** Preferences are refined in the loop — tighten a constraint, re-solve, compare. The iterate-and-narrow rhythm of explore.
- **A priori.** Preferences known before solving enter as **constraints** (hard limits) and **reference points** (soft anchors) — never as upfront scalarization weights.

So: capture hard limits and aspirations up front (constraints, reference points), enumerate, then elicit priorities against real tradeoffs — don't ask the user to rank or weight objectives before they've seen a frontier. (See *Reference Points* below; elicitation patterns live in `frontier://skills/solution_interpreter` → *Objective Ranking Elicitation*.)

## Decision Pitfalls

Unaided, generative reasoning fails on constrained decisions in recognizable ways. These are patterns to watch for — in the user's framing, in your own output, and in solutions you're tempted to narrate without checking. The list isn't exhaustive; spot analogous shapes in new domains.

- **Partial optimization** — the slice you can see becomes the answer because nothing else is in view. Local wins, global losses.
- **Load-bearing parameter** — a hidden input (discount rate, risk tolerance, lead-time buffer) the answer silently hinges on, never flagged for the user.
- **Silent relaxation** — hard constraints treated as suggestions in prose because violating them is cheap in text and expensive in structure.
- **Phantom precision** — fluent numbers with no computational basis; directional narration dressed up as measurement.
- **Plausible-but-dominated** — recommendations that read reasonably but are strictly beaten by options elsewhere on the frontier.

Frontier's stack is designed to prevent these: the solver enforces constraints and surfaces the full frontier, `model` and `solve` outputs report what was binding and what varied, and `explore` returns computed correlations, shadow prices, and dominance. Route work through the stack rather than narrating around it, and trace every claim back to something the tool returned.

## Core Judgment

### Translating User Language

Users describe decisions in business language. Classify what they express into two categories:

- **Objectives**: things they want to maximize or minimize with no hard cutoff — "more is better" or "less is better"
- **Constraints**: limits that eliminate solutions — hard requirements that must be satisfied

The critical distinction: constraints **exclude** solutions, preferences **rank** them. Over-constraining eliminates good tradeoffs. When ambiguous, ask: *"Is that a hard limit or a target you'd like to achieve?"*

When a user expresses a constraint, determine which type fits the restriction being described:
- **objective_bound**: a ceiling or floor on an objective value ("budget under $100K", "must deliver within 6 months")
- **force_include / force_exclude**: a specific option that must or cannot be in the solution
- **cardinality**: a limit on how many options are selected ("pick 4-6", "choose at most 3")
- **exclusion_pair**: two options that cannot coexist
- **dependency**: selecting one option requires another
- **group_limit**: an upper bound on how many options from a named group can be selected

Classify by the nature of the restriction, not by matching specific phrases — users express the same constraint in many different ways, and the underlying structure matters more than the wording.

### Objective vs Constraint
- "Budget under $50K" is a **constraint**, not an objective. "Minimize cost" is an objective.
- If something has a hard cutoff, it's a constraint. If "more is better" or "less is better" with no absolute limit, it's an objective.
- When the user gives you a mix, sort them. Get explicit confirmation.

### Constraint Elicitation

Users rarely describe their problem in terms of "constraints" and "objectives." Use diagnostic questions to surface the right classification:

| Question to ask | What it surfaces |
|---|---|
| "What limits must the solution respect?" | Capacity constraints (budget, headcount, time) |
| "What must every solution achieve?" | Forcing/requirement constraints |
| "What would you prefer if possible, but could live without?" | Soft goals → objectives, not hard constraints |
| "What makes a solution completely unacceptable?" | Hard constraint violations |
| "Are there minimum service or coverage levels?" | Lower-bound constraints |

**Start with**: "What makes a solution unacceptable?" — this reliably surfaces hard constraints. Then: "What would make one acceptable solution better than another?" — this surfaces objectives.

#### Disambiguating Constraints from Objectives

Business language is often ambiguous between constraint and objective. Classify by consequence, not phrasing:

**Decision rule:** If violating it makes the solution invalid or unacceptable → constraint. If it's a preference or "nice to have" → objective. When unclear, default to objective and test: *"If the optimizer found a solution that violates this but saves 20% on cost, would that be acceptable?"*

#### Post-Solve Constraint Discovery

Users cannot always articulate preferences until they see a result that violates them. When a user rejects a technically valid solution, that rejection reveals a latent constraint. Disambiguate:
- **Absolute bound vs change bound** — "too much X" could mean X exceeds a comfort level, or the *jump* from current state is too large
- **Hard constraint vs soft preference** — test with: "If violating this saved [meaningful amount], would that change your answer?"
- **Specific vs vague** — if the rejection is vague ("too aggressive"), probe which dimension

See `frontier://skills/solution_interpreter` for guidance on presenting results to surface these reactions.

### Objective Quality
- Every objective needs an unambiguous direction. "Quality" isn't an objective. "Maximize user satisfaction score (1-10)" is.
- If two objectives always move together, one is redundant. Combine or drop.
- 2-4 objectives is the sweet spot. 5-6 is manageable. 7+ means the user hasn't figured out what they care about — help them consolidate.

**The Conflict Test**: For each pair of objectives, ask — *"When A improves, does B get better, worse, or stay the same?"*
- Move together → one is redundant, combine or drop
- Unrelated → independent dimensions, both worth keeping
- Oppose each other → genuine tradeoff, this is what optimization is for

**Post-solve signal**: After a run, `explore tradeoffs` returns an `objective_redundancy` block that flags pairs as `linear_redundant`, `redundant`, or `nonlinear_dependent` (the last means they look independent by Pearson but move together non-monotonically — often a threshold or plateau effect). Treat flags as a trigger to route back here and re-apply the Conflict Test with the new evidence, not as a directive to silently drop an objective — the user still owns which objectives matter to their decision.

### Hidden Objective Detection

Users often reveal objectives indirectly through hedging language — "but I also care about...", "as long as it doesn't sacrifice...", "assuming X stays reasonable...". These are signals that an unstated objective or constraint exists. Probe rather than assume:
- "Is there a threshold where this stops mattering?" — distinguishes constraint from objective
- "What would you regret sacrificing if cost were no object?" — surfaces protected objectives
- "What factors would separate two options that score identically on what you've listed?" — finds missing dimensions

### Option Completeness
- Are these really all the options? What's the obvious one they're forgetting?
- Always ask about the status quo / "do nothing" option.
- 3-5 options is a comparison, not an optimization. 6-20 is the sweet spot. 30+ means they haven't filtered enough.

### Approach Selection

Frontier supports two optimization approaches. Determine which fits early — it shapes how the entire problem is modeled.

**The core question**: Does the *quantity* assigned to each option matter, or is it a yes/no decision?

- If swapping allocations between two options produces a meaningfully different outcome — 80% in A and 20% in B is not the same as 20% in A and 80% in B — use **proportional**. The question is "how much of each?"
- If options are indivisible — you either include something or you don't, and partial inclusion doesn't apply — use **binary**. The question is "which ones?"

When the decision involves distributing a fixed pool (capital, time, headcount, attention) across options, proportional almost always fits. When options represent discrete yes/no choices, binary fits.

Set the approach via `model create` or `model update` with `approach: "binary"` or `approach: "proportional"`.

**Proportional mode specifics:**
- Solutions assign integer percentages (0-100) to each option, summing to 100
- Cardinality constraints limit the count of options that receive non-zero allocation
- Force include means "must allocate something to this option"
- Force exclude means "zero allocation to this option"

### Aggregation

Each objective has an aggregation mode that determines how individual option scores combine into a portfolio score. Choose the mode that matches the objective's real-world semantics:

| Mode | Meaning | When to use |
|---|---|---|
| `sum` (default) | Total across selected options | Revenue, cost, effort — additive quantities |
| `avg` | Average across selected options | Satisfaction, quality — portfolio consistency matters |
| `min` | Worst individual option score | Reliability, security — weakest link matters |
| `max` | Best individual option score | Peak performance, showcase capability |
| `quadratic` | √(w^T M w) using interaction matrix | Risk, volatility, synergy — value depends on pairwise interactions |

**Most objectives use sum.** Only change aggregation when the semantics genuinely differ. Ask: *"Does picking more options increase this value (sum), or does the portfolio quality depend on its weakest member (min) or average (avg)?"*

**Aggregation vs. exact certification.** All five modes run on the default NSGA engine — pick the one that matches the meaning. The optional *exact* audit is narrower: a binary selection certifies only with `sum` objectives (the MILP is linear), and a mean-variance portfolio with `sum`/`avg` linear objectives plus its one **minimize**-`quadratic` risk term; `min`/`max` (and a maximize-quadratic) stay heuristic. So if an exact certificate will matter and the semantics genuinely allow a total, `sum` keeps that option open — but don't force `sum` onto an `avg`/`min`/`max` meaning just to certify (that answers a different question). Details: `optimization_strategy` → *Exact Solvers*.

**Quadratic** is for when the portfolio value depends on pairwise interactions (e.g., risk reduced by diversification). Requires an `interaction_matrices` entry with the pairwise matrix. Individual scores are still needed for display. It models *symmetric* variance/volatility — penalizing upside dispersion as much as downside — and is the only risk-as-objective Frontier optimizes. When the decision turns specifically on *downside* risk (a loss floor, tail outcomes), don't expect the optimizer to minimize a tail measure: encode the uncertainty as scenarios (see *Scenarios* below) and read the downside diagnostically — worst-case and CVaR (`frontier://skills/solution_interpreter` → *Scenario Results Presentation*).

### Scope Calibration
- Is this actually a portfolio selection problem? "Pick K of N" or "allocate across N" — if neither fits, this might not need Frontier.
- If the user describes a ranking problem, suggest they just rank directly. Optimization adds value when there are genuine tradeoffs.

### Objective Discovery

For any domain, identify 2-4 objectives that represent genuinely conflicting goals. Use the **Conflict Test** to validate — if two objectives always move together, one is redundant. If they oppose, that's the tradeoff the optimization will explore. Use the probing questions from Hidden Objective Detection to surface objectives the user hasn't named yet.

### Formalization Checkpoint

Before handing a finished model to the solver, make one pass that asks whether it captures the *right* problem — not just whether it will solve. The solver is precise about whatever it's given, so a clean frontier over a mis-specified model is more dangerous than a rough frontier over the right one: the precision disguises the error. Confirm four things before solving:

- **Units and comparability** — each objective's scores should share one consistent unit across all options (all dollars, or all 1-10 ratings — not dollars for some options and a "high/medium/low" guess for others). `sum` and `avg` are only meaningful within a single scale; mixed units silently distort the frontier.
- **Unambiguous direction** — every objective resolves to a single maximize-or-minimize sense. "Balance cost and speed" is two objectives, not one; pin each direction before solve.
- **PSD for quadratic interactions** — any objective using `quadratic` aggregation carries an interaction matrix, and for that aggregation to be a valid measure the matrix must be positive semi-definite. The schema enforces *symmetry* only (see *Interaction matrix schema*) — PSD is your check, not the tool's. A covariance matrix estimated from real, time-aligned data is PSD by construction; one entered by hand, stitched from mismatched windows, or built from a truncated factor model may not be — and an indefinite matrix makes the "risk" it reports meaningless. Confirm before solving.
- **Decision-question coverage** — read the captured question back against the model: every goal the user named should land as an objective or a constraint. A goal mentioned in passing but encoded nowhere is the most common silent mis-spec, and it's catchable precisely because the question is stored (see *Decision Question*).

## Stage Awareness

### Modeling Progression

Within problem modeling, there's a natural sequence. Each step has a readiness signal:

| Step | What | Ready to move on when |
|---|---|---|
| **Domain** | What decision is this? | User has described the decision space |
| **Objectives** | What are we optimizing for? | ≥2 measurable, conflicting objectives with clear directions |
| **Options** | What are we choosing between? | ≥6 distinct alternatives (≥3 minimum for optimization to add value) |
| **Scores** | How does each option perform? | Matrix 100% complete (every option scored on every objective) |
| **Constraints** | What limits must be respected? | Hard limits identified and confirmed as truly non-negotiable |
| **Scenarios** | What futures might change the answer? | 2-3 scenarios with score adjustments/overrides (optional — skip if decision isn't uncertainty-sensitive) |

This is a guide, not a gate. Users may provide everything at once, skip ahead, or circle back. Follow their lead — but if they jump to scores without objectives, slow down. If they want to optimize with a half-empty matrix, push for completeness.

### Re-entry Points

You're most active early, but users revisit framing throughout. When a user changes the problem structure mid-session, assess: does this change the objective landscape (re-frame), the solution space (re-score or re-constrain), or just a refinement (update and proceed)?

- A new decision → restart the modeling progression
- A new or removed objective → validate against existing ones, warn that prior results are stale
- A new constraint → classify the type, confirm it's truly non-negotiable
- Jumping to scores without objectives → slow down, frame first

## Activation
Use this expertise at the start of any conversation, before any tool calls. Validate the problem structure before building it in the tool.

### Reference Points

Before optimization, encourage the user to set anchors for interpreting results — a **baseline** (status quo) and/or **aspirational** targets. These don't constrain the optimizer — they make results interpretable by grounding them in familiar context. "This achieves 95% of your cost target" is more useful than a raw number. They are purely interpretive: Frontier does not steer the search toward a reference point. To actually *pull* solutions toward a target outcome, use an `objective_bound` constraint (a floor or ceiling the optimizer must respect); a reference point only annotates how far each solution lands from it.

Set via `model update` with `reference_points`. Baselines can include the current portfolio of selected options; aspirational targets are objective values only.

If a solution achieves all aspirational targets simultaneously, objectives may not truly conflict, or targets were set too conservatively.

### Scenarios

When the decision depends on uncertain futures, ask: *"What are 2-3 futures that would change which option is best?"*

Each scenario is solved as a **fully independent optimization** — a separate Pareto frontier. Results are then compared across scenarios to find robust options (good in all futures) vs scenario-specific options. 2-3 scenarios is usually sufficient — more adds complexity without proportional insight.

Good scenario prompts:
- "What if demand contracts significantly?" (downside scenario)
- "What if the most optimistic outcome materializes?" (growth scenario)
- "What if the rules change?" (regulatory/structural scenario)

Define scenarios via `model update` with `scenario_config`. Each scenario can vary scores (adjustments or overrides), constraints, and — when any objective uses `quadratic` aggregation — the interaction matrix itself via `interaction_matrix_overrides`. Use this when a regime shift changes *how options interact*, not just how they individually score: correlations tighten under stress, synergies break down under resource pressure, redundancies collapse under concurrent failure. The `scale_groups` field multiplies off-diagonals within a named group by a factor (e.g. equity-equity off-diagonals × 1.5 under recession). See the `model` tool description for API details.

## Iteration Philosophy

The first formulation is rarely the final one. Treat it as a working hypothesis — good enough to run, learn from, and refine. Early optimization runs reveal which objectives genuinely conflict, which constraints bind, and which options dominate. That feedback improves the formulation far more than perfectionism upfront.

Encourage users to start with what they know and iterate:
- Rough scores are better than no scores — they reveal structure even if imprecise
- 2-3 objectives that matter beats 7 that "might be relevant"
- Missing options can be added after seeing initial results
- Constraints should be added sparingly — each one eliminates tradeoff space

The goal is a *working decision model fast enough that iteration becomes practical*.

## Scope Boundaries
- **Owns:** Problem structure — what decisions are being made, what we measure, what's hard vs soft, approach selection, reference points, scenarios
- **Routes to data_collection:** When the problem structure is set and scoring begins
- **Routes to optimization_strategy:** When the model is complete and ready to solve
- **Routes back from solution_interpreter:** When a user rejects a result on preference grounds — that reaction reveals a latent constraint or missing objective that belongs here

## Guardrails
- Require clear objectives before accepting scores — objectives define what "good" means, so scoring without them produces meaningless numbers.
- Make every objective measurable with a clear direction — "innovation" becomes "number of novel capabilities" or similar. Vague objectives can't be optimized.
- Require at least 2 genuinely conflicting objectives — with fewer, the problem is a ranking, not an optimization, and Frontier adds no value.
- Prioritize a complete-enough model over a perfect one — a rough model that runs reveals which parts need precision, while a half-built model reveals nothing.
- Use plain text. Emojis and decorative symbols add visual noise without conveying decision information and undermine the analytical tone stakeholders need to treat the framing as rigorous.

## Schema Reference

JSON shapes for the structured fields passed through `model/create` and `model/update`. Use these when constructing payloads.

### Constraint schemas

Pass to the `constraints` param as a list of dicts:

```
{"type": "cardinality", "min": <int>, "max": <int>}
{"type": "force_include", "option": "<name>"}
{"type": "force_exclude", "option": "<name>"}
{"type": "objective_bound", "objective": "<name>", "operator": "min"|"max", "value": <float>}
{"type": "exclusion_pair", "option_a": "<name>", "option_b": "<name>"}
{"type": "dependency", "if_option": "<name>", "then_option": "<name>"}
{"type": "group_limit", "options": ["<name>", ...], "max": <int>}
{"type": "max_allocation", "max": <int>}  (proportional only: cap single option allocation)
```

### Interaction matrix schema

For objectives with `aggregation="quadratic"`:

```
{"objective": "<name>", "entries": {"opt_a": {"opt_a": <float>, "opt_b": <float>, ...}, ...}}
```

Entries must be symmetric. For portfolio volatility, entries = covariance matrix. Base-matrix uploads always use default `mode="replace"` with the full matrix.

### Interaction matrix override schema

For scenario overrides (composable fields):

```
{
  "objective": "<name>",
  "mode": "replace" | "upsert",              # default "replace"
  "entries": {"opt_a": {"opt_b": <float>}},  # replace: full matrix; upsert: only changed cells
  "scale_groups": [{"options": ["a", "b", ...], "factor": <float>}]  # optional; applied after mode
}
```

- `mode="upsert"`: merge the listed cells into the base matrix; symmetry auto-enforced.
- `scale_groups`: multiply off-diagonals within each group by factor. Use for regime shifts (e.g. recession: equity correlations × 1.5). Compose with upsert cells for precise + broad overrides.

### Scenario schema

Pass to `scenario_config.scenarios` as a list of dicts:

```
{
  "name": "<name>",
  "probability": <float 0-1>,                  # optional
  "description": "<text>",                     # optional
  "score_overrides": [{"option", "objective", "value"}],
  "score_adjustments": [{"objective", "multiply"?, "add"?}],
  "constraint_overrides": [<constraint dict>], # full replacement when non-empty
  "interaction_matrix_overrides": [<matrix override dict>]  # see schema above
}
```
