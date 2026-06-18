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

These two rows are the two spaces a decision lives in: **objective space** (the outcomes — what you get) and **decision space** (the selected options / allocations — what you actually do). Users judge in objective space but must act in decision space, and the map is many-to-one — different option mixes can reach nearly the same outcome. Keep both in view: whenever you present an outcome, name the actions that produce it. (See *Two Spaces: Objective and Decision* in `frontier://skills/problem_framing`; `explore composition` clusters in decision space to expose when different plans land at the same outcome.)

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

These modes map onto *when* preferences enter (see *Preference Timing* in `frontier://skills/problem_framing`): Exploring and Refining are a-posteriori and interactive elicitation — the user forms preferences by seeing the frontier; a user who set reference points before solving is expressing a-priori preferences and often jumps straight to Confirming. Don't ask for objective rankings before the user has tradeoffs to react to.

### Presentation Order: Extremes → Balanced → Inflection → Risk → Preference
1. Start with the extremes: "This solution maximizes revenue but has the highest effort. This one minimizes effort but sacrifices revenue."
2. Show the balanced middle: "This solution is the closest to ideal across all objectives."
3. Show inflection points (if present): anchor the marker with the actual objective values and the jump magnitude. Template: *"[A]-vs-[B] inflection at [A=value, B=value] — jump factor [X]×. Past this point, each extra unit of A costs roughly X× more B than before."* The `inflection_point_candidates` entries in `tradeoffs` output carry `objective_values` and `jump_factor` — use both. A named "stop here" with numbers is far more actionable than a directional sentence.
4. **If `scenarios_available` is set on the tradeoffs response**, layer in scenario_risk *before* asking for preference — the frontier you're showing is one possible future. Call `explore scenario_results` and weave in expected vs CVaR per objective, plus core/marginal option tiers. Skipping this step means the user picks under a single-future illusion when scenario data is on disk. Fetch *Scenario Results Presentation* for the framing patterns (`get_skill('solution_interpreter', section='Scenario Results Presentation')`).
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
- When two solutions are near-identical in objective space but differ in which options they use, that decision-space difference is the deciding, low-regret lever — surface it and let the user choose on implementability (`explore composition` clusters make these look-alike-but-built-differently families explicit).

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

## Reference Sections (fetch on demand)

The situational presentation playbooks live in `references/presentation-refinements.md` — part of this skill, fetched per section with `get_skill('solution_interpreter', section='<heading>')`. Explore responses name the exact section in their `guidance_pointer`; fetch it before presenting that output. The Core Judgment above is always-apply; these are the at-the-moment depth:

Frontier Quality and Completeness Signals · Solution Quality Ladder · Root Cause Taxonomy · Frontier Shape Interpretation · Binding Analysis · Exact Sensitivity — Shadow Prices & Reduced Costs (solver duals) · Reading the Certificate (explore certify) · Denoting Certification — Prose & Tables · Reading the Audit (explore audit) · Marginal Analysis Interpretation · Frontier Visualization · Mining the Solution Set · Run Diff Interpretation · Reference Point Narration · Scenario Results Presentation · Sensitivity Intuition · Diagnostic Patterns · Recommendation Calibration · Preference Learning · Common Misconceptions · Solution Curation · Stakeholder Writeup & the Why-Triplet

(`references/explore-diagnostics.md` additionally holds the raw field schemas for the tradeoffs and scenario result blocks.)

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
