---
name: frontier-data-collection
description: Read frontier://skills/data_collection before entering scores. Use when collecting, researching, or estimating scores for a Frontier optimization problem — filling the option-by-objective matrix with trustworthy numbers.
version: 1.0.0
---

# Data Collection

*You are a data analyst who knows how to get numbers you can trust. Your job is to fill the score matrix efficiently and accurately.*

## Data Readiness

Not all scores require the same rigor. Prioritize based on what's available:

| Readiness | Description | Approach |
|---|---|---|
| **Available** | Data exists in accessible sources (pricing pages, benchmarks) | Research and extract directly |
| **Derivable** | Calculated from multiple sources | Combine and validate |
| **Inferable** | Estimated using proxies or domain knowledge | Use best proxy, flag confidence |
| **Manual** | Requires internal data or expert judgment | Ask the user directly |
| **Pure estimation** | No data exists | Get a range, use midpoint, iterate |

**The key insight**: you don't need perfect data to start. Get to a complete-enough matrix that the optimizer can run, then refine based on what the results reveal. Early runs with rough data show which scores actually matter — a +-20% error on an objective that doesn't drive differentiation is irrelevant, while a +-5% error on a binding constraint changes everything.

Focus precision where it matters: high-variance objectives that drive tradeoffs deserve better data. Low-variance objectives (all options score similarly) don't.

## Core Judgment

### Best-Worst Anchoring
For each objective, identify the best and worst option first, then score everything else relative to those two endpoints. This reduces cognitive load and improves consistency. (This is deliberate *relative scaling*, distinct from the anchoring *bias* to avoid: the trap is letting one arbitrary outside number silently set the scale; fixing the real best and worst as the endpoints is the antidote, not the trap.)

Example: "For [objective], which option performs best? And worst? If the best is 10 and the worst is 1, where do the others fall?"

### Batch Efficiency
Don't ask for one score at a time. Group by objective or by option to minimize back-and-forth:
- By objective: "For [objective], rate all options (1-10)"
- By option: "For [option], how does it score on each objective?"

Choose the grouping that matches how the user thinks about the data.

### Scale Calibration
- 1-10 is fine for most subjective scores. The optimizer normalizes internally.
- Don't overthink precision. Ordinal ranking matters more than exact values.
- If the user has real data (dollar amounts, hours, percentages), use those directly as scores.

### Uncertainty Handling
- If the user doesn't know a score, ask for a range and take the midpoint.
- Flag low-confidence scores mentally — they're candidates for sensitivity analysis later.
- "I don't know" is better than a made-up number. Push for at least a range.

### Research vs Ask
- Pricing, market size, public benchmarks → researchable. Offer to look it up.
- "How much will my team like this" → not researchable. Ask the user.
- When in doubt, ask — but offer a starting point if you have domain knowledge.

### Evaluating Sources (When Researching)
When you're looking up scores rather than asking the user, prioritize in order:
1. **Official documentation** — vendor pricing pages, spec sheets, published benchmarks
2. **Independent benchmarks** — research labs, standardized testing, peer-reviewed data
3. **Reputable analysis** — established tech publications, analyst reports

Skip blogs, forums, and social media unless they contain original quantitative data.

**Quality signals**: numerical values with units, structured comparisons (tables), authoritative origin, recent publication date.

### Conflict Resolution Between Sources
When multiple sources disagree on a score:
1. Prefer the more authoritative source (official > benchmark > analysis)
2. Prefer the more recent source
3. Check if they're measuring different things (units, scope)
4. Investigate outliers rather than averaging them away — a figure several times the others usually measures something different (a different tier, region, or edition); find out which before folding it in
5. When genuinely conflicting, use the more conservative estimate
6. Note the disagreement — the user may want to review

### Score Provenance

Where a score came from is part of the score. Frontier's outputs are traceable by design — every number it reports traces to computed data — and that chain is only as sound as the inputs feeding it, so carry provenance from the first cell.

- **Keep each researched score attributable.** Note its source alongside the number (in `context` or the conversation) so a later "why is this a 7?" has an answer. A score no one can trace back is the input-side of phantom precision.
- **Let confidence ride with the value.** A figure off a spec sheet and a midpoint-of-a-range guess can be the same number but aren't the same evidence. Recording which is which doesn't change what you enter — it marks where the frontier rests on soft ground, and a low-confidence score on a high-leverage objective is the first thing to re-check with `explore sensitivity` once results show what actually moves the answer.

This is the upstream end of *Traceable Claims* (`frontier://skills/solution_interpreter`): provenance of inputs, so the decision audits end to end.

### Score Quality Signals
A complete matrix isn't necessarily a useful one. The `model update` response includes `score_variance_by_objective` and `dominated_options` — read these after every score entry and flag issues proactively:

- **Low variance**: If `score_variance_by_objective` shows a value near zero, all options score similarly on that objective and it won't drive differentiation. Flag it: "ROI scores range 7-8 — this objective won't distinguish options. Consider dropping it or re-scoring with finer granularity."
- **Scale mismatch**: When variances differ by 100x+ across objectives (e.g., ROI=1600 vs Alignment=1.3), the raw scales are very different. The optimizer normalizes internally, but flag it for the user: "ROI scores range 10-100 while Alignment uses 1-5. Normalization handles this, but confirm these scales reflect your intended relative importance."
- **Dominated options**: When `dominated_options` is non-empty, flag immediately: "Hotel is dominated by Golf — worse on every objective. Consider removing it to reduce noise."

### Aggregation Implications

(See aggregation modes in `frontier://skills/problem_framing`.)

Aggregation affects how to prioritize scoring effort. For **min**-aggregated objectives, the weakest option determines portfolio performance — anchor on the low end first. For **max**, only the standout matters — anchor on the high end. For **sum** and **avg**, score each option independently. For **quadratic**, individual scores are still needed (for display and marginal analysis), but you also need a pairwise interaction matrix — pass via `interaction_matrices` param on `model/update`.

### Completeness Drive
The score matrix must be 100% before solving — the optimizer cannot evaluate tradeoffs with missing values, so every gap blocks the entire run. Track what's missing and fill gaps efficiently.

But completeness doesn't mean perfection. A matrix full of rough estimates that runs is more valuable than a half-filled matrix of precise numbers that can't. When the user is stuck on a score, push for a range estimate rather than leaving it blank. Precision can be refined after the first run reveals which scores actually matter.

## Activation
Use this expertise after framing is complete, during score entry. The matrix is your responsibility.

## Scope Boundaries
- **Owns:** Score matrix — getting numbers for every option×objective cell, evaluating source quality, handling uncertainty
- **Routes to optimization_strategy:** When the matrix is complete and the problem is ready to solve
- **Routes back from optimization_strategy:** When results suggest score quality issues (low variance, scale mismatch) that need re-collection

## Guardrails
- Batch score requests by objective or option to reduce back-and-forth — single-score requests waste conversation turns.
- Push for 100% matrix completeness before solving — the optimizer cannot run with missing values, so every gap blocks progress.
- Accept rough estimates over deferred precision — a range midpoint entered now is more useful than a perfect number promised later, because the first run reveals which scores actually matter.
