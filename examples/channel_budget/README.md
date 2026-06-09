# Marketing channel budget

Split a media budget across 22 channel × audience/geo combinations balancing four conflicting goals: Conversions, Reach, ROAS, and Brand Lift. Direct-response channels convert and return well but reach few people; broad upper-funnel channels reach and build brand but convert poorly, and same-audience channels overlap, so reach combines sub-additively. Too combinatorial and interaction-laden for a spreadsheet or an LLM to allocate by hand.

- **`problem.json`**: 4 objectives (Conversions / ROAS / Brand Lift averaged, Reach quadratic via an audience-overlap matrix), proportional approach, constraints (no channel >15%, ≤1 line item per platform, blended ROAS ≥2.0x), and two scenarios (`signal_loss`, `demand_pullback`).
- **`scores.json`**: the 22 channels with per-channel scores, plus the Reach audience-overlap interaction matrix (negative off-diagonals between same-audience channels = diminishing combined reach).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="channel_budget"`, or paste this to an agent connected to Frontier:

> Allocate my marketing budget across the 22 channels in scores.json to maximize Conversions, Reach, ROAS, and Brand Lift. Treat Reach with the audience-overlap matrix, so same-audience channels combine sub-additively. Constraints: no channel over 15%, at most one line item per platform, blended ROAS at least 2.0x. Show the tradeoff frontier and where the knees are, and how it shifts under measurement signal loss and a demand pullback.

## The workflow

1. **Solve** (`solve run`, plus `solve run_scenarios` for the macro regimes): the optimizer produces the Conversions/Reach/ROAS/Brand-Lift frontier and a per-scenario frontier for `signal_loss` (measurement erosion cuts attributable conversions) and `demand_pullback` (a downturn softens response and efficiency).
2. **Explore the tradeoffs** (`explore tradeoffs`): where each mix leans efficiency versus reach versus brand, the knees, and how the frontier shifts across the scenarios.
3. **Certify and examine** (`explore sensitivity`): maximizing Reach is a non-convex quadratic, so the exact backend declines this shape: there is no exact certificate, and the examine reports the frontier-inferred binding analysis (which caps and the ROAS floor bind) rather than solver duals.
4. **Decide** (`explore curate`): pin a few allocations and commit on the tradeoffs.

This example stays on the heuristic frontier: the audience-overlap term makes Reach a maximize-quadratic, outside the exact solver's convex minimize-risk scope. For the exact mean-variance counterpart, where the quadratic is a minimize-risk term, see [`supplier_selection`](../supplier_selection/) and [`investment_portfolio`](../investment_portfolio/).
