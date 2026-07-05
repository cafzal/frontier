# Marketing channel budget

Split a media budget across 22 channel × audience/geo combinations balancing four conflicting goals: Conversions, Reach, ROAS, and Brand Lift. Direct-response channels convert and return well but reach few people; broad upper-funnel channels reach and build brand but convert poorly, and same-audience channels overlap, so reach combines sub-additively. Too combinatorial and interaction-laden for a spreadsheet or an LLM to allocate by hand.

- **`problem.json`**: 4 objectives (Conversions / ROAS / Brand Lift averaged, Reach quadratic via an audience-overlap matrix), proportional approach, constraints (no channel >15%, ≤1 line item per platform, blended ROAS ≥2.0x), and two scenarios (`signal_loss`, `demand_pullback`).
- **`scores.json`**: the 22 channels with per-channel scores, plus the Reach audience-overlap interaction matrix (negative off-diagonals between same-audience channels = diminishing combined reach).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="channel_budget"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) + [reach_overlap.csv](reach_overlap.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“How should we split the marketing budget across these 22 channels? Show me the real tradeoffs — remembering that channels sharing an audience overlap rather than add — and what happens if measurement signal degrades or demand pulls back.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the Conversions/Reach/ROAS/Brand-Lift frontier (Reach combining sub-additively via the overlap matrix) plus the `signal_loss` and `demand_pullback` scenario frontiers.
2. *“Where do diminishing returns kick in, and which limit is holding the mix back?”*
   `explore tradeoffs` knees + `explore sensitivity`: maximizing Reach is a non-convex quadratic, so the exact backend *declines* this shape — no certificate; the examine reports the frontier-inferred binding analysis instead.
3. *“Keep the balanced split and the conversion-max one, and write the plan up for the growth review.”*
   `explore curate` per pick → `explore curated format="markdown"`: the handoff table.

This example stays on the heuristic frontier: the audience-overlap term makes Reach a maximize-quadratic, outside the exact solver's convex minimize-risk scope. For the exact mean-variance counterpart, where the quadratic is a minimize-risk term, see [`supplier_selection`](../supplier_selection/) and [`investment_portfolio`](../investment_portfolio/).
