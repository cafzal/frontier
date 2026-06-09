# Marketing channel budget

Loadable Frontier example — split a media budget across 22 channel x audience/geo options balancing four genuinely conflicting goals (Conversions, Reach, ROAS, Brand Lift). Direct-response channels convert and return well but reach few people; broad upper-funnel channels reach and build brand but convert poorly, and same-audience channels overlap, so reach combines sub-additively. Too combinatorial and interaction-laden for a spreadsheet or an LLM to allocate by hand.

- **`problem.json`** — definition: 4 objectives (Conversions / ROAS / Brand Lift averaged, **Reach quadratic** via an audience-overlap matrix), proportional approach, constraints (no channel >15%, ≤1 line item per platform, blended ROAS ≥2.0x), and two scenarios: `signal_loss` (measurement erosion cuts attributable conversions) and `demand_pullback` (a consumer downturn softens response and efficiency).
- **`scores.json`** — the 22 channels with per-channel Conversions / Reach / ROAS / BrandLift scores, plus the Reach audience-overlap interaction matrix (negative off-diagonals between same-audience channels = diminishing combined reach).

Load both into Frontier (`model create` → `model update` with the objectives/options/scores/constraints/interaction_matrices/scenarios → `solve run` → `explore`), or paste this to an agent connected to Frontier:

> Allocate my marketing budget across the 22 channels in scores.json — maximize Conversions, Reach, ROAS, and Brand Lift. Treat Reach with the audience-overlap matrix (same-audience channels overlap, so combined reach is sub-additive — not weighted-average reach). Constraints: no channel over 15%, at most one line item per platform, blended ROAS at least 2.0x. Show the tradeoff frontier — which mixes lean efficiency vs reach vs brand, and where the knees are — and how it shifts under measurement signal loss and a demand pullback. Not one "best."
