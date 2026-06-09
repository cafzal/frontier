# Budget allocation

Loadable Frontier example — split a fixed growth budget across 8 initiatives, trading **ROI** against **strategic reach**, with no single initiative above 35%. Two purely linear objectives over a continuous allocation make this an **exact multi-objective LP** — the showcase for Frontier's solver-exact **duals**: which initiative is a near-miss, and which limit is binding.

- **`problem.json`** — the definition: 2 objectives (ROI maximize, Strategic Reach maximize), proportional approach, one constraint (no initiative over 35%).
- **`scores.json`** — the 8 initiatives and their per-objective scores (ROI %, reach 0–10).
- **`solutions.json`** — the results: the exploratory NSGA `run` and the exact-LP `exact_run` overlay (HiGHS), every point carrying solver-exact shadow prices + reduced costs.

Load into Frontier (`model create` → `model update` with the objectives/options/scores/constraints → `solve run` → `solve run` with `solver="highs"` → `explore`), or paste this to an agent connected to Frontier:

> Split a fixed growth budget across the initiatives in scores.json to maximize ROI and strategic reach, with no initiative over 35%. Solve it exactly (solver=highs), show me the ROI-vs-reach frontier and a few representative plans, and use the solver's duals to tell me which initiative is the closest near-miss and which cap is holding me back — not one "best."

## Explainability — shadow prices & near-misses

Because the objectives are purely linear, this routes to the **exact LP path**, so `explore sensitivity` returns **`solver_exact`** duals (not the frontier-inferred estimate) on every point:

- **Where to invest** — the binding-constraint shadow prices. At the balanced plan, the **Strategic Reach** floor prices at **~4.0**: each additional point of reach costs ~4% ROI. Across the frontier that reach shadow price **rises** (≈0 → 1.7 → 4.0 → …) — the exact diminishing-returns curve (`frontier_shadow_price_trend`).
- **Near-misses** — initiatives left at 0, ranked by reduced cost (closest first): **Localization (10)**, **Data Platform (11)**, **Enterprise SSO (16)**, **Partner API (18)**. Localization's ROI would need to improve by ~10 before it earns a slot; the smallest reduced cost is the one to re-examine.
- **Capped options** — **AI Copilot** and **Self-Serve Onboarding** sit at the **35% cap** with a negative reduced cost: the solver would fund *more* if the cap allowed. The lever there isn't the option — it's *your own limit*.

The decision read: a small **near-miss** says "improve the option"; a binding **cap** says "relax the limit you set." Both come straight from the solver's exact output — the same duals HiGHS and cuOpt expose on the LP path. (For the quadratic mean-variance counterpart, see [`investment_portfolio`](../investment_portfolio/); for binary selection with no duals, [`capital_project_selection_120`](../capital_project_selection_120/).)
