# Generation capacity planning

Choose a generation capacity mix (share %) across 22 candidate projects to balance the energy trilemma: cheap power is dirty or intermittent, while clean and firm power is expensive. Too combinatorial and too constrained for a spreadsheet, and the correlated-intermittency risk is nonlinear, so an LLM can't reason it through alone. The minimize-risk quadratic puts it on Frontier's exact mean-variance (QP) path.

- **`problem.json`**: 5 objectives (LCOE / CO2 / VariabilityRisk / LandUse minimize, Firmness maximize; VariabilityRisk is quadratic), proportional approach, constraints (no project >25%, CO2 ≤0.20 t/MWh, Firmness ≥50), and two scenarios (`carbon_price`, `low_renewables_year`).
- **`scores.json`**: 22 projects across 9 technologies with LCOE/CO2/Firmness/LandUse/VariabilityRisk scores, plus the VariabilityRisk covariance matrix (stacking correlated renewables raises portfolio risk super-linearly).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="capacity_planning"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) + [variability_interactions.csv](variability_interactions.csv) + [variability_low_renewables.csv](variability_low_renewables.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“Help me build the generation mix from these 22 projects. Show me the real tradeoffs between cost, carbon, firmness, variability, and land use — and how they shift in a low-renewables year.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the five-objective capacity frontier plus per-scenario frontiers for `carbon_price` and `low_renewables_year`.
2. *“Keep the balanced mix and the firmest one. Are these actually optimal?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify`: the exact mean-variance QP overlay, sharpest at the minimum-variability-risk corner.
3. *“Which limit binds hardest — the CO2 cap or the firmness floor — and what would relaxing it buy?”*
   `explore sensitivity`: solver-exact duals naming the binding lever, the closest near-miss project, and which sit pinned at a cap.
4. *“Write it up for the planning board.”*
   `explore curated format="markdown"`: the handoff table.

For the same QP shape, see [`supplier_selection`](../supplier_selection/) and [`investment_portfolio`](../investment_portfolio/).
