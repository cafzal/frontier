# Generation capacity planning

Loadable Frontier example — choose a generation capacity mix (share %) across 22 candidate projects to balance the energy trilemma: cheap power is dirty or intermittent, while clean + firm power is expensive. Too combinatorial and too constrained for a spreadsheet, and the correlated-intermittency risk is nonlinear, so an LLM can't reason it through alone.

- **`problem.json`**: 4 conflicting objectives (LCOE / CO2 minimize, Firmness maximize, VariabilityRisk minimize-quadratic) plus LandUse, proportional approach, constraints (no project >25%, a per-technology group limit, CO2 ≤0.20 t/MWh cap, Firmness ≥50 floor), and `carbon_price` / `low_renewables_year` scenarios.
- **`scores.json`** — 22 projects across 9 technologies (solar, onshore/offshore wind, gas CCGT, nuclear SMR, battery storage, geothermal, hydro, biomass) with LCOE/CO2/Firmness/LandUse/VariabilityRisk scores, plus the VariabilityRisk interaction (covariance) matrix that makes stacking correlated renewables raise portfolio risk super-linearly.

Load both into Frontier (`model create` → `model update` with the objectives/options/scores/constraints/interaction_matrices/scenarios → `solve run` → `explore`), or paste this to an agent connected to Frontier:

> Build a generation capacity mix from the 22 projects in scores.json — minimize LCOE, minimize CO2, maximize Firmness, and minimize VariabilityRisk (use the covariance matrix, not weighted-average risk), plus minimize LandUse. Constraints: no project over 25%, a cap per technology, CO2 under 0.20 t/MWh, Firmness at least 50. Show the tradeoff frontier across base and a low-renewables year where solar+wind correlations rise — the range of non-dominated mixes and where the knees are, not one "best."
