# Generation capacity planning

Choose a generation capacity mix (share %) across 22 candidate projects to balance the energy trilemma: cheap power is dirty or intermittent, while clean and firm power is expensive. Too combinatorial and too constrained for a spreadsheet, and the correlated-intermittency risk is nonlinear, so an LLM can't reason it through alone. The minimize-risk quadratic puts it on Frontier's exact mean-variance (QP) path.

- **`problem.json`**: 5 objectives (LCOE / CO2 / VariabilityRisk / LandUse minimize, Firmness maximize; VariabilityRisk is quadratic), proportional approach, constraints (no project >25%, a per-technology group limit, CO2 ≤0.20 t/MWh, Firmness ≥50), and two scenarios (`carbon_price`, `low_renewables_year`).
- **`scores.json`**: 22 projects across 9 technologies with LCOE/CO2/Firmness/LandUse/VariabilityRisk scores, plus the VariabilityRisk covariance matrix (stacking correlated renewables raises portfolio risk super-linearly).
- **`solutions.json`**: the exploratory NSGA `run` plus the per-scenario `scenario_run`.

Load with `model load source="capacity_planning"`, or paste this to an agent connected to Frontier:

> Build a generation capacity mix from the 22 projects in scores.json to minimize LCOE, minimize CO2, maximize Firmness, minimize VariabilityRisk (via the covariance matrix), and minimize LandUse. Constraints: no project over 25%, a cap per technology, CO2 under 0.20 t/MWh, Firmness at least 50. Explore the tradeoffs across the base case and a low-renewables year, solve it exactly (solver=highs), certify it, and read the duals.

## The workflow

1. **Explore** (`solve run`): the capacity-mix frontier across the five objectives, with its extremes, a balanced mix, and the knees.
2. **Stress-test** (`solve run_scenarios`): re-solve under `carbon_price` and `low_renewables_year` (solar and wind correlations rise) to find robust mixes.
3. **Certify** (`solve solver="highs"`, then `explore certify`): the exact mean-variance QP overlay audits the heuristic frontier and sharpens the minimum-variability-risk corner.
4. **Examine sensitivity** (`explore sensitivity`): solver-exact duals: which constraint (the CO2 cap, the Firmness floor) is the binding lever, which project is the closest near-miss, and which sit pinned at a cap.
5. **Decide** (`explore curate`): pin a few capacity mixes and choose on the tradeoffs.

For the same QP shape, see [`supplier_selection`](../supplier_selection/) and [`investment_portfolio`](../investment_portfolio/).
