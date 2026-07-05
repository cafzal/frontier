# The ask (step 1 input — paste this with `data.csv`, `variability_interactions.csv`, and `variability_low_renewables.csv`)

> We're choosing a generation capacity mix across 22 candidate projects (`data.csv`):
> LCOE ($/MWh), CO2 (t/MWh), a firmness rating, a variability-risk rating, and land use,
> plus each project's technology. Stacking correlated renewables compounds portfolio
> variability — `variability_interactions.csv` carries the pairwise covariance our
> planners use (correlated within a resource region, near zero across).
>
> The decision is each project's share of the mix — shares total 100%. Minimize cost,
> carbon, correlated variability risk, and land use; maximize firmness
> (allocation-weighted; variability combines through the covariance matrix).
>
> Hard rules:
> - No project above 25% of the mix.
> - Portfolio CO2 at or below 0.20 t/MWh.
> - Portfolio firmness at or above 50.
>
> Two futures to stress-test:
> - **Carbon price** — LCOE reads 15% higher across the board.
> - **Low renewables year** — resource correlation worsens: swap in the
>   `variability_low_renewables.csv` covariance; everything else unchanged.

This brief plus the three CSVs is the complete upstream input: framing it should land on
the canonical model (proportional; LCOE/CO2/LandUse minimize + Firmness maximize avg +
VariabilityRisk minimize quadratic; the 25% cap; the CO2 and firmness bounds; both
scenarios — one score adjustment, one interaction-matrix override).
