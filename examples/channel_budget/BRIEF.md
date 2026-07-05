# The ask (step 1 input — paste this with `data.csv` and `reach_overlap.csv`)

> We're splitting next quarter's media budget across 22 channel line items (`data.csv`):
> conversions, reach, ROAS, and brand-lift ratings per line item, plus which platform
> each belongs to. Reach doesn't add linearly — line items sharing an audience overlap,
> so combined reach is sub-additive; `reach_overlap.csv` carries the pairwise overlap
> matrix our media team measured (negative = shared audience).
>
> The decision is what percent of the budget each line item gets — shares total 100%.
> Maximize conversions, reach (with the overlap correction), ROAS, and brand lift
> (allocation-weighted; reach combines through the overlap matrix).
>
> Hard rules:
> - No line item above 15% of budget.
> - At most one active line item per platform group (the `platform_group` column).
> - Blended ROAS must stay at or above 2.0x.
>
> Two futures to stress-test:
> - **Signal loss** — measurement degrades: conversions read 20% lower across the board.
> - **Demand pullback** — conversions −15% and ROAS −10% across the board.

This brief plus the two CSVs is the complete upstream input: framing it should land on
the canonical model (proportional; Conversions/ROAS/BrandLift avg maximize, Reach
quadratic maximize with the overlap interaction matrix; the 15% cap; six ≤1 platform
group limits; the ROAS ≥ 2.0 bound; both scenarios as score adjustments).
