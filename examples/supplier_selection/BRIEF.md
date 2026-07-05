# The ask (step 1 input — paste this with `data.csv` and `concentration_interactions.csv`)

> We're splitting a production order across 25 global suppliers (`data.csv`): unit cost,
> reliability, lead time, and ESG risk ratings, plus each supplier's region. Regional
> disruptions are correlated — concentrating within a region compounds the risk — so
> `concentration_interactions.csv` carries the pairwise concentration-risk matrix (high
> within a region, near zero across regions).
>
> The decision is what percent of the order each supplier gets — shares total 100%.
> Minimize cost, lead time, ESG risk, and correlated concentration risk; maximize
> reliability (allocation-weighted; concentration risk combines through the matrix).
>
> Hard rules:
> - No supplier above 15% of the order.
> - At most 3 active suppliers per region.
> - Weighted reliability must stay at or above 78.
>
> Two futures to stress-test:
> - **China disruption** — every CN supplier goes offline (zero allocation allowed);
>   every other rule stays as-is.
> - **Demand surge** — per-supplier capacity tightens: the cap drops from 15% to 10%;
>   every other rule stays as-is.

This brief plus the two CSVs is the complete upstream input: framing it should land on
the canonical model (proportional; four minimize + one maximize objectives with
ConcentrationRisk quadratic; the 15% cap; six ≤3 region group limits; the reliability
floor; both scenarios as full constraint restatements).
