# The ask (step 1 input — paste this with `data.csv`)

> We have one quarter of constrained memory-chip supply to divide across 36 customers, and
> committed demand runs about 1.8× what the fabs will deliver — every plan cuts someone.
> Attached (`data.csv`) is the account list: for each customer, the revenue ($M), strategic
> value, and demand fragility we'd get **per 1% of the quarter's supply**, plus contractual
> minimum shares and credit caps where they exist.
>
> The decision is what percent of supply each customer gets — shares must total 100%. We
> want the most revenue and strategic value we can get while minimizing exposure to demand
> that could evaporate.
>
> Hard rules:
> - Anti-concentration: no customer above 8% of supply.
> - The contractual minimums in the CSV are binding floors (HYP-01 ≥6%, HYP-02 ≥5%,
>   HYP-03 ≥4%, IND-01 ≥4%, IND-02 ≥3%).
> - The credit caps in the CSV are binding ceilings (DST-01 ≤4%, DST-02 ≤5%).
> - Board mandate: total strategic value of the allocation must reach at least 4.8.
>
> Two futures I want stress-tested:
> - **Fab outage** — supply drops ~25%. The absolute contractual commitments don't move,
>   so as shares of the smaller quarter the five floors rise to 8/7/5/5/4. Every other rule
>   stays as-is.
> - **Spot surge** — spot pricing spikes ~35% on the distributor (DST) channel; contracts
>   hold, so only DST revenue changes.

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model in `problem.json` + `scores.json` (proportional approach; three
sum-aggregated objectives; 8% max allocation; five allocation-bound floors + two caps; the
StrategicValue ≥ 4.8 objective bound; the two scenarios — fab_outage as a full constraint
restatement with raised floors, spot_surge as DST revenue score overrides).
