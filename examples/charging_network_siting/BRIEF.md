# The ask (step 1 input — paste this with `data.csv` and `catchment_overlap.csv`)

> We're picking which of 72 candidate EV fast-charging sites to build (`data.csv`):
> drivers served (k/day) and build cost ($M) per site, plus its metro or corridor, any
> competing lot it's mutually exclusive with, and whether it's already committed (the
> North-corridor flagship is). Two nearby sites share drivers — linear reach overstates
> coverage — so `catchment_overlap.csv` carries the pairwise catchment-overlap matrix
> (strong within a catchment, near zero across regions). For the sites affected, the
> sheet also carries drivers-served under a corridor adoption surge and build cost if
> metro grid quotes come back high.
>
> The decision is which sites to build — each is in or out. Maximize total drivers
> served; minimize total cost and the overlap correction (through the matrix).
>
> Hard rules:
> - Total build cost at or below $34M.
> - Build between 16 and 24 sites.
> - Every metro (HAR, CED, EAS, BRK, KNG, NOR, WES) gets at least 1 site and at most 4.
> - Each highway corridor (NCR, CCR, VCR) gets at most 5.
> - For each mutually-exclusive pair, build at most one.
> - The committed flagship (NCR-01) must be built.
>
> Two futures to stress-test:
> - **Adoption surge** — corridor demand jumps: the affected sites' drivers-served move
>   to the `drivers_under_adoption_surge` column values.
> - **Grid cost inflation** — metro quotes come back high: the affected sites' costs
>   move to the `cost_under_grid_inflation` column values.

This brief plus the two CSVs is the complete upstream input: framing it should land on
the canonical model (binary; DriversServed sum maximize, Overlap quadratic minimize with
the interaction matrix, Cost sum minimize; the budget bound, 16–24 cardinality, seven
metro floor/cap groups, three corridor caps, five exclusion pairs, the force-include;
both scenarios as score overrides).
