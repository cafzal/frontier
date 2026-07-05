# The ask (step 1 input — paste this with `data.csv`)

> We're re-planning how to load the plant across ten products (`data.csv`): per-unit
> margin ($), throughput (k units/week), and a sustainability rating, plus which of our
> three lines each product runs on. For the four commodity SKUs the sheet also carries
> the re-priced margin if the input-cost spike we're worried about lands.
>
> The decision is what percent of plant capacity each product gets — shares total 100%.
> We want the best blended margin, throughput, and sustainability (allocation-weighted
> averages).
>
> Hard rules:
> - No product above 30% of capacity.
> - Changeovers limit each line to at most 2 active SKUs.
>
> Two futures to stress-test:
> - **Input cost spike** — the four commodity SKUs re-price to the margins in the
>   `margin_under_input_cost_spike` column; everything else unchanged.
> - **Capacity crunch** — a demand surge tightens the per-product cap from 30% to 25%;
>   every other rule stays as-is.

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model in `problem.json` + `scores.json` (proportional; three avg maximize
objectives; the 30% cap; three per-line ≤2 group limits; both scenarios).
