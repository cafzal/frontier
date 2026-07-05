# The ask (step 1 input — paste this with `data.csv`)

> We're setting next year's growth budget and I want help splitting it across eight
> candidate initiatives (`data.csv`): each is rated on ROI (%) and strategic reach (0–10).
>
> The decision is what percent of the budget each initiative gets — shares total 100%,
> and an initiative can get nothing. We want the highest blended ROI and the most
> strategic reach; both read as the allocation-weighted average of the ratings.
>
> One hard rule: no single initiative may take more than 35% of the budget.

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model in `problem.json` + `scores.json` (proportional approach; two
avg-aggregated maximize objectives; the 35% per-initiative cap).
