# The ask (step 1 input — paste this with `data.csv`)

> We're deciding this cycle's interconnection approvals (`data.csv`): 42 large-load
> requests plus 16 enabling substation upgrades, each with net value ($M), capex ($M),
> and a reliability-risk score (upgrades relieve system stress, so theirs can be
> negative), plus the request's zone, any enabler it requires (several requests share
> one; upgrade U12 is the staged second phase of U11), and any mutually-exclusive
> node-headroom rival.
>
> The decision is which requests and upgrades to approve — each is in or out. Maximize
> total net value; minimize total capex and total reliability risk.
>
> Hard rules:
> - Total capex at or below the $400M envelope.
> - At most 9 approvals per zone (the `zone` column).
> - An option with a `requires` entry can only be approved if its enabler is approved.
> - For each mutually-exclusive pair, approve at most one.
>
> And the real question is budget-shaped: run four capex envelopes as futures —
> **$320M / $400M / $480M / $560M** — same rules each time, only the capex ceiling
> moves. Which approvals survive every envelope, and what does each step up buy?

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model (binary; three sum objectives; the capex bound, four ≤9 zone groups, 31
dependencies, five exclusion pairs; four scenarios, each a full constraint restatement
with only the capex bound moved).
