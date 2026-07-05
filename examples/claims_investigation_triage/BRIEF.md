# The ask (step 1 input — paste this with `data.csv`)

> Our special-investigations unit needs this month's triage list. `data.csv` has the 180
> model-flagged claims: expected recovery ($k), investigator hours, and a customer-friction
> score per claim, plus its line of business and whether it's a regulator-mandated
> referral. Four big-ticket claims dominate the book — PRP-1001, LIA-1001, PRP-1002, and
> WC-1001 — none of them among the mandated referrals.
>
> The decision is which claims to open — each is in or out. Maximize total expected
> recovery; minimize total hours and total friction.
>
> Hard rules:
> - Nine investigators give us at most 1,170 hours this month.
> - The quarterly plan requires at least $4,840k of expected recovery.
> - Work between 45 and 100 claims.
> - Per-line concentration caps: at most 38 Auto (AUT), 32 Property (PRP),
>   28 Liability (LIA), and 26 Workers' comp (WC) claims.
> - The six mandated referrals (marked in the CSV) must be worked.
>
> One future to stress-test:
> - **Capacity cut** — a catastrophe event pulls investigators onto CAT duty: capacity
>   drops to 1,140 hours; the recovery target and every other rule stay as-is.

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model (binary; three sum objectives; the hours cap, recovery floor, 45–100
cardinality, four line caps, six force-includes; the capacity_cut scenario as a full
constraint restatement with only the hours bound moved).
