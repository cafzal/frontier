# The ask (step 1 input — paste this with `data.csv`)

> We're finalizing next cycle's capital plan and I want help picking the portfolio. Attached
> is our project list (`data.csv`): 120 candidates, each scored with NPV ($M), cost ($M), a
> risk score, and a strategic-fit score, plus its category, any enabler project it requires,
> any project it's mutually exclusive with, and whether it's already committed.
>
> The decision is which projects to fund — each one is in or out. We want the most total NPV
> and total strategic fit we can get while holding down total spend and total risk exposure.
>
> Hard rules:
> - Total cost must stay within the $610M budget.
> - Fund between 18 and 40 projects.
> - Category caps: at most 8 Growth, 6 Digital, 6 R&D, and 7 Maintenance projects.
>   Compliance and Efficiency are uncapped.
> - A project with a `requires` entry can only be funded if that enabler is funded too.
> - For each mutually-exclusive pair, fund at most one.
> - The three projects marked `committed` are already contracted — they must be in.

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model in `problem.json` + `scores.json` (binary approach; four sum-aggregated
objectives; budget as an objective bound; cardinality 18–40; four category group caps; 10
dependencies; 10 exclusion pairs; 3 force-includes).
