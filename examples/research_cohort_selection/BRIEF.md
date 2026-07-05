# The ask (step 1 input — paste this with `data.csv`)

> We're selecting the cohort for a biomarker study from 144 screened volunteers
> (`data.csv`): a signal-strength score, a retention-risk score, and cost per
> participant ($k), plus each volunteer's stratum and clinical site, any same-household
> volunteer they can't both enroll with, and a screen-failed flag (V-118 failed —
> exclude them).
>
> The decision is who's in the cohort — the protocol requires exactly 24 participants.
> Maximize total signal strength; minimize total retention risk and total cost.
>
> Hard rules:
> - Exactly 24 volunteers.
> - Stratum floors: at least 4 each from strata A, B, and C; at least 3 each from D and
>   E; at least 2 from F — and no more than 8 from any single stratum.
> - At most 4 volunteers per clinical site.
> - Same-household pairs (in the CSV): enroll at most one of each pair.
> - The screen-failed volunteer is out.

This brief plus `data.csv` is the complete upstream input: framing it should land on the
canonical model (binary; three sum objectives; exact-24 cardinality; six stratum
floor/cap groups; eight ≤4 site groups; three exclusion pairs; one force-exclude).
