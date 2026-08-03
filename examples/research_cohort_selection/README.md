# Research cohort selection

**The decision.** Pick exactly 24 of 144 screened volunteers for a biomarker study, maximizing signal strength while holding down retention risk and per-participant cost. The conflict is built into the screening reality: the highest-expression volunteers sit at the expensive academic centers and carry the most dropout risk, so a top-24-by-signal list lands entirely in one stratum — the floors require all six.

**Why Frontier.** Exact-K selection (`cardinality min=max=24`) over C(144,24) ≈ 10^28 possible cohorts, with per-stratum **floors** and per-site caps, all-`sum` objectives — certifiable — and the **emergent-guarantee audit**: prove a cap nobody wrote down.

**What ships here** — the raw inputs (step 1), the canonical model they frame into, and pre-solved results:

- **`data.csv`**: the raw inputs a decision owner would actually have — everything step 1 pastes.
- **`problem.json`**: 3 objectives (SignalStrength maximize, RetentionRisk / CostPerParticipant minimize, all `sum`), binary approach, and the structural-diversity constraints (exactly 24; stratum floors 4/4/4/3/3/2 with a deliberately loose stated cap of 8; at most 4 per site; three same-household exclusion pairs; one screen-failed volunteer `force_exclude`d — the data-hygiene beat).
- **`scores.json`**: the 144 volunteers scored per objective, signal clustering at the costly high-dropout sites by construction.
- **`solutions.json`**: the exploratory NSGA `run` plus the exact-MILP `exact_run` overlay — every point exactly 24 volunteers, floors held.

## The runbook

1. **Frame it from the raw inputs** — paste this ask, together with `data.csv`, into a fresh session:

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

   Framing that input (`model create` + `model update`) lands on exactly this problem — the ask plus the data reconstruct `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load source="research_cohort_selection"` is the shortcut: it skips framing and restores the pre-solved runs too.

2. *“Help me pick the strongest 24-person cohort from these 144 volunteers — signal against dropout risk against cost.”*
   `solve run` → `explore tradeoffs`: the frontier of exact-24 cohorts — it reaches ~99% of the best floor-feasible top-by-signal ranking's signal at lower risk *and* cost, because the floors leave real choice inside each stratum.
3. *“Keep the balanced cohort and the max-signal one. Are these actually optimal?”*
   `explore curate` per pick → `solve solver="highs" exact=true` → `explore certify`: the exact MILP overlay; the binding read reports the site caps and stratum floors (the floor read prices one extra member above the floor).
4. *“Tell me something the protocol doesn't say: how many rare-variant (F) volunteers can any legal cohort carry — does "at most 6" hold across every one, and does "at most 5"?”*
   `explore audit`: "at most 6 from F" `holds` across every feasible cohort — nobody wrote 6 anywhere; the floors on A–E (4+4+4+3+3) against K=24 *imply* it. "At most 5" flips to `violated`, with a witness cohort carrying exactly 6.
5. *“Write the shortlist up for the steering committee.”*
   `explore curated format="markdown"`: the handoff table.

**Emergence note.** This is the audit flavor [`claims_investigation_triage`](../claims_investigation_triage/) doesn't show: triage proves a *stated-policy* guarantee (mandated claims always investigated); here the guarantee is *unstated arithmetic* — the interplay of exact-K and floors tightening a loose cap from 8 to a provable 6. Same tool, different epistemics: one verifies the model encodes the policy, the other reveals what the policy entails. For the same exact-K + floors mechanics at selection-with-budget scale, see [`capital_project_selection_300`](../capital_project_selection_300/); for the same cardinality-band + group-floors MILP arc where the *inputs* are the hard part, [`vendor_consolidation`](../vendor_consolidation/).
