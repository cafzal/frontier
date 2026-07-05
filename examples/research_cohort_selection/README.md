# Research cohort selection

Pick exactly 24 of 144 screened volunteers for a biomarker study, maximizing signal strength while holding down retention risk and per-participant cost. The conflict is built into the screening reality: the highest-expression volunteers sit at the expensive academic centers and carry the most dropout risk, so a top-24-by-signal list lands entirely in one stratum — the floors require all six. Exact-K selection (`cardinality min=max=24`) over C(144,24) ≈ 10^28 possible cohorts, with per-stratum **floors** and per-site caps, all-`sum` objectives — certifiable — and the **emergent-guarantee audit**: prove a cap nobody wrote down.

- **`problem.json`**: 3 objectives (SignalStrength maximize, RetentionRisk / CostPerParticipant minimize, all `sum`), binary approach, and the structural-diversity constraints (exactly 24; stratum floors 4/4/4/3/3/2 with a deliberately loose stated cap of 8; at most 4 per site; three same-household exclusion pairs; one screen-failed volunteer `force_exclude`d — the data-hygiene beat).
- **`scores.json`**: the 144 volunteers scored per objective, signal clustering at the costly high-dropout sites by construction.
- **`solutions.json`**: the exploratory NSGA `run` plus the exact-MILP `exact_run` overlay — every point exactly 24 volunteers, floors held.

Load with `model load source="research_cohort_selection"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“Help me pick the strongest 24-person cohort from these 144 volunteers — signal against dropout risk against cost.”*
   `solve run` → `explore tradeoffs`: the frontier of exact-24 cohorts — it reaches ~99% of a top-24-by-signal ranking's signal at lower risk *and* cost, because the floors leave real choice inside each stratum.
2. *“Keep the balanced cohort and the max-signal one. Are these actually optimal?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify`: the exact MILP overlay; the binding read reports the site caps and stratum floors (the floor read prices one extra member above the floor).
3. *“Tell me something the protocol doesn't say: how many rare-variant (F) volunteers can any legal cohort carry — does "at most 6" hold across every one, and does "at most 5"?”*
   `explore audit`: "at most 6 from F" `holds` across every feasible cohort — nobody wrote 6 anywhere; the floors on A–E (4+4+4+3+3) against K=24 *imply* it. "At most 5" flips to `violated`, with a witness cohort carrying exactly 6.
4. *“Write the shortlist up for the steering committee.”*
   `explore curated format="markdown"`: the handoff table.

**Emergence note.** This is the audit flavor [`claims_investigation_triage`](../claims_investigation_triage/) doesn't show: triage proves a *stated-policy* guarantee (mandated claims always investigated); here the guarantee is *unstated arithmetic* — the interplay of exact-K and floors tightening a loose cap from 8 to a provable 6. Same tool, different epistemics: one verifies the model encodes the policy, the other reveals what the policy entails. For the same exact-K + floors mechanics at selection-with-budget scale, see [`capital_project_selection_120`](../capital_project_selection_120/).
