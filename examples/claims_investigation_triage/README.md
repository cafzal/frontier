# Claims investigation triage

Pick which of 180 model-flagged claims a special-investigations unit works this month, maximizing expected recovery while holding down investigator hours and the friction of investigating customers who turn out clean. A 1,170-hour capacity (nine investigators), a $4.84M recovery target, per-line concentration caps, and six regulator-mandated referrals couple every pick to every other: the best recovery-per-hour ratios sit on borderline claims where friction accrues, so a ratio ranking breaches the Auto cap (57 picks against a cap of 38) and pays 429 friction points where the frontier meets the same target at 313. At 2^180 candidate plans this is far past any spreadsheet; binary, all-`sum` objectives — the full explore → certify arc — and the **`explore audit` governance showcase**: prove what holds across *every* feasible plan, not just the frontier.

- **`problem.json`**: 3 objectives (ExpectedRecovery maximize, Hours / Friction minimize, all `sum`), binary approach, the coupled constraints (1,170-hour cap, $4,840k recovery floor, per-line caps Auto 38 / Property 32 / Liability 28 / Workers' comp 26, portfolio 45–100, six `force_include` referrals), and the `capacity_cut` scenario (a CAT deployment drops capacity to 1,140 hours).
- **`scores.json`**: the 180 claims scored per objective — framed as the output of an upstream fraud-scoring model, so shaky scores are part of the story (see the data_collection skill).
- **`solutions.json`**: the exploratory NSGA `run`, the exact-MILP `exact_run` overlay, and the per-scenario `scenario_run`.

Load with `model load source="claims_investigation_triage"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“Which claims should we work this month? Show me my real choices across recovery, hours, and how much we'd annoy legitimate customers — and how the picture changes if the CAT event pulls one of my seniors.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the frontier over 2^180 plans (at the recovery target, friction spans 313 to ~430 — the axis a recovery-per-hour ranking can't see) plus the `capacity_cut` scenario frontier; the hours capacity and the Auto line cap bind.
2. *“Keep the balanced plan and the gentlest one that still hits the target. How much should I trust these?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify`: the exact-MILP overlay — its dominance audit is the scale story at 180 binary options — plus coverage, the invariant, and quality gates.
3. *“Is there any legal plan that skips the regulator referrals or any of the four big-ticket claims? And is LIA-1002 covered too?”*
   `explore audit` with a LIST of ten force_include properties: verdict `holds` with a per-property breakdown — a proof over the whole feasible space in one call. Only six are mandated; the four whales are forced by *arithmetic* (see the emergence note). Audit LIA-1002 alone and it flips to `violated`, with a concrete plan that skips it.
4. *“What's holding us back, what should we stress-test next quarter, and write this up for the claims committee.”*
   `explore sensitivity` (frontier-inferred binding analysis + `suggested_scenarios` — no MILP duals) → `explore curated format="markdown"`: lead the writeup with the audited guarantee, since it holds whichever finalist is picked.

**Emergence note.** The audit's headline guarantee is *not* a restatement of a constraint. Nothing in the model forces the four whales — drop any one and the remaining book tops out below the $4,840k floor inside 1,170 hours, so their inclusion is a *theorem* of the constraint set, provable only by reasoning over the whole feasible region. That's what `explore audit` adds beyond reading the frontier: every frontier plan happening to include a claim is an observation; `holds` is a proof. For the same binary MILP arc at portfolio scale, see [`capital_project_selection_120`](../capital_project_selection_120/); for what audit can't do here (a quadratic objective blocks nothing, but a *bound* on one would), see [`charging_network_siting`](../charging_network_siting/).
