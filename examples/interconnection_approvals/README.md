# Interconnection approvals

A regional grid operator approves which of 42 large-load interconnection requests to accept this cycle, alongside 16 enabling substation upgrades — maximizing net value while holding down capex and reliability risk, under zone queue caps, node-headroom rivalries, and the enabler structure that makes 2^58 portfolios combinatorial: upgrades carry **zero standalone value** and pay only through the requests they unlock, several requests share one enabler, one upgrade is the **staged second phase** of another (U12 requires U11), and two speculative upgrades with no committed load should never be bought (the frontier confirms they never are). The **scenarios-on-binary showcase**: four capex envelopes ($320M / $400M / $480M / $560M) modeled as scenarios, so the deliverable isn't one portfolio — it's which approvals survive every budget, and what each next $80M actually buys.

- **`problem.json`**: 3 objectives (NetValue maximize, Cost / ReliabilityRisk minimize, all `sum`), binary approach, base constraints (the $400M capex bound, ≤9 approvals per zone, 31 dependencies — 30 request→upgrade plus the staged U12→U11 — and 5 exclusion pairs), and the four budget scenarios — each `constraint_overrides` list **restates the full constraint set** (overrides replace, not merge) with only the capex bound moved.
- **`scores.json`**: the 58 options scored per objective; enabler upgrades carry negative ReliabilityRisk (they relieve system stress), the two speculative ones carry nothing.
- **`solutions.json`**: the base NSGA `run`, the exact-MILP `exact_run` overlay, and the per-budget `scenario_run`.

Load with `model load source="interconnection_approvals"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

0. **Start upstream (the real step 1):** paste [BRIEF.md](BRIEF.md)'s ask together with [data.csv](data.csv) — the raw inputs a decision owner would actually have. Framing that input (`model create` + `model update`) lands on exactly this problem: the kit reconstructs `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load` is the shortcut that skips this step.
1. *“Which requests should we approve this cycle at the $400M envelope? Show me the real value/capex/reliability choices.”*
   `solve run` → `explore tradeoffs`: the base approval frontier over 2^58 portfolios.
2. *“Run all four budget levels: which approvals survive every envelope, and what does each step up to $560M actually buy?”*
   `solve run_scenarios` → `explore scenario_frontiers` + `explore scenario_results`: max net value climbs $941M → $1,020M → $1,167M → $1,370M; R33 and R41 sit in the value-oriented plans at *every* envelope, and the stretch envelope admits a nameable block — U4 with R04, R08, and R12, a shared enabler arriving *with* the requests that pay for it.
3. *“Keep the base pick and the stretch portfolio. How sure can we be — and which upgrades earn their cost only through what they unlock?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify` (the overlay never buys the two speculative upgrades — the cheapest sanity check in the bundle) → `explore composition`: U2 unlocks five requests and appears wherever two or more of them do; the staged U11→U12 chain means stage two never appears without stage one.
4. *“Write the approval recommendation up for the commission.”*
   `explore curated format="markdown"`: the handoff table.

**Scale note.** Fifty-eight options keeps the curated finalists narratable while 2^58 portfolios with 31 dependencies and 5 exclusions is far past hand-pruning. For the same binary MILP arc at 120-option scale (where exact coverage reclaim is the headline), see [`capital_project_selection_120`](../capital_project_selection_120/); for scenarios that shock *scores* rather than the constraint set, see [`supplier_selection`](../supplier_selection/) and [`charging_network_siting`](../charging_network_siting/).
