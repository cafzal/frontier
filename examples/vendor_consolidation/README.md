# Post-merger vendor consolidation

**The decision.** Two merged subsidiaries — Northgate and Selwyn — arrive with 48 live tooling contracts across eight categories, and the consolidated estate keeps 14–18 of them: cheapest annual spend, strongest capability, lowest transition risk. The conflict is structural. Capability concentrates in the expensive contracts, the cheap ones carry the roughest migrations, and the category floors force coverage of the two corners a spend ranking abandons first.

**Why Frontier.** The showcase here is **messy upstream data inside the system under test**. The two books arrive as two spreadsheets that disagree with each other: Northgate rates capability 1–5, Selwyn rates it 0–100; the same vendor appears on both lists under variant names; two Selwyn rows ship a blank transition-risk cell; one contract is beaten on every axis by another. Framing surfaces all four *as the model goes in* — `model update` echoes `score_variance_by_objective`, `near_duplicate_options`, `dominated_options`, and the score-matrix check on every structural change — so the cleanup is part of the decision record instead of a spreadsheet step nobody witnessed. Binary MILP on top: HiGHS certify **and** `explore audit` are both first-class.

**What ships here** — the raw inputs (step 1), the canonical model they frame into, and pre-solved results:

- **`vendors_northgate.csv`** and **`vendors_selwyn.csv`**: the two books exactly as they arrive — different capability scales, overlapping vendors under variant names, two blank transition-risk cells. Both are what step 1 pastes.
- **`problem.json`**: 3 objectives (AnnualCost minimize `$k/yr`, Capability maximize `0-10`, TransitionRisk minimize `0-10`, all `sum`), binary approach, and the constraints the ask states — keep 14–18 contracts; at least 1 and at most 3 vendors per category (8 categories); three `force_include`s for the contracts with four or more years left on term; one `exclusion_pair` for the two endpoint agents that can't coexist; three more `exclusion_pair`s for the duplicate-tool pairs; one `force_exclude` for the dominated contract.
- **`scores.json`**: the 48 contracts on the common 0–10 capability scale, with the two unrated contracts carried at the stated 6.4 estimate.
- **`solutions.json`**: the exploratory NSGA `run` (40 plans), the exact-MILP `exact_run` overlay (24 plans), and the `novation_repricing` `scenario_run`.

## The runbook

1. **Frame it from the raw inputs** — paste this ask, together with both CSVs, into a fresh session:

   > We merged two subsidiaries and now run two tooling books. `vendors_northgate.csv` and
   > `vendors_selwyn.csv` list all 48 live contracts: annual cost ($k), a capability rating, a
   > transition-risk rating (0-10), the tooling category, years left on term, and any product
   > this one conflicts with.
   >
   > The two books rate capability on different scales — Northgate scored it 1-5, Selwyn scored
   > it 0-100. Put both on a common 0-10 scale before anything else: a Northgate rating maps as
   > (rating - 1) * 2.5, a Selwyn rating as rating / 10, each rounded to 2 decimals.
   >
   > Three things to fix on the way in:
   > - Selwyn left the transition-risk cell blank on two contracts. Our transition team's
   >   default for an unrated vendor is 6.4 — use that for both.
   > - The books overlap, sometimes under variant names. Once the ratings are on the common
   >   scale, treat two contracts whose annual cost, capability, and transition risk are each
   >   within 2% of that column's spread across all 48 as the same tool, and let us sign at
   >   most one of the pair.
   > - Drop any contract that another contract beats on all three — cheaper, more capable, and
   >   easier to move to.
   >
   > The decision is which contracts the merged company keeps — each is in or out. Minimize
   > total annual cost and total transition risk; maximize total capability.
   >
   > Hard rules:
   > - Keep between 14 and 18 contracts.
   > - Every tooling category keeps at least 1 vendor, and no category runs more than 3.
   > - Contracts with four or more years left on term stay — we can't exit them.
   > - The two conflicting products (marked in the CSVs) can't both be kept.
   >
   > One future to stress-test:
   > - **Novation repricing** — novating the Selwyn contracts to the merged entity re-opens
   >   their pricing and every Selwyn line comes back 12% higher. Capability and transition
   >   risk hold.

   Framing that input (`model create` + `model update`) lands on exactly this problem — the two CSVs plus the ask's stated rules reconstruct `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load source="vendor_consolidation"` is the shortcut: it skips framing and restores the pre-solved runs too.

2. *“Before I trust any of this — what did the merge of those two spreadsheets actually give me?”*
   Every structural `model update` echoes a `metrics.data` block, so the intake audits itself as it lands:
   - **`score_variance_by_objective`** reads `Capability: 2.02` on the normalized model. Paste the two books without the scale mapping and the same column reads **361.0** — a **179x** artifact of two rating scales stacked in one column, and nothing at all about vendor quality.
   - **`near_duplicate_options`** names exactly three pairs, each within **1.15%** of every objective's spread: *Aldercrest Inc / Aldercrest*, *Northwind Data Systems / NW Data*, *Quillon Security Labs / Quillon Labs*. The same tool, bought twice, entered twice. The merge rule turns each into an `exclusion_pair` — sign one.
   - **`dominated_options`** names exactly one: **Thornbury Analytics** ($176.3k, capability 3.5, risk 7.3), beaten on all three by **Oakhurst Metrics** ($172.3k, 3.9, 7.0). It leaves as a `force_exclude`.
   - Paste the CSVs with the blank cells left blank and the score-matrix check refuses the model and names the two cells — `Corvid Tables` and `Petrichor Notes` on `TransitionRisk` — which is where the 6.4 estimate in the ask comes from.

   **Why this beat earns its keep.** The alternative — eyeball the two lists for duplicates, then rank what's left by capability-per-dollar and keep the top 16 — gets all three of these wrong: name-matching merges *Aldercrest Inc / Aldercrest* and misses the other two, so the shortlist signs both halves of the Quillon pair and pays **$431.8k for one product**; the dominated Thornbury contract makes the cut (16th of 16); and the ranking leaves **ITSM** with no vendor at all, breaking a floor the merged company committed to. Its headline numbers ($2,836k for 70.4 capability) belong to a plan the rules forbid.

3. *“Show me the real range — what does the consolidated estate cost across cheap, capable, and low-disruption?”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: 40 plans spanning **$2,516k–$5,474k**, capability **39.0–76.5**, transition risk **49.9–114.0**. The conflict shows up in the frontier itself: across those plans annual cost correlates **+0.64** with capability and **−0.34** with transition risk — capability is what you pay for, and the cheapest estates are the ones that hurt most to migrate. The `novation_repricing` frontier lands **8.0%** above the base band at the lean end and **10.2%** above it at the rich end, because the richer estates lean harder on the Selwyn book.

4. *“Keep the lean estate and the balanced one. Are they actually optimal?”*
   `explore curate` per pick → `solve solver="highs" exact=true` → `explore certify`: the zero-gap MILP overlay, 24 certified plans, reported per pick. Against it the heuristic frontier gives up little and is honest about where — exact strictly dominates **4 of the 40** NSGA plans (10%) and reclaims **8.1%** of hypervolume; NSGA dominates **no** exact point, so the invariant holds. The overlay sharpens both corners the floors squeeze: the leanest certified estate is 14 contracts at **$2,401.7k**, and the strongest reaches capability **79.2** where the heuristic stopped at 76.5.

5. *“Tell me something the rules don't say. Eighteen of these contracts are still inside a multi-year term — how many of them can any legal estate actually carry?”*
   `explore audit` with a list of nine properties — a floor of at least 1 vendor in each of the eight categories, plus "at most 16 of the 18 in-term contracts". Verdict `holds`, with a per-property breakdown: the floors confirm the model encodes the commitment, and the cap of 16 is a **theorem** of the rules rather than one of them. Nothing caps in-term contracts anywhere; the 18-contract ceiling, the three-per-category cap, and the floors on SecurityOps and Collaboration — the two categories with no in-term contracts at all — jointly reserve two slots that no locked contract can occupy. Tighten it to 15 and the verdict flips to `violated`, with a concrete witness: an 18-contract estate carrying exactly 16 in-term.

6. *“Write it up for the integration steering committee.”*
   `explore scenario_results` → `explore curated format="markdown"`: eight contracts hold their place across the repricing — including all three no-exit contracts — so the handoff can lead with the picks that survive the price the merger itself sets, and the audited guarantee, which holds whichever finalist the committee takes.

**Data-quality note.** Every other bundled example ships data that is already clean, which quietly puts the hardest part of a real decision outside the frame. Here the mess is the subject: the same engine call that frames the model reports what's wrong with it, and each advisory maps to a named repair the ask states — a scale mapping, three `exclusion_pair`s, one `force_exclude`, one filled estimate. The shipped model's hygiene constraints are exactly the options the advisories named, which is a property the generator asserts rather than a claim the README makes. For the same binary MILP arc on clean data — exact-K selection with group floors, and an audit that tightens a stated cap — see [`research_cohort_selection`](../research_cohort_selection/); for the compound-property audit at scale, [`claims_investigation_triage`](../claims_investigation_triage/).
