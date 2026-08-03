# Annual maintenance outage program

**The decision.** A regional utility locks next year's maintenance outage program: 30 jobs, each of which either takes an outage window in one of the four quarters or waits another year — maximizing risk reduction (MWh of unserved energy avoided) while holding down disruption cost and capex, under six outage crews a quarter, a $76M program budget, a mid-year deadline on the safety-critical work, phase-2 jobs that must follow their phase 1 in the very next window, and corridor rivals that stay out of the same quarter.

**Why Frontier.** The **time-phased showcase**: *when* carries as much value as *whether*. Deferring a job inside the year costs risk reduction at that asset's own deterioration rate, while the disruption of taking it out swings with the season and the asset class — so 73% of the later windows survive a same-job comparison against their own job's earliest window (62 of 85) — live options rather than strictly worse copies — and the schedule is a choice, not a formality. The coupling makes those choices cascade: at the certified best-risk corner, `HY-03` (Vantry draft tube repair phase 1) sits in Q1 and its phase 2 `HY-04` in Q2; pin `HY-03` one quarter later and `HY-04`'s Q2 window dies with it — the staging rule admits only the window right after phase 1 — so `HY-04` slides to Q3 and the best attainable risk reduction falls 7,876 → 7,862 MWh. Lift the staging rule for that one pair and the same pin lands at 7,875 MWh: **13 of those 14 MWh are the knock-on on the dependent job**, not the pinned job's own 23 MWh of decay (the program reshuffles around the rest). A planner's own heuristic misses all of it: worst-asset-first-do-it-now puts 25 jobs into Q1 against six crews and spends $124.1M against a $76M budget, and the repaired version (deadline work first, then worst-first into the earliest open quarter) lands on a legal program that **three certified programs beat outright** — the best of them avoiding 470 MWh more at $2.5M less disruption and $2.6M less capex.

**What ships here** — the raw inputs (step 1), the canonical model they frame into, and pre-solved results:

- **`projects.csv`**: the job list a maintenance planner already has — 30 rows, no quarters in it. Risk reduction if done in Q1, the asset's per-quarter decay rate, base capex and disruption, the safety-critical flag, the phase-1 predecessor, and the corridor.
- **`quarter_factors.csv`**: the seasonal table — what each quarter does to disruption cost and capex, by asset class.
- **`problem.json`**: 3 objectives (RiskReduction maximize, DisruptionCost / Capex minimize, all `sum`), binary approach, and 69 constraints over the (job, quarter) pairs (the $76M bound, four per-quarter crew caps, one at-most-once rule per job, the June-30 deadline on the eight safety-critical jobs, 15 staging dependencies, and 11 corridor exclusions), plus the two futures — each `constraint_overrides` list **restates the full constraint set** (overrides replace, not merge) with one quarter's crew cap moved, and the summer future re-prices every Q3 pair on top of that.
- **`scores.json`**: the 115 (job, quarter) options the framing derives, each scored on the three objectives — the job's base values put through its own decay rate and its class's seasonal factors.
- **`solutions.json`**: the exploratory NSGA `run`, the exact-MILP `exact_run` overlay (HiGHS, zero-gap), and the per-future `scenario_run`.

## The runbook

1. **Frame it from the raw inputs** — paste this ask, together with `projects.csv` and `quarter_factors.csv`, into a fresh session:

   > We're locking next year's maintenance outage program and I'd like help scheduling it.
   > `projects.csv` is the job list: 30 jobs, each with the risk reduction we'd book if we
   > run it in Q1 (MWh of unserved energy avoided), how much of that we lose per quarter we
   > defer it (the asset's deterioration rate), its base capex ($M), its base disruption
   > cost ($M), whether it's safety-critical, whether it's phase 2 of another job, and which
   > transmission corridor it sits on. `quarter_factors.csv` is the seasonal table: what
   > each quarter does to disruption cost and to capex, by asset class.
   >
   > The decision is which jobs we run and which quarter each one runs in — a job we can't
   > place waits for next year. We want the most total risk reduction, at the least total
   > disruption cost and the least total capex.
   >
   > Hard rules:
   > - We can field six outage crews in a quarter, so at most 6 outages a quarter.
   > - The program stays within the $76M budget.
   > - A job runs at most once — one quarter, or not at all.
   > - The safety-critical jobs all have to run, and be in the ground by June 30.
   > - A phase-2 job runs in the quarter right after its phase 1 — the asset can't sit in
   >   its interim configuration any longer than that.
   > - Two jobs on the same corridor stay out of the same quarter.
   >
   > Two futures to stress-test:
   > - **Summer demand surge** — a hotter summer: every Q3 outage costs 60% more in
   >   replacement power and interruption exposure, and operations releases only 3 crews'
   >   worth of Q3 windows.
   > - **Crew shortage** — spring storm restoration pulls line crews off planned work: Q2
   >   fields 4 crews instead of six. Everything else holds.

   Framing that input (`model create` + `model update`) lands on exactly this problem: the agent turns the job list into one option per job per quarter it could run in — `TX-01 Q1`, `TX-01 Q2`, … — scores each from the job's own decay rate and its class's seasonal factors, and encodes the rules against those pairs (a phase-2 job has no Q1 option at all; its phase 1 has to come first). The ask plus the two CSVs reconstruct `problem.json` and `scores.json` verbatim (guarded by `tests/test_upstream_kits.py`). `model load source="outage_program_scheduling"` is the shortcut: it skips framing and restores the pre-solved runs too.

2. *“Which jobs make next year's program, and which quarter does each one run in? Show me the real risk/disruption/spend choices.”*
   `solve run` → `explore tradeoffs`: the frontier over 2^115 candidate programs. The shipped run spans the smallest legal program — the eight safety-critical jobs and nothing else, 3,853 MWh avoided at $23.1M disruption and $37.9M capex — up to an 18-outage program at 7,369 MWh, $50.5M and $73.7M that fills Q1 and Q2 and pushes four jobs into the expensive summer. The clearest read of what the calendar alone is worth sits at the bottom of that range: **that same eight-job set holds three separate places on the frontier, differing only in quarters** — 3,853 MWh at $23.1M, 3,958 MWh at $25.2M, and 4,061 MWh at $28.0M. Identical work, four jobs pulled from Q2 into Q1: +208 MWh of avoided unserved energy for $4.9M more disruption.

3. *“Keep the front-loaded program and the balanced one as finalists. How much should I trust them?”*
   `explore curate` per pick → `solve solver="highs" exact=true` → `explore certify`: the zero-gap MILP overlay. On the shipped runs it dominates **15 of the 40** heuristic programs (38% — whole-program dominance, each heuristic program measured against every certified one), reclaims 11% of the covered tradeoff volume, holds the invariant (NSGA dominates no exact point), and sharpens the risk corner from 7,369 to **7,876 MWh (+507)**. Both finalists come back `dominated` with a named certified counterpart — the standard next beat is re-curating each counterpart (`explore curate` on the certified point that beats it):
   - *front-loaded* (7,369.3 MWh / $50.46M / $73.68M) → beaten by a certified 18-outage program at **7,508.7 MWh / $50.37M / $72.39M** — +139 MWh for $1.29M *less* capex.
   - *balanced* (5,579.4 MWh / $35.46M / $55.06M) → beaten by a certified 13-outage program at **5,696.0 MWh / $34.64M / $51.10M** — +117 MWh, $0.82M less disruption, $3.96M less capex.

4. *“Which of these survive a hot summer or a crew shortage in Q2?”*
   `solve run_scenarios` → `explore scenario_results` + `explore scenario_frontiers`: the program re-solved per future. Under the summer surge (Q3 re-priced +60%, Q3 crews cut to three) 36 of the 40 base programs stay legal — the front-loaded finalist is one of the four that break, and it re-prices from $50.5M to **$56.0M** of disruption; the best attainable risk reduction falls to 7,819 MWh. The Q2 crew shortage is the harsher future: only **5 of 40** base programs survive it, and the ceiling falls 7,876 → **7,738 MWh**.

5. *“Which quarter is the bottleneck, what would one more crew there buy, and can you guarantee the safety work lands before the deadline whatever program we pick?”*
   `explore sensitivity` → `explore audit` → `explore composition`. Integer selections carry no solver duals, so sensitivity returns the frontier-inferred binding read: the Q1 and Q2 crew caps sit at their limit in **86%** of frontier programs, Q3 in 16%, Q4 in none of them, and the $76M budget in 18%. That ranks the year but leaves Q1 and Q2 tied, so take the rate as a direction and confirm it by re-solve — which splits them cleanly: a **seventh crew in Q1 is worth +101 MWh** (7,876 → 7,977), the same crew in Q2 +52, in Q3 +9, and in Q4 nothing at all. Then the guarantee, over **every** feasible program rather than the frontier: `[no safety-critical work in H2] AND [Ivyloch phase 1 always runs in Q1] AND [at least 2 of the 7 deadline jobs with a Q1 window run in Q1]` comes back `holds`. All three are emergent — the model says where the safety work *must* go, never where it may not, and the Ivyloch pin falls out of the staging rule meeting the deadline (phase 2's only H1 window is Q2, so phase 1 has nowhere but Q1). Tighten the Q1 floor to 3 and the audit returns a concrete counterexample: an 8-outage program carrying exactly 2. `explore composition` shows the same structure from the frontier side — `SB-02 Q1` and `SB-03 Q2` appear in 100% of the shipped run's programs, the staged pair moving as one unit, which is where a deferral turns into a cascade. (SB-03 wears both labels here: it is one of the eight safety-critical jobs *and* the phase-2 half of the staged pair. Staging strips its Q1 window, which is why the audit's floor counts 7 deadline jobs rather than 8 — and its own June-30 deadline is met in Q2, the only H1 window staging leaves it.)

6. *“Write the program up for the reliability review.”*
   `explore curated format="markdown"`: the handoff table, quarter by quarter.

**Encoding note.** The user talks about jobs and quarters; the model works in (job, quarter) pairs, and the translation is the agent's job. It buys the whole time dimension inside the standard binary MILP shape: deferral value decay and seasonality become scores, "at most one quarter" becomes a per-job group limit, per-quarter crew capacity becomes a group limit over that quarter's pairs, and precedence becomes dependencies between the pairs. What the vocabulary expresses exactly is *consecutive-window* staging ("phase 2 in the quarter right after phase 1"), which is why the phase rule reads that way; a looser "any earlier quarter" precedence would need a disjunction over windows that a single dependency doesn't carry.

**Scale note.** Thirty jobs stay narratable while 2^115 (job, quarter) programs is far past hand-scheduling, and the per-option dominance advisory is worth reading with the scope of each count in mind. The advisory's scope is **global** — each pair against every other option, cross-job comparisons included — and it flags 82 of the 115 pairs (its echo lists the first 20). Restricted to **same-job** dominance — a pair beaten by another window of its own job — the count is 58; the overview's 73% reads the narrowest cut of all from the survivor side — 62 of 85 later windows clear their job's earliest window alone. And the flag is advice about an option, not a verdict on a program: 36 of the 82 still appear in certified-optimal programs — with six crews a quarter, a "worse" window is often the only window. For the same binary MILP arc without the time dimension see [`capital_project_selection_300`](../capital_project_selection_300/); for dependencies and envelopes as scenarios, [`interconnection_approvals`](../interconnection_approvals/).
