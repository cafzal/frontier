# Runbook — capital project selection (binary MILP path)

Ordered prompts walking FRAME → SCORE → EXPLORE → CURATE → CERTIFY → EXAMINE → DECIDE over this
example, each naming the MCP tool it exercises and the response shape to expect. FRAME and SCORE
are pre-baked in `problem.json` / `scores.json`, so step 1 loads them. The README stays the
narrative overview; this is the paste-and-drive script.

1. **FRAME + SCORE — load the model**
   Prompt: *"Load the capital_project_selection_120 example and summarize what decision it frames."*
   Tool: `model` (action=load, then get/summary).
   Expect: a `problem_id`; 4 objectives (NPV↑, Cost↓, Risk↓, StrategicFit↑), 120 options, the
   budget / category caps / dependencies / exclusions / cardinality constraints echoed back.

2. **Pre-solve check**
   Prompt: *"Is the model ready to solve? Probe feasibility exactly."*
   Tools: `solve` (action=validate), `explore` (action=audit, no property).
   Expect: `ready: true`; audit verdict `feasible` with a concrete witness plan. (If you tighten
   the budget into contradiction, the verdict flips to `no_feasible_plan` and `conflicts` names a
   minimal set of constraints that cannot all hold together — leads to relax, never auto-relaxed.)

3. **EXPLORE — map the frontier**
   Prompt: *"Map the efficient frontier of funding plans."*
   Tools: `solve` (action=run) → `explore` (action=tradeoffs).
   Expect: a frontier `run`; tradeoffs with `objective_ranges`, `extreme_solutions`,
   `balanced_solution`, `inflection_point_candidates` (each with `jump_factor` and a one-line
   `rationale`), `binding_analysis`, and `option_selection` consensus stats.

4. **CURATE — shortlist with names**
   Prompt: *"Pin the balanced plan and the low-risk extreme as finalists."*
   Tool: `explore` (action=curate, per solution).
   Expect: `curated: true` per pick plus a `quality` gate (GOOD / WARNING / DEGENERATE, with the
   triggering check named in plain terms). Flagged finalists stay in the set — the call is yours.

5. **CERTIFY — prove the finalists**
   Prompt: *"Solve it exactly and certify the frontier."*
   Tools: `solve` (solver="highs") → `explore` (action=certify).
   Expect: the exact-MILP overlay in `exact_run`; a certificate with `dominance_audit` (how many
   heuristic points exact strictly beats), `coverage`, the `invariant` (NSGA dominates no exact
   point), `corner_sharpening` per objective, and `quality_gates` flagging any degenerate exact
   optimum.

6. **EXAMINE — why, and what's binding**
   Prompt: *"What's the tightest lever, and what should we stress-test?"*
   Tools: `explore` (action=sensitivity), `explore` (action=audit, with a property).
   Expect: integer selections carry no solver duals, so sensitivity returns the
   `frontier_inferred` binding analysis — plus `suggested_scenarios` seeded from the most binding
   constraints (copy a suggestion's `motivated_by` onto any scenario you create from it). A
   property audit (e.g. an objective bound you hope always holds) returns `holds` (proven over
   every feasible plan), `violated` (with a counterexample witness), or `inconclusive` (with the
   raw solver status — not evidence either way).

7. **DECIDE — the handoff**
   Prompt: *"Export the shortlist and write it up for the review."*
   Tool: `explore` (action=curated, format="markdown").
   Expect: the handoff table with a `quality` column per finalist and a pointer to the
   stakeholder-writeup playbook (context → decided → why → confidence → impact → next steps).
