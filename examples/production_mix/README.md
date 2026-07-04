# Production mix allocation

Allocate a fixed plant capacity across 10 products on three lines, trading margin, throughput, and sustainability, with no product above 30% of capacity and at most two SKUs per line, stress-tested across two scenarios. Three purely linear objectives over a continuous allocation make this an exact multi-objective LP: a full pass of the Frontier workflow, from the frontier through scenario stress tests to the exact certify and solver duals.

- **`problem.json`**: 3 objectives (all maximize), proportional approach, a 30% per-product cap, three per-line `group_limit`s (≤2 active SKUs each), and two scenarios (`input_cost_spike`, `capacity_crunch`).
- **`scores.json`**: the 10 products scored on margin ($/unit), throughput (k units/wk), and sustainability (0–10).
- **`solutions.json`**: the exploratory NSGA `run`, the exact-LP `exact_run` overlay (HiGHS) with solver-exact duals per point, and the per-scenario `scenario_run`.

Load with `model load source="production_mix"`, then drive it the way a user would — one ask per phase, with the tools that fire and what to expect:

## The workflow

1. *“How should we load the plant across these ten products? Show me the margin/throughput/sustainability tradeoffs — and how they hold up under the stress scenarios.”*
   `solve run` + `solve run_scenarios` → `explore tradeoffs`: the frontier plus `input_cost_spike` and `capacity_crunch` scenarios — the knees where the exchange rate jumps, and how the achievable frontier *shrinks* under the crunch.
2. *“Keep the balanced plan and the max-margin one. Are these actually optimal?”*
   `explore curate` per pick → `solve solver="highs"` → `explore certify`: the exact LP overlay audits the heuristic frontier (all three per-line limits bind).
3. *“Where should we invest next — what's binding, and which product just missed the cut?”*
   `explore sensitivity`: the Throughput floor as the binding lever (~0.5 margin per unit, rising into diminishing returns), Gear Sets the closest near-miss, Aerospace / Optics / Fasteners pinned at the 30% cap, and the line-limited products filtered as structural exclusions. (Values as read at the balanced anchor of the shipped exact overlay — duals are anchor-specific, so expect them to shift at another frontier point or after a re-solve.)
4. *“Write the loading plan up for the ops review.”*
   `explore curated format="markdown"`: the handoff table.

The interior, near-zero-reduced-cost SKU (here Pump Assemblies) is the swing that sets the marginal price; Frontier infers it from the duals today, and an exposed solver basis status would name it directly. The exact LP path runs on either backend (HiGHS on CPU, cuOpt on GPU), so the same shadow prices and reduced costs come back on GPU. For the two-objective LP see [`budget_allocation`](../budget_allocation/); for the mean-variance QP, [`investment_portfolio`](../investment_portfolio/).
