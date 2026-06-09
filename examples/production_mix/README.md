# Production mix allocation

Allocate a fixed plant capacity across 10 products on three lines, trading margin, throughput, and sustainability, with no product above 30% of capacity and at most two SKUs per line, stress-tested across two scenarios. Three purely linear objectives over a continuous allocation make this an exact multi-objective LP: a full pass of the Frontier workflow, from the frontier through scenario stress tests to the exact certify and solver duals.

- **`problem.json`**: 3 objectives (all maximize), proportional approach, a 30% per-product cap, three per-line `group_limit`s (≤2 active SKUs each), and two scenarios (`input_cost_spike`, `capacity_crunch`).
- **`scores.json`**: the 10 products scored on margin ($/unit), throughput (k units/wk), and sustainability (0–10).
- **`solutions.json`**: the exploratory NSGA `run`, the exact-LP `exact_run` overlay (HiGHS) with solver-exact duals per point, and the per-scenario `scenario_run`.

Load with `model load source="production_mix"`, or paste this to an agent connected to Frontier:

> Allocate plant capacity across the products in scores.json to maximize margin, throughput, and sustainability, with no product over 30% and at most two SKUs per line. Explore the tradeoffs across the base case and the stress scenarios, solve it exactly (solver=highs), certify the exact overlay, and read the duals for where to invest and what's close.

## The workflow

1. **Solve** (`solve run`, plus `solve run_scenarios` for the stress scenarios): the optimizer produces the margin/throughput/sustainability frontier and a per-scenario frontier for `input_cost_spike` (a materials/energy cost spike squeezes the thin-margin commodity line) and `capacity_crunch` (the per-product cap tightens 30% to 20%, forcing a broader mix).
2. **Explore the tradeoffs** (`explore tradeoffs`): the per-objective extremes, a balanced plan, the knees where the exchange rate jumps, and how the achievable frontier *shrinks* under the capacity crunch (no product can dominate, so both the margin and throughput corners pull in).
3. **Certify and examine** (`solve solver="highs"` → `explore certify` → `explore sensitivity`): the exact LP overlay audits the heuristic frontier (all three per-line limits bind); the duals at the balanced plan show the Throughput floor as the binding lever (each extra unit costs ~0.44 of margin, rising into diminishing returns), Gear Sets as the closest near-miss, Aerospace / Optics / Fasteners pinned at the 30% cap, and the line-limited products filtered as structural exclusions.
4. **Decide** (`explore curate`): pin a few plans and commit on the tradeoffs.

The interior, near-zero-reduced-cost SKU (here Pump Assemblies) is the swing that sets the marginal price; Frontier infers it from the duals today, and an exposed solver basis status would name it directly. The exact LP path runs on either backend (HiGHS on CPU, cuOpt on GPU), so the same shadow prices and reduced costs come back on GPU. For the two-objective LP see [`budget_allocation`](../budget_allocation/); for the mean-variance QP, [`investment_portfolio`](../investment_portfolio/).
