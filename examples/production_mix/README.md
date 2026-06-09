# Production mix allocation

Allocate a fixed plant capacity across 10 products on three lines, trading margin, throughput, and sustainability, with no product above 30% of capacity and at most two SKUs per line. Three purely linear objectives over a continuous allocation make this an exact multi-objective LP: a compact end-to-end run of the full Frontier workflow on a richer constraint set.

- **`problem.json`**: 3 objectives (all maximize), proportional approach, a 30% per-product cap, and three per-line `group_limit`s (≤2 active SKUs each).
- **`scores.json`**: the 10 products scored on margin ($/unit), throughput (k units/wk), and sustainability (0–10).
- **`solutions.json`**: the exploratory NSGA `run` plus the exact-LP `exact_run` overlay (HiGHS), with solver-exact duals per point.

Load with `model load source="production_mix"`, or paste this to an agent connected to Frontier:

> Allocate plant capacity across the products in scores.json to maximize margin, throughput, and sustainability, with no product over 30% and at most two SKUs per line. Explore the tradeoffs, solve it exactly (solver=highs), certify the exact overlay, and read the duals for where to invest and what's close.

## The workflow

1. **Explore** (`solve run`): a margin/throughput/sustainability frontier with per-objective extremes, a balanced plan, and the knees where the exchange rate jumps.
2. **Certify** (`solve solver="highs"`, then `explore certify`): the exact LP overlay audits the heuristic frontier for dominated points and corner sharpening; all three per-line limits bind.
3. **Examine sensitivity** (`explore sensitivity`): solver-exact duals at the balanced plan. The Throughput floor is the binding lever (each extra unit costs ~0.44 of margin, rising along the frontier into diminishing returns); Gear Sets is the closest near-miss; Aerospace, Optics, and Fasteners sit at the 30% cap and would take more if allowed; products held out by the line limit are filtered as structural exclusions.
4. **Decide** (`explore curate`): pin a few plans and choose on the tradeoffs.

The interior, near-zero-reduced-cost SKU (here Pump Assemblies) is the swing that sets the marginal price; Frontier infers it from the duals today, and an exposed solver basis status would name it directly. For the two-objective LP see [`budget_allocation`](../budget_allocation/); for the mean-variance QP, [`investment_portfolio`](../investment_portfolio/).
