# Production mix allocation

Allocate a fixed plant capacity across 10 products on three lines, trading **margin / throughput / sustainability** — no product over 30% of capacity, at most two SKUs per line. Three purely linear objectives over a continuous allocation make this an **exact multi-objective LP** — a compact end-to-end run of the full Frontier workflow on a richer constraint set.

- **`problem.json`** — 3 objectives (all maximize), proportional approach, a 30% per-product cap, and three per-line `group_limit`s (≤2 active SKUs each).
- **`scores.json`** — the 10 products × (margin $/unit, throughput k-units/wk, sustainability 0–10).
- **`solutions.json`** — the exploratory NSGA `run` + the exact-LP `exact_run` overlay (HiGHS), with solver-exact duals per point.

Load with `model load source="production_mix"`, or paste this to an agent connected to Frontier:

> Allocate plant capacity across the products in scores.json to maximize margin, throughput, and sustainability — no product over 30%, at most two SKUs per line. Explore the tradeoffs, then solve it exactly (solver=highs), certify the exact overlay, and read the duals to tell me where to invest and what's close — not one "best."

## The workflow

- **Explore** (`solve run`) — a margin↔throughput↔sustainability frontier: the per-objective extremes, a balanced plan, and the knees where the exchange rate jumps. A menu, not one answer.
- **Certify** (`solve solver="highs"` → `explore certify`) — the exact LP overlay audits the heuristic frontier: dominated points, corner sharpening, the NSGA-never-dominates invariant. All three per-line limits bind.
- **Sensitivity** (`explore sensitivity`) — solver-exact duals at the balanced plan: the **Throughput floor** is the binding lever (each extra unit costs ~0.44 of margin, rising along the frontier — exact diminishing returns); **Gear Sets** is the closest near-miss; **Aerospace / Optics / Fasteners** sit at the 30% cap (the solver would run more if allowed); products held out by the line limit are filtered as structural exclusions, not near-misses.
- **Decide** (`explore curate`) — pin a few plans and choose on the tradeoffs.

(The interior, near-zero-reduced-cost SKU — here Pump Assemblies — is the swing that sets the marginal price; Frontier infers it from the duals today, and an exposed solver **basis status** would name it directly. For the two-objective LP see [`budget_allocation`](../budget_allocation/); for the mean-variance QP, [`investment_portfolio`](../investment_portfolio/).)
