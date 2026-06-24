# Problem Framing — Pre-Solve Validation

Load right before the first solve, once the model is built (`get_skill('problem_framing', section='Pre-Solve Validation')`). The core's *Formalization Checkpoint* checks that each objective is sound (units, direction, PSD); this checks that the model *as a whole* captures the decision — the gaps a clean solve over a wrong model will hide.

## Pre-Solve Validation

Make one pass over the assembled model against the decision question stored in `context`. A solver is precise about whatever it is given, so a structural omission survives as a confident-looking frontier rather than an error. These are the omissions worth catching first:

- **Unstated goal** — a goal the user named (in the decision question, or in passing) that no objective or constraint encodes. This is the most common silent mis-spec, and it is findable precisely because Frontier stores the question: read it back and confirm every "want" maps to an objective and every "must" to a constraint. *"You said the pick has to be defensible to the board — we have cost and impact, but nothing captures defensibility. Is that an objective, a constraint, or out of scope?"*
- **Orphaned element** — an objective that barely varies across options, or an option that can never be selected under the current constraints. It adds compute and noise without shaping the frontier; drop it, or fix what makes it inert. (For the variance read, see *Score Quality Signals* in `frontier://skills/data_collection`.)
- **Redundant constraint** — two constraints where one already implies the other (a cardinality ≤ 3 alongside a group limit ≤ 5 over a 4-option group). Harmless to the math, but a sign the user's intent is not yet crisp; consolidate and confirm which one they actually mean.
- **Over-tight stack** — constraints that are each reasonable but jointly leave little or no feasible space. Confirm this exactly *before* spending a solve with `explore audit` (no property = feasibility probe). For the fix, see *Infeasibility Response* in `frontier://skills/optimization_strategy`.

This is a confirm-the-right-problem pass, not a solver gate — surface what looks off as a question and let the user decide; over-correcting silently is its own failure. It is the pre-solve mirror of *Status Literacy* (`frontier://skills/optimization_strategy`): here you confirm the model is right going in; there you confirm a thin result coming out is a real finding, not silence dressed as an answer.
