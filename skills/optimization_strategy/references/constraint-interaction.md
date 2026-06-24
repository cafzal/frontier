# Optimization Strategy — Constraint Interaction

Load when a solve comes back infeasible or thin and relaxing a single constraint doesn't explain it (`get_skill('optimization_strategy', section='Constraint Interaction Patterns')`). The core's *Infeasibility Response* and *Binding Constraint Detection* handle the single-constraint case; constraints also interact, and the interactions have distinct fixes.

## Constraint Interaction Patterns

| Pattern | How it shows up | What to do |
|---|---|---|
| **Contradictory** | Zero feasible solutions, and no single constraint is obviously the cause | Find the conflicting *pair* — a force_include against an exclusion_pair, an objective bound against a cardinality floor. `explore audit` (no property) confirms infeasibility exactly, and its witness shows what *is* reachable. |
| **Cascading tightness** | Relaxing the tightest constraint helps less than expected — the frontier barely widens | Another constraint was binding underneath. Relax and **re-solve between each step**, not all at once, so you see which limit actually governs and stop once the frontier opens. |
| **Redundant** | Removing a constraint doesn't change the frontier at all | One constraint already implied the other. Drop the redundant one to keep the model legible — it isn't doing work, and it hides which limit is real. |
| **Symmetry** | Interchangeable options get systematically different treatment | A constraint names specific options where the intent was a group. Check whether the asymmetry is deliberate; if not, recast as a `group_limit`. |

The headline is **cascading tightness**. When the user asks "why didn't relaxing the budget help?", the usual answer is that cardinality (or a group cap) was binding underneath — so relax iteratively, re-solving each step, rather than expecting one move to open the space. Which relaxation is worth most is a separate read: `binding_analysis` (heuristic/MILP) or exact `sensitivity` (continuous), narrated via `frontier://skills/solution_interpreter`. Pattern-match on the *structure* of the interaction, not on specific constraint names — the same four shapes recur across domains.
