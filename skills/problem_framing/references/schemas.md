# Problem Framing — Schema Reference

JSON shapes for the structured fields passed through `model/create` and `model/update`, factored out of the `frontier://skills/problem_framing` core (they're API shapes, not framing judgment). Fetch when constructing a payload: `get_skill('problem_framing', section='Constraint schemas' | 'Interaction matrix schema' | 'Interaction matrix override schema' | 'Scenario schema')`.

## Constraint schemas

Pass to the `constraints` param as a list of dicts:

```
{"type": "cardinality", "min": <int>, "max": <int>}
{"type": "force_include", "option": "<name>"}
{"type": "force_exclude", "option": "<name>"}
{"type": "objective_bound", "objective": "<name>", "operator": "min"|"max", "value": <float>}
{"type": "exclusion_pair", "option_a": "<name>", "option_b": "<name>"}
{"type": "dependency", "if_option": "<name>", "then_option": "<name>"}
{"type": "group_limit", "options": ["<name>", ...], "max": <int>}  (optional "min": <int> floor — at least that many selected/active from the group; exact-certifiable on binary, NSGA-only on proportional)
{"type": "max_allocation", "max": <int>}  (proportional only: one global cap on every option's allocation)
{"type": "allocation_bound", "option": "<name>", "min": <int>, "max": <int>}  (proportional only: per-option floor/cap in percent — contractual minimums, service floors, per-channel caps; effective cap = min(global, per-option); a floor > 0 force-activates the option and carries a dual on the exact LP/QP path)
```

Every type also accepts an optional `"motivated_by": "<why this rule exists>"` — the user's own words, carried and echoed beside the constraint's cost in `binding_analysis` and in the stakeholder writeup. Fill it whenever the reason isn't recoverable from the rule itself, which is most of the time (see *Post-Solve Constraint Discovery* in the core).

## Interaction matrix schema

For objectives with `aggregation="quadratic"`:

```
{"objective": "<name>", "entries": {"opt_a": {"opt_a": <float>, "opt_b": <float>, ...}, ...}}
```

Entries must be symmetric — send the upper triangle and the mirror is filled for you. For portfolio volatility, entries = covariance matrix.

A base matrix takes the same `mode` as an override below: default `"replace"` sends the full matrix in one call, and **`mode="upsert"` merges the listed cells into the existing base matrix**. Chunk with upsert whenever the full matrix would not fit one tool call — a 72-option matrix is 5,184 cells (2,628 sent as the upper triangle), which lands as roughly 8 calls of one option-block each. This is not an optimization: a write that overruns the output cap is dropped whole, and every retry hits the same wall, so a large matrix entered in one call never lands at all.

## Interaction matrix override schema

For scenario overrides (composable fields):

```
{
  "objective": "<name>",
  "mode": "replace" | "upsert",              # default "replace"
  "entries": {"opt_a": {"opt_b": <float>}},  # replace: full matrix; upsert: only changed cells
  "scale_groups": [{"options": ["a", "b", ...], "factor": <float>}]  # optional; applied after mode
}
```

- `mode="upsert"`: merge the listed cells into the base matrix; symmetry auto-enforced.
- `scale_groups`: multiply off-diagonals within each group by factor. Use for regime shifts (e.g. recession: equity correlations × 1.5). Compose with upsert cells for precise + broad overrides.

## Scenario schema

Pass to `scenario_config.scenarios` as a list of dicts:

```
{
  "name": "<name>",
  "probability": <float 0-1>,                  # optional
  "description": "<text>",                     # optional
  "score_overrides": [{"option", "objective", "value"}],
  "score_adjustments": [{"objective", "multiply"?, "add"?}],
  "constraint_overrides": [<constraint dict>], # full replacement when non-empty
  "interaction_matrix_overrides": [<matrix override dict>]  # see schema above
}
```
