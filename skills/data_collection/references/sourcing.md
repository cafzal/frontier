# Data Collection — Sourcing Depth

Load when you are actively *researching* scores from external sources (pricing pages, benchmarks, PDFs, databases) rather than taking numbers the user already has. The core (`data_collection` SKILL.md) carries the always-apply judgment — anchoring, batching, source ranking, conflict resolution, and the *Score Provenance* principle; this is the situational detail for messy extraction. Fetch with `get_skill('data_collection', section='Extraction Tiers')`.

## Extraction Tiers

A source rarely states "this option scores X on this objective" outright. Degrade gracefully rather than returning an empty cell — and let the tier you land on set the confidence that rides with the score (the *Score Provenance* principle in the core):

- **Exact** — the value for this option–objective pair is stated directly. → high confidence.
- **Flexible naming** — the option or objective appears under an alias, abbreviation, or parent-product name; match on meaning, not the literal string. → high–medium.
- **Contextual** — not stated, but derivable from what is (a per-unit figure × a quantity, a stated ratio). → medium; show the derivation.
- **Option-level** — only a general figure about the option exists, not tied to this objective. → low–medium; flag it.
- **Domain proxy** — only a category average or a sibling benchmark applies. → low; a placeholder to refine, not an answer.

A low-confidence estimate entered now beats a blank that blocks the whole run — but it must carry its confidence so later `explore sensitivity` knows where the soft spots are. Return nothing for a cell only when no tier yields anything after a real look. This is graceful degradation, not keyword matching: judge which tier a source actually supports for *this* pair, in any domain.

### Worked example

Objective *Monthly cost* (minimize), option *Vendor B*. The pricing page lists "Team plan — $40/seat/mo" and the problem assumes 10 seats; no line says "Vendor B monthly cost." That's a **contextual** extraction: $40 × 10 = $400/mo, medium confidence, source = the pricing URL. Enter 400, record the per-seat basis and seat count in `context`, and — because cost is a high-variance objective here — mark it as a candidate to revisit with `explore sensitivity` once the frontier shows whether cost is driving the tradeoff. The number, its basis, and its confidence travel together; that is what makes "why is Vendor B a 400?" answerable later.
