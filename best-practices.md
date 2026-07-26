# Frontier — Skill, Prompt & MCP Design Best Practices

A living reference for designing skills, tool descriptions, and agent instructions in this project. Seeded from [Anthropic's official prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) and refined through iterative auditing of Frontier's skill files.

**Related docs:** [`architecture.md`](architecture.md) — system architecture, tool/skill reference, data flow | [`README.md`](README.md) — user setup and usage guide

**Using this guide (by lifecycle phase)** — pull up the part that applies at each phase instead of re-reading the whole file:
- **Designing** a new skill or MCP tool → §1 (Skill File Design), §3 (MCP Tool Description Design), §4 (Context Engineering — start with the **model floor**, which sets the guidance budget), §5 (Division of Labor — which side of the code↔LLM boundary each piece belongs on)
- **Developing** prompts, tool descriptions, or skill content → §2 (Prompt Best Practices), §3 schema-expressiveness rule, §5 payload-English cap
- **Reviewing** a skill or prompt change → §1 agent-usability criteria + conciseness; §4 model floor ("does this survive a weak model?") and the one-teaching-one-layer rule; verify cross-references and MECE boundaries; §5 gate rule (every scaffold ships with its regression gate)
- **Testing** a new feature → §1 safety patterns (confirmation gates, validation loops)

---

## 1. Skill File Design

Skills are MCP resources (markdown files) that provide contextual judgment the agent consults at different stages of a workflow. They are not tool documentation — they guide *when* and *why*, while tool descriptions handle *how*.

### Structure

Every skill file should follow this structure:

```
# Skill Name
*One-sentence role definition.*

## [Context / Framing Section]
Why this skill exists and what mindset to adopt.

## Core Judgment
The critical principles that shape every interaction. Get these right first.

## [Situational Sections]
Guidance that applies in specific circumstances.

## Activation
When to use this skill (stage of workflow).

## Guardrails
Positive guidance with reasoning — what to do and why.
```

### Principles

**Role definition matters.** Even a single sentence ("You are a decision analyst") focuses behavior and tone. (Source: [Anthropic — Give Claude a role](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role))

**Separate judgment from API details.** Skills guide contextual reasoning — when to use scenarios, why to set reference points, how to interpret results. Tool parameter documentation belongs in tool descriptions. Skills should say "Define scenarios via `model update` with `scenario_config`" and stop — not list every parameter.

**Priority hierarchy.** In long prompts, everything competes for attention. Mark critical principles explicitly ("Core Judgment") and separate them from refinements. The LLM will weight sections by apparent importance. (Source: [Anthropic — Long context prompting](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips))

**Cross-reference, don't duplicate.** When two skills need the same concept (e.g., aggregation modes), define it once in a canonical location and cross-reference: `(See aggregation modes in frontier://skills/problem_framing.)` Duplication wastes tokens and risks inconsistency.

**Consistent section names.** Use the same heading names across files (`## Core Judgment`, `## Activation`, `## Guardrails`) so the LLM builds a structural expectation.

### Agent Usability

The ultimate quality gate for any skill. If an agent can't discover, navigate, adapt, and execute from the skill alone, the skill isn't done.

- **Discovery:** Does the skill's description trigger for realistic user tasks (and not trigger for unrelated ones)?
- **Navigation:** Can an agent find the right section in 1–2 lookups, not wandering?
- **Pattern adaptation:** Given a novel problem, can an agent locate a relevant example and adapt it without hallucinating?
- **Self-sufficiency:** Can an agent go from skill content to working output without external docs or guessing?
- **Negative test:** Does the skill clearly redirect when the task is out of scope?

### Progressive Disclosure

Manage context window budget by layering information:

- **SKILL.md** loads on trigger — keep it lean (under 2,000 words / 500 lines). Contains judgment and workflow.
- **references/** loads only when explicitly read. Contains schemas, lookup tables, detailed examples.
- References should be one level deep from SKILL.md (no deep nesting).

### Conciseness

- **No explaining the obvious:** Omit what the model already knows (general concepts, standard libraries). Every token should earn its place.
- **Concise over exhaustive:** Stepwise guidance with a working example beats encyclopedic coverage. If content covers every edge case, most are better left to model judgment.
- **Defaults over menus:** When multiple approaches apply, pick one as default with a brief escape hatch — not equal-weight lists.

### Safety Patterns

- **Confirmation gates:** For destructive or external-facing actions, pause and show proposed changes before executing.
- **Validation loops:** For tasks with verifiable output, instruct the agent to validate its own work before moving on (run → fix → re-validate).

---

## 2. Prompt Best Practices (Anthropic)

Key principles from Anthropic's official guides, applied to this project.

### Tell Claude what to do, not what not to do

**Instead of:** "Don't say 'the best solution'"
**Write:** "Present tradeoffs, never 'the best solution' — every Pareto solution is optimal at its particular tradeoff."

Positive instructions with reasoning are more reliably followed than bare prohibitions. (Source: [Anthropic — Control the format of responses](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#control-the-format-of-responses))

### Add context / motivation

Explain *why* an instruction matters. The LLM generalizes from the reasoning, not just the rule:

**Instead of:** "Score matrix must be 100% before solving"
**Write:** "Score matrix must be 100% before solving — the optimizer cannot evaluate tradeoffs with missing values, so every gap blocks the entire run."

(Source: [Anthropic — Add context to improve performance](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#add-context-to-improve-performance))

### Prefer general instructions over prescriptive steps

The LLM's reasoning frequently exceeds what a hand-written step-by-step plan would produce. Give the principle and let the model figure out the execution:

**Instead of:** 30 lines of rules for when to use scatter plots vs parallel coordinates
**Write:** "Choose the visualization that best reveals the tradeoff structure. Scatter plots for pairwise comparisons, parallel coordinates for multi-objective views, tables when there are few solutions."

(Source: [Anthropic — Thinking and reasoning](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#leverage-thinking--interleaved-thinking-capabilities))

**Scope: this holds for judgment calls, not for workflow obligations.** A model's reasoning reliably fills in *how* to do something it has decided to do; a step it never considered stays missing. When server.py's CERTIFY line omitted linear-allocation LP, a weak model skipped certify entirely on that shape while capable models certified unprompted. When a beat is something the workflow *owes the user* rather than a choice about execution, name it explicitly and name every shape it applies to — see §4, model floor.

### Avoid keyword matching and lookup tables

LLMs can classify intent from natural language. Teaching keyword triggers ("user says X → do Y") is fragile and narrows generalization. Instead, teach the *classification principle*:

**Instead of:** `| "Pick N", "Choose a few" | Cardinality |`
**Write:** "When a user describes a limit on how many options are selected, that's a cardinality constraint."

### Avoid domain-specific anchoring

Listing domain examples (Investment → proportional, Hiring → binary) creates anchoring bias. Users in unlisted domains get worse results. Instead, teach the *reasoning heuristic*:

**Instead of:** "Investment/portfolio → proportional. Budget allocation → proportional."
**Write:** "Does the quantity assigned to each option matter, or is it a yes/no decision? If swapping allocations produces a meaningfully different outcome, use proportional."

Domain-agnostic examples are fine for illustrating concepts. Domain-specific lookup tables are not.

### Use generic placeholders in examples

When examples are for illustrating a *pattern* (not a specific domain), use "[Option A]", "[objective]" etc. rather than "SSO", "Revenue", "Engineering Effort". Domain-specific names anchor the LLM to that domain.

---

## 3. MCP Tool Description Design

Tool descriptions are the primary interface between the agent and the system. They should be precise, complete, and complementary to skills.

### Responsibility split: Tools vs Skills

| Aspect | Tool description | Skill file |
|--------|-----------------|------------|
| **What parameters exist** | Yes | No — cross-reference |
| **Valid parameter values** | Yes (types, enums, ranges) | No |
| **When to use this action** | Brief heuristic | Detailed judgment |
| **Why / strategic reasoning** | No | Yes |
| **Error interpretation** | Brief | Detailed (infeasibility, diagnostics) |

### Principles

**Let the schema carry the instruction.** Before writing prose about how to use a parameter, ask whether a parameter name, an enum, or a required field could carry it instead. An enum of `"sum"|"avg"|"min"|"max"|"quadratic"` plus one line per value at the point of use teaches more reliably than a paragraph elsewhere explaining when to pick each — the enum is attached to the field the model is filling in at the moment it fills it, and it binds structurally rather than by persuasion. This is the highest-leverage portability lever the surface has: schema constrains the weakest model and costs the strongest nothing.

The input-side twin of §5's structured classification, applied to what the agent sends rather than what it reads.

Open instance: the aggregation guidance in server.py's pre-create checklist teaches when to pick `sum` vs `avg` vs `quadratic` from the instruction block, far from the `objectives` payload it governs. It belongs in the `objectives` Field description, on the parameter at construction time. Relocating across always-visible layers still touches the block every model reads, so it ships with the §4 behavioral gate — the answer-key eval before vs after, covering the rate-like-objective-as-`sum` trap.

**Be explicit about parameter semantics.** Don't assume the LLM knows that `scores` uses merge semantics while `objectives` uses full replacement. State it.

**Document side effects.** If an action marks results stale, clears cached data, or archives a run, say so in the tool description. The agent can't reason about what it doesn't know.

**Use the description to prevent common mistakes.** If users frequently pass wrong parameter formats, add a brief note. Tool descriptions are read every time; skills are read when prompted.

**Exception — bulky, rarely-needed schemas.** When full JSON shapes would bloat an always-visible tool description, put them in a skill's `references/` fetched per section (e.g. `problem_framing/references/schemas.md`) and point to them from both layers — the §4 token-budget rule outranks the responsibility-split table for content that is large and needed only at payload-construction time.

---

## 4. Context Engineering

How skills, tool descriptions, and server instructions work together as a system.

### The three layers

1. **Server instructions** (system prompt in MCP server): brief, high-level workflow guidance. Points to skills.
2. **Tool descriptions** (per-tool docstrings): parameter documentation, API semantics, side effects.
3. **Skills** (MCP resources): contextual judgment, principles, guardrails. Read on-demand by the agent.

### The model floor

Token budget answers *how much*. The model floor answers *how much for whom*.

Guidance a strong model would have supplied on its own is dead weight to it — a sound case for cutting hard, given a known reader. Frontier has no such guarantee: the agent layer runs on whatever model the user brought, so **the weakest model that must succeed sets the guidance floor, whatever the strongest manages on its own.** A capable model doing the right thing tells you about that model; the surface earns its verdict from the weakest reader.

The resolution is placement, not volume — **relocation** over deletion, moving guidance out of what every model reads on every call and into the layer a strong model skips and a weak one gets *pointed at*:

| Content | Where it belongs | Why |
|---|---|---|
| A workflow obligation, and every shape it applies to | Always-visible: server instructions, tool docstrings, **the schema itself** (§3) | Binds every model; a weak one won't infer the beat |
| A fact the engine can decide | Deterministic code (§5) | Redundant for strong models, load-bearing for weak ones |
| The reasoning behind a rule, and its situational depth | Skill `references/`, fetched by `get_skill(name, section=)` | Strong models skip it; weak ones arrive via `guidance_pointer` |
| Anything derivable from the code or the response itself | Nowhere — cut it | Costs every model, teaches none |

The machinery for the third row already exists: per-section `get_skill`, the injection throttle, `guidance_pointer`. Relocation is what lets the floor rise without the ceiling paying for it.

### One teaching, one layer

Duplication across always-visible layers is the failure a rising floor invites. A weak model reading the same teaching twice acts on it once, while the second copy doubles the surface where terminology drifts. The rule:

**The always-visible layer states the rule in one line, at the point of use. The skill carries the reasoning and the how. Never the same teaching at paragraph length in both.**

Live instances to hold to this: "never say 'best'" (server instructions + `solution_interpreter`) is correct — a one-line rule up top, the reasoning in the skill. The pre-create approach/aggregation checklist (server instructions + `problem_framing` §Approach Selection and §Aggregation) still owes the fix: it carries a paragraph of the same teaching in both layers, and §3 names the remedy.

### Design principles

**Minimize total token budget.** Every token of instruction competes for attention. Be concise. If something can be derived from the code, don't document it in a skill. If something is said once in a tool description, don't repeat it in a skill.

**Point at the artifact, don't describe it.** Prefer a reference to real code, a test, an `examples/` bundle, or a schema over prose restating what it contains — the artifact can't drift from itself, and the prose always will. Frontier's strongest instance already exists: an example's runbook *is* its user-test script, so the bundle is simultaneously the spec, the demo, and the eval. When a skill section teaches a pattern a bundle demonstrates, point at the bundle and say when to open it.

**Put critical instructions in the layer that's always visible.** Server instructions and tool descriptions are always in context. Skills are read on-demand. The most important guardrails should be in the always-visible layer, with skills providing depth.

**Skills should be stage-aware.** Each skill has an Activation section that says *when* to use it. This lets the agent load the right context at the right time rather than flooding the context with all guidance at once.

**Cross-reference convention.** Use `(See [section name] in frontier://skills/[skill_name].)` for cross-references between skills. Use "See the `[tool]` tool description for API details" to point from skills to tools.

---

## 5. Division of Labor — Deterministic Code vs LLM Judgment

Frontier scaffolds a probabilistic agent with deterministic Python. The strategy is deliberate, and it has a boundary; when adding a feature, decide which side each piece belongs on before writing it.

**Deterministic code earns its place in exactly three areas:**

1. **Math** — compute what the LLM can't: the solve itself, hypervolume, regret, duals, dominance. Never ask the model to estimate what the engine can prove.
2. **Structured classification** — compress raw results into a stable vocabulary of enums and verdicts (`linear_redundant`, `under_covered`, `frontier_inferred`, scale bands, certificate blocks). This vocabulary is what skills teach against ("when you see X, say Y"), what answer-key evals assert on, and what makes claims traceable. It also makes behavior portable across models — classification in code is redundancy for strong models and load-bearing for weak ones. (This is the read-side half of the model floor, §4; §3's schema-expressiveness rule is the write-side half.)
3. **Guidance routing** — the injection throttle, `guidance_pointer`s, and section resolver. *When* to deliver skill content is a state-machine problem (once per phase, re-arm on shape change, point at the section governing this payload), not a judgment call.

**The boundary rule: code decides what is true; skills decide how to say it and when it matters; the model decides what the user meant.** Semantic mapping of user intent stays with the model (see §2 — no keyword tables); narration and judgment stay in skills; facts, labels, and state stay in code.

**No deterministic scaffold ships without its gate.** Deterministic scaffolding is brittle by nature — hardcoded section titles, payload contracts, workflow beats. That brittleness is affordable only when each scaffold has a regression gate: the resolver-integrity test guards every pointer target and heading, wire-level tests guard payload contracts, answer-key evals guard workflow beats. A scaffold without a gate is rot waiting to happen; add the gate in the same PR.

**Cap payload English — a caption, not a copy.** Hand-written English inside tool responses (`note` / `next_steps` / `recommendation` / `hint`) is the bloat- and drift-prone stratum: it double-bookkeeps with skills, it's invisible to prose review, and it's where terminology drift happens. Two kinds earn their place:
- **Epistemic captions** — a per-response fact the skill can't know: a truncation ("table capped at N"), an absence ("no adjacent-cardinality solutions to estimate from"), an artifact label ("rounding of the continuous optimum, not an infeasible plan").
- **Routing pointers** — the next tool call or the skill section that governs the read.

If a payload string starts *explaining* — teaching semantics, coaching narration, restating what a skill section says — it belongs in that skill section, with the payload shrunk to a pointer. The feature template: **new structured fields + one skill section + a pointer wiring it + at most a sentence of payload English.** When a `guidance_pointer` on the same response already names the covering section, the prose it covers is redundant — trim it.

---

## 6. Patterns Applied in This Project

Patterns discovered during skill file auditing and refactoring:

| Pattern | Before | After |
|---------|--------|-------|
| Keyword trigger table | `"Pick N" → cardinality` | Principle: "limit on selection count → cardinality" |
| Domain lookup | `Investment → proportional` | Heuristic: "does quantity matter?" |
| Negative anti-patterns | "Don't say 'best'" | "Present tradeoffs — every solution is optimal at its tradeoff" |
| Missing motivation | "Score matrix must be 100%" | "...because the optimizer can't run with gaps" |
| API details in skills | 15 lines of `score_adjustments` parameters | "Define via `model update`. See tool description." |
| Domain-specific examples | "SSO, Mobile App, Analytics Dashboard" | "[Option A], [Option B]" or pattern description |
| Redundancy across files | Aggregation explained in 3 files | Canonical in `problem_framing`, cross-referenced |
| Flat priority | All sections at same heading level | "Core Judgment" (critical) vs "Presentation Refinements" |
| Prose where schema would bind | A paragraph on when to pick each aggregation | Enum + one line per value on the parameter (§3) |
| Depth in the always-visible layer | Situational guidance in server instructions | `references/` + a `guidance_pointer` to the section (§4) |
| Same teaching in two visible layers | Approach/aggregation in instructions *and* skill | One-line rule up top, reasoning in the skill (§4) |
| Prose describing an artifact | Skill section restating a bundle's shape | Point at the `examples/` bundle, say when to open it |

---

## Sources

- [Anthropic — The new rules of context engineering for Claude 5 generation models](https://claude.com/blog/the-new-rules-of-context-engineering-for-claude-5-generation-models)
- [Anthropic — Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [Anthropic — System prompts](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts)
- [Anthropic — Long context prompting tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips)
- [Anthropic — Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)
