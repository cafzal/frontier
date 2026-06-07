<p align="center">
  <img src="assets/frontier-logo.png" alt="Frontier" width="160" />
</p>

<p align="center">
  Multi-objective decision optimization toolkit. Use in any MCP client or via the integrated web UI.
</p>

## Summary

Frontier gives AI agents a grounded optimization engine for hard decisions. The agent describes a problem in business terms; Frontier enumerates the full Pareto frontier — every non-dominated solution that balances conflicting objectives under hard constraints — and the agent narrates the tradeoffs back. Evolutionary/approximate (NSGA-II/III via pymoo) and exact (cuOpt and HiGHS) under the hood, exposed as 4 MCP tools (`model`, `solve`, `explore`, `get_skill`). Frontier is the engine; the agent is the interface.

The design is **explainable, governable optimization**: the engine owns the *deterministic guardrails* — hard constraints it never violates, reproducible runs (same inputs + seed → same frontier), dominance filtering, pre-solve validation, and quality gates — while the *human judgment* stays at the two calls that matter: which objectives and constraints define the problem, and which non-dominated solution to commit to. The agent explains every tradeoff (shadow prices, frontier shape, marginal rates, dominance) and never names a "best"; every claim it makes traces back to returned data, so the result is explainable and the decision is auditable line by line. The wedge is combinatorial, constrained, portfolio-like decisions with conflicting objectives.

## Examples

Several [worked examples](examples/) are included for you to learn from and adapt — each ships a ready prompt you can paste into any connected client to reproduce the result. Select results displayed below.

<table>
<tr>
<td width="50%" valign="top">
<img src="assets/example-portfolio.png" alt="ETF efficient frontier with a plain-language tradeoff read" /><br/>
<sub>Investment portfolio</sub>
</td>
<td width="50%" valign="top">
<img src="assets/example-capital.png" alt="Capital project formulation and per-scenario frontiers" /><br/>
<sub>Capital project selection</sub>
</td>
</tr>
</table>

## Purpose

LLMs can reason about tradeoffs conversationally but can't *solve* them — they can't reliably enumerate a combinatorial option space, enforce hard constraints, and produce the actual Pareto frontier. These are the decisions teams used to grind out in spreadsheets, until the spreadsheet hit a complexity wall: too many options, too many interacting constraints, objectives that genuinely conflict. Frontier supplies the missing half — the **LLM translates** the decision into a structured model and narrates the result; a real **optimization solver** does the math neither a spreadsheet nor an LLM can. It fits problems where data can score options, objectives genuinely conflict (no single "best"), and the space is too large and too constrained for intuition.

**By shape, not domain:** any decision that selects a subset from many options (which K of N) or allocates a budget across them (how much of each), balancing conflicting objectives under hard constraints — with data to score the options and a space too large for a spreadsheet or intuition. Pairwise interactions (covariance, audience overlap, correlated risk) make it genuinely nonlinear.

**What Frontier adds beyond an LLM alone:**
- **The full non-dominated frontier** — every Pareto-optimal tradeoff, not a single recommendation or a weighted ranking
- **An optional exact auditor over the frontier** — for subset selection and mean-variance allocation, the agent can overlay an exact inner solve (HiGHS on CPU or NVIDIA cuOpt on GPU, two first-class backends): explore the whole frontier fast, then certify the finalists. Its payoff is *trust and coverage at scale*: confirming each pick is optimal for its tradeoff and catching dominated points the heuristic presented as efficient — not a better answer on small problems, where the heuristic already covers the frontier.
- **Hard constraints, enforced** — 8 constraint types (cardinality, forced include/exclude, objective bounds, exclusion pairs, dependencies, group limits, allocation caps), never violated during search
- **Auditable by construction** — every reported tradeoff traces to returned data (scores, shadow prices, dominance), not a fluent guess; runs are reproducible (same inputs + seed → same frontier; `seed_used` is recorded), so a stakeholder can re-examine the decision line by line
- **Scenario & risk modeling** — independent frontiers per scenario, plus CVaR / worst-case / expected risk per objective
- **Longitudinal state** — problems persist across sessions; curated picks track survival across re-runs

*Why not just ask an agent to write a solver?* You can — for a one-shot problem. Frontier is the turnkey pairing: an LLM translation-and-narration layer over a real solver, grounded (every number computed, not guessed), auditable, and reusable across problems and re-runs — instead of bespoke optimization code rebuilt and re-verified each time.

## Workflow

You drive Frontier by talking to an AI agent — in a coding-agent MCP client or the hosted web chat — in plain language. The agent translates your decision into Frontier's model, runs the solver, and reads the results back. A typical sequence (you describe what you want; the agent picks the tools):

1. **Frame it.** Name the objectives (what to maximize / minimize), the options to choose among, and any hard constraints — plus scenarios if the future is uncertain. *e.g. "We're choosing a CRM for a 10-person startup: maximize features and support, minimize cost; budget under $50k/yr; pick one."*
2. **Score the options.** Hand over the numbers, or let the agent estimate and flag what's shaky. *e.g. "Score these five CRMs on cost and support from their pricing pages."*
3. **Solve.** The agent validates the setup, then runs the optimizer for the Pareto frontier — optionally once per scenario, and on a final run it can audit the frontier against an exact solver where the problem shape supports it (`explore certify`), sharpening the risk corner and flagging any dominated points. *e.g. "Solve it."*
4. **Explore the tradeoffs.** Frontier shape, the extremes, the balanced/knee, the marginal cost of pushing an objective, robustness across scenarios — and curate the picks you like. *e.g. "Show the tradeoffs, recommend a balanced pick, and curate it as 'Lean choice'."*
5. **Iterate.** Tighten a constraint, add a scenario, re-solve, and compare against the previous run. *e.g. "Cap cost at $40k and re-run — what dropped off the frontier?"*

Behind the conversation: four tools — `model` (define), `solve` (optimize), `explore` (navigate results), `get_skill` (workflow guidance) — with skills that auto-inject at each transition, so the agent classifies objectives vs constraints, elicits scores without anchoring bias, and presents tradeoffs without ever naming a single "best."

### Saving & loading

Every problem is auto-persisted in the engine's store (`data/`, keyed by id) — session state you don't manage. Separately, `model save` writes a **named, portable copy** in the [examples](examples/) format, to reload or share by name:

- **`model save problem_id=… save_as="<name>"`** — save to your gitignored `saved/` library (override with `FRONTIER_SAVED_DIR`), bundling the solved frontier when present.
- **`model load source="<name>"`** — rebuild a problem, resolving `saved/` first, then bundled `examples/`; omit `source` to list available names.

## Setup

Two ways to use Frontier:

- **Web UI** — a browser chat shell over the engine, with interactive charts (2D/3D scatter and parallel coordinates that adapt to objective count — a 3-objective frontier can also render as a 2D scatter colored by a chosen objective — per-scenario overlays, and exact-certified points denoted on the frontier once you certify) and curate-straight-from-the-chart. Try the hosted app at **[frontier-ui.onrender.com](https://frontier-ui.onrender.com/)** (password-gated — ask @cafzal for access), or run/deploy your own (requires an API key; see [`ui/`](ui/) and [Deploy your own](#deploy-your-own))
- **MCP client** — connect any MCP-compatible client (Claude Code, Claude Desktop, claude.ai, Cursor, Codex). The hosted beta engine (`https://frontier-592q.onrender.com/sse`) is gated by a token — ask @cafzal for `FRONTIER_TOKEN`; or [self-host](#self-host) your own (ungated by default).

The MCP-client snippets below assume the hosted engine.

### Claude Code (terminal)

```bash
claude mcp add frontier --transport sse \
  --url https://frontier-592q.onrender.com/sse \
  --header "Authorization: Bearer $FRONTIER_TOKEN"
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "frontier": {
      "transport": "sse",
      "url": "https://frontier-592q.onrender.com/sse",
      "headers": { "Authorization": "Bearer YOUR_FRONTIER_TOKEN" }
    }
  }
}
```

### claude.ai (MCP integrations)

Add Frontier as a remote MCP server in claude.ai settings using the SSE URL `https://frontier-592q.onrender.com/sse`, with an `Authorization: Bearer <FRONTIER_TOKEN>` header.

### Self-host

Run your own instance instead of using the hosted one. Requires Python 3.11+.

```bash
git clone https://github.com/cafzal/frontier.git
cd frontier
pip install -e .

# stdio transport (for Claude Desktop / coding agents on the same machine)
python -m mcp_server.server

# SSE transport (for remote MCP clients)
MCP_TRANSPORT=sse python -m mcp_server.server

# Gate a public instance with a shared bearer token — clients must then send
# `Authorization: Bearer <token>`. Leave unset for an open local instance.
FRONTIER_MCP_TOKEN=your-secret MCP_TRANSPORT=sse python -m mcp_server.server
```

Point your MCP client at the local server — for SSE that's `http://localhost:8000/sse`. The optional web UI lives in [`ui/`](ui/) — see its [README](ui/README.md).

**Optional exact-solver audit layer.** Beyond the default NSGA-II/III, Frontier can wrap the NSGA search around an *exact inner solve*. Two first-class backends share one scalarization engine and differ only in the inner solve: `highs` (`pip install highspy`, CPU, cross-platform) and `cuopt` (NVIDIA GPU) — pick by hardware, identical certificate either way. Because an exact point is optimal for its scalarization, overlaying it on the heuristic frontier can only confirm or sharpen it, never worsen it (the invariant: NSGA never dominates an exact point). The agent does this in one call — **`explore certify`** to audit the NSGA `run` against the `exact_run` overlay and returns a dominance audit of NSGA points an exact solve dominates, the hypervolume coverage the overlay reclaims, and a recommendation. See [`architecture.md`](architecture.md#solver-backends-pluggable) for scope and details. The same exact run also exposes **solver-exact shadow prices + reduced costs** via **`explore sensitivity`** (`where_to_invest` / `near_misses`) — a QP-path feature; MILP and LP runs fall back to the frontier-inferred estimate.

### Deploy your own

Both pieces are plain web services — host them anywhere (Render, Fly, Railway, a VPS, Docker):

- **Engine** (Python) — `pip install -r requirements.txt`, then `MCP_TRANSPORT=sse python -m mcp_server.server`. Set `MCP_HOST=0.0.0.0` and `FRONTIER_MCP_TOKEN`; the host supplies `$PORT`. Must be publicly reachable — Anthropic's MCP connector calls it.
- **Web UI** (Node, in `ui/`) — `npm install && npm run build`, then `npm start`. Set `FRONTIER_MCP_URL` (the engine's `/sse`), `FRONTIER_MCP_TOKEN`, `ANTHROPIC_API_KEY`, `AGENT_BACKEND=messages-api`, and `UI_ACCESS_PASSWORD`. Long sessions are compacted between turns to keep the resent transcript under the model's context window (older turns drop to a breadcrumb); tune with the optional `AGENT_CONTEXT_WINDOW` / `AGENT_OUTPUT_RESERVE` / `AGENT_KEEP_RECENT_MESSAGES` if your model differs from the Opus default. On the Anthropic backends the engine's native context management additionally clears old tool results (or summarizes) server-side — `AGENT_CONTEXT_MANAGEMENT=clear_tool_uses` (default) | `compact` | `off`. Requests are also **prompt-cached** (system prompt, tool definitions, and the stable history prefix incl. injected skill guidance) so a multi-turn session re-reads them at ~10% of input cost — the biggest credit saver; `AGENT_PROMPT_CACHE=off` disables.

`FRONTIER_MCP_TOKEN` must match on both — that's what authenticates the UI to the engine.

**Render (one-click example):** [`render.yaml`](render.yaml) provisions both as a blueprint — auto-generates the shared token, derives the engine URL, leaves only `ANTHROPIC_API_KEY` to set. Point Render at your fork (New → Blueprint).

## Architecture

Frontier is a Python MCP server (FastMCP) wrapping pymoo's NSGA-II/III evolutionary solvers, with two first-class exact-solver backends (HiGHS on CPU, cuOpt on GPU) the agent can elect per run as an audit layer over the heuristic frontier. State persists per-problem as JSON; the optimizer produces a Pareto frontier with quality indicators, scenario-aware results, and shadow-price rates per binding constraint. Domain expertise lives in skill markdown files that the server auto-injects at workflow transitions, so the same rigor reaches every MCP client.

For full schemas, action parameters, data model, persistence layout, and the skill auto-injection mechanism, see [`architecture.md`](architecture.md). For skill, prompt, and MCP design principles, see [`best-practices.md`](best-practices.md).

### Tools

Four MCP tools — full action lists and parameters in [`architecture.md`](architecture.md):

- **`model`** — define and edit the problem: objectives (2–7; sum/avg/min/max/quadratic aggregation), options, scores, 8 constraint types, interaction matrices, reference points, and scenarios; plus save/load of named problems.
- **`solve`** — validate and optimize via NSGA-II/III: fast/thorough modes, seeded reproducibility (`seed_used`), per-scenario runs, frontier-quality gates, and infeasibility analysis; plus optional exact backends (HiGHS/cuOpt) to audit the frontier on supported shapes, paired with `explore certify`. Long runs (thorough, exact, large) execute in the **background** — `solve` hands back a `{status:"running", job_id}` handle and the agent polls `solve status` to completion, so a multi-minute solve never times out the turn; an optional `time_limit` bounds a run (best-so-far, flagged `time_limited`).
- **`explore`** — navigate results: tradeoffs and frontier shape, extremes / balanced / inflection points, shadow prices, exact sensitivity (`explore sensitivity` — where-to-invest shadow prices + near-miss reduced costs on exact continuous runs), marginal rates, scenario robustness (incl. CVaR), run comparison, curation, and feedback.
- **`get_skill`** — fetch the workflow guidance below.

### Skills

Markdown guides the server auto-injects at workflow transitions (also fetchable via `get_skill`) — domain judgment, not tool docs:

- **`problem_framing`** — objectives vs constraints, approach + aggregation, scenario definition.
- **`data_collection`** — score elicitation without anchoring bias, quality signals.
- **`optimization_strategy`** — iteration, constraint strategy, infeasibility, re-run judgment.
- **`solution_interpreter`** — presenting tradeoffs without a "best", eliciting preferences, curation.

## Background

Optional background — the thinking behind Frontier and how it's evolved:

- [Building an AI-Powered Decision Tool Prototype: A Product Manager's Journey](https://camafzal.substack.com/p/building-an-ai-powered-decision-tool) — May 2025
- [Lowering the Barriers to Decision Optimization with AI](https://camafzal.substack.com/p/lowering-the-barriers-to-decision) — Sep 2025
- [Making optimization accessible: AI as the translation layer](https://camafzal.substack.com/p/making-optimization-accessible-ai) — Jan 2026
- [Agents have a convergent reasoning gap](https://camafzal.substack.com/p/agents-have-a-convergent-reasoning) — Apr 2026

## Contributing

Contributions welcome — start with the developer docs:

- [`architecture.md`](architecture.md) — system architecture & data flow
- [`best-practices.md`](best-practices.md) — skill & prompt design guidelines

## Acknowledgements

Frontier builds on excellent open-source optimization work, with thanks to:

- **[pymoo](https://github.com/anyoptimization/pymoo)** (Apache-2.0) — the NSGA-II / NSGA-III evolutionary solvers at Frontier's core. Blank, J. & Deb, K. (2020). *pymoo: Multi-Objective Optimization in Python.* IEEE Access, 8, 89497–89509. The underlying algorithms are Deb et al., NSGA-II (2002) and Deb & Jain, NSGA-III (2014).
- **[HiGHS](https://github.com/ERGO-Code/HiGHS)** (MIT) — CPU exact-solver backend (`solver="highs"`). Huangfu, Q. & Hall, J.A.J. (2018). *Parallelizing the dual revised simplex method.* Mathematical Programming Computation, 10(1), 119–142.
- **[NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)** (Apache-2.0) — GPU exact-solver backend (`solver="cuopt"`).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
