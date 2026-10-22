# eve-esa-agents

Pip-installable **LangGraph** agent definitions used by the Pi School / EVE backend.

The importable package name is **`agents`** (top-level `agents/` folder). After install:

```bash
# Prefer the same venv as your backend (install backend requirements first).
pip install git+https://github.com/eve-esa/agents.git
# If pip wants to upgrade LangGraph/LangChain, either align this package's pins in
# pyproject.toml with your app or: pip install eve-esa-agents --no-deps

# Editable from a clone:
pip install -e .
```

Use in the backend via `AGENT_GRAPH_TYPE`:

- Short name `react` — ReAct loop with tools (default).
- Short name `simple` — single LLM node, **no tools** (good for smoke tests).
- Short name `eo_multi_agent` — EO Multi-Agent: planner, parallel domain specialists,
  reporter and rule-based verifier (see below).
- Fully qualified examples: `agents.graphs.react.graph.ReactAgent`,
  `agents.graphs.simple.graph.SimpleChatAgent`,
  `agents.graphs.eo_multi_agent.graph.EOMultiAgent`.

The backend also accepts the graph per request (`"agent": "eo_multi_agent"` in the
generation request), which makes side-by-side comparisons easy.

## Layout

```
agents/
  __init__.py
  graphs/
    __init__.py
    base.py          # AgentGraph, LatencyInterceptor
    utils.py         # Mistral/EVE text tool-call helpers, etc.
    react/
      __init__.py
      graph.py       # ReactAgent
      prompts.yaml   # default system prompt (same content as EVE ``system.yaml``)
    simple/
      __init__.py
      graph.py       # SimpleChatAgent
      prompts.yaml   # shorter default prompt
    eo_multi_agent/
      __init__.py
      graph.py       # EOMultiAgent
      team.py        # which specialist owns which MCP tools
      plan.py        # planner reply -> stages; playbooks
      verify.py      # rule-based answer checks (numbers, artifact links)
      prompts.yaml   # team, playbooks and the prompts of each role
```

All imports inside `graphs/` are **relative** so this tree can be developed as its own repository.

Each graph ships ``prompts.yaml`` next to ``graph.py``. ``AgentGraph`` fills the ``prompts``
dict from that file at init (typically a ``system`` string for the lead-in). ``compile``
receives ``history`` (message list) and ``summary`` from the runner. The base
``format_history`` serialises summary + turns to text, and ``instruction_text``
prepends that to ``prompts['system']``.

Fault tolerance (requires ``langgraph>=1.2``) is configured inside the graph:

- **LLM nodes** — ``TimeoutPolicy`` (``llm_run_timeout`` / ``llm_idle_timeout``),
  an in-place ``RetryPolicy`` (``LLM_RETRY``) for transient failures, and an
  ``error_handler`` that routes to a dedicated ``agent_fallback`` recovery node
  once retries are exhausted.  This follows LangGraph's canonical recovery-node
  pattern: the handler returns ``Command(goto="agent_fallback")`` to a separate
  node rather than looping back to the failing node.  The ``agent_fallback`` node
  runs the same logic with ``fallback_llm`` and itself has ``LLM_RETRY`` for
  transient failures; unhandled failures on ``agent_fallback`` bubble up.  The
  ``agent_fallback`` node is registered only when the backend supplies
  ``fallback_llm`` to ``compile``.
- **Tool nodes** — failures are returned to the agent as ``ToolMessage`` content
  so the ReAct loop can recover (no node-level retry, which would re-invoke
  every tool call in the turn).

The backend should pass ``fallback_llm`` (a bound or raw chat model) and may
override timeout kwargs when calling ``compile``.

## EO Multi-Agent (`eo_multi_agent`)

One ReAct agent holding every tool works well with a large model. With a small
one it loses track in long tool chains: it stops halfway, rewrites image links
and states values no tool returned. EO Multi-Agent gives each model call one
small job:

```
planner ─┬─> specialist (forest)  ─┐
         ├─> specialist (climate) ─┼─> next stage … ─> reporter ─> verifier ─> publish
         └─> …    in parallel     ─┘                      ^            │
                                                          └── once ────┘ (numbers not in the evidence)
```

- **Planner** picks a *playbook* (a tested plan for a known request, declared in
  `prompts.yaml`) or writes its own stages. Tasks in one stage run in parallel;
  a later stage sees the reports of the earlier ones. An unusable reply falls
  back to one *generalist* task with every tool, which behaves like `react`.
- **Specialists** each run a small ReAct loop with only their own tools. A tool
  belongs to the specialist whose MCP server prefixes its name
  (`forest_compliance_eudr_check_plots` → `forest`); unclaimed tools go to the
  specialist with server `"*"`. Each writes a factual report for the team.
- **Reporter** writes the answer from the reports, without tools.
- **Verifier** is rule-based, not a model: it drops image markdown (the backend
  appends the real artifact stubs) and links to artifacts no tool produced, and
  sends the draft back once if it holds numbers not found in the question, the
  tool results or the reports.

Only the published answer and the tool calls reach the user: internal model
calls are tagged `nostream`, and plans, reports and drafts live in state keys
other than `messages`. Tools run through the same tool objects the backend
passes in, so MCP interceptors (artifact capture, latency) see every call.
Retries and the fallback model wrap each model call, not the node, so a retry
never repeats a specialist's tool calls.

To add a specialist or a playbook, edit `agents/graphs/eo_multi_agent/prompts.yaml`;
no code change is needed.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Release

`main` is what dev runs. A push to `main` dispatches `deploy-dev` in
[eve-esa/backend](https://github.com/eve-esa/backend) on that repo's `main`,
and the image installs this commit.

release-please keeps a `chore(main): release x.y.z` pull request open. Merging
it tags `vX.Y.Z`, publishes the sdist and wheel on the GitHub Release, and
opens a backend pull request titled `fix(deps): pin agents to vX.Y.Z` that
writes the tag to `agents.ref`. Merging that, then the backend's own release
pull request, builds the image staging promotes. Production stays a manual
dispatch of the backend's promote-prod workflow, with one approval.

Dev installs a commit SHA, never the branch name `main`. The backend Dockerfile
caches the agents install on the build-arg text, so a floating branch name
would reuse a stale layer.

Pull request titles must be [Conventional Commits](https://www.conventionalcommits.org/).
Squash merges use the title as the commit subject, and that subject is what
release-please versions.

The `BACKEND_DISPATCH_TOKEN` secret on this repo has to be able to dispatch
workflows and open pull requests on `eve-esa/backend`. Without it, the dev
dispatch and the pin pull request fail.
