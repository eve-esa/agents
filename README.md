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
- Fully qualified examples: `agents.graphs.react.graph.ReactAgent`,
  `agents.graphs.simple.graph.SimpleChatAgent`.

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
```

All imports inside `graphs/` are **relative** so this tree can be developed as its own repository.

Each graph ships ``prompts.yaml`` next to ``graph.py``. ``AgentGraph`` fills the ``prompts``
dict from that file at init (typically a ``system`` string for the lead-in). ``compile``
receives ``history`` (message list) and ``summary`` from the runner. The base
``format_history`` serialises summary + turns to text, and ``instruction_text``
prepends that to ``prompts['system']``.

Fault tolerance (requires ``langgraph>=1.2``) is configured inside the graph:

- **LLM nodes** — ``TimeoutPolicy`` (``llm_run_timeout`` / ``llm_idle_timeout``)
  and an ``error_handler`` that re-runs the node with ``fallback_llm`` when the
  backend supplies one.
- **Tool nodes** — ``RetryPolicy`` for transient MCP/API failures.

The backend should pass ``fallback_llm`` (a bound or raw chat model) and may
override timeout kwargs when calling ``compile``.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

(optional `[dev]` can be added in `pyproject.toml` later)

## Publish to GitHub

From this directory (after `git clone` or `git init` with `origin` set):

```bash
git add -A
git commit -m "Initial release 0.1.0 — pip package agents.graphs"
git branch -M main
git push -u origin main
```

Then pin in downstream `requirements.txt`:

```
eve-esa-agents @ git+https://github.com/eve-esa/agents.git@main
```
