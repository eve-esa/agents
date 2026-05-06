# eve-esa-agents

Pip-installable **LangGraph** agent definitions used by the Pi School / EVE backend.

The importable package name is **`agents`** (under `src/agents/`). After install:

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
src/agents/
  __init__.py
  graphs/
    __init__.py
    base.py          # AgentGraph, LatencyInterceptor
    utils.py         # Mistral/EVE text tool-call helpers, etc.
    react/
      __init__.py
      graph.py       # ReactAgent
      yaml/
        prompts.yaml # default system prompt (same content as EVE ``system.yaml``)
    simple/
      __init__.py
      graph.py       # SimpleChatAgent
      yaml/
        prompts.yaml   # shorter default prompt
```

All imports inside `graphs/` are **relative** so this tree can be developed as its own repository.

Each graph package ships ``yaml/prompts.yaml`` with a ``system_prompt`` field. The backend
runner passes ``system_prompt=None`` to ``compile``; graphs call
``AgentGraph.resolve_system_prompt`` so that file is used unless an explicit string is
passed. An optional conversation-summary prefix is still prepended by the runner.

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
