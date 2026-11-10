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
- Short name `deep` — [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
  harness with files and subagents. Needs the `deep` extra:
  `pip install "eve-esa-agents[deep] @ git+https://github.com/eve-esa/agents.git"`.
- Fully qualified examples: `agents.graphs.react.graph.ReactAgent`,
  `agents.graphs.simple.graph.SimpleChatAgent`, `agents.graphs.deep.graph.DeepAgent`.

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
    deep/
      __init__.py
      graph.py       # DeepAgent
      middleware.py  # EVE middleware on top of LangChain built-ins
      prompts.yaml   # addendum appended to the react system prompt
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

## Deep agent

``DeepAgent.compile`` returns the ``create_deep_agent`` graph itself, so its
``model`` and ``tools`` nodes are top-level and tokens stream with
``subgraphs=False``. It takes the same arguments as ``react``, plus optional
``backend``, ``skills``, ``subagents``, ``memory``, ``permissions`` and ``store``,
which go to ``create_deep_agent``.

Fault tolerance is middleware, attached to the main agent and to every
declarative subagent (including ``general-purpose``):

- ``EveFallbackMiddleware`` — LangChain ``ModelFallbackMiddleware`` plus
  ``on_policy`` events (``timeout`` or ``error_handler``, then ``fallback``).
  Answers from the fallback model carry
  ``response_metadata["eve_fallback_used"] = True`` (``FALLBACK_MARKER``) and
  the seconds the fallback call took in
  ``response_metadata["eve_fallback_latency_s"]`` (``FALLBACK_LATENCY_KEY``);
  there is no ``agent_fallback`` node. Failure events carry
  ``extra["failed_after_s"]``, the seconds the failed model spent, retries
  included.
- ``EveRetryMiddleware`` — LangChain ``ModelRetryMiddleware`` with the
  ``LLM_RETRY`` settings, a per-attempt ``llm_run_timeout`` (1.5x for the
  fallback model) and ``retry`` events. ``llm_idle_timeout`` is not enforced by
  the graph; the provider client's read timeout covers stalled streams.
- ``EveCompatMiddleware`` — the ``react`` clock, Mistral message clean-up and
  ``[TOOL_CALLS]`` parsing.
- LangChain ``ToolErrorMiddleware`` — a failing tool returns
  ``ToolMessage(status="error", content="Tool error: ...")``.

``EveTrimMiddleware`` replaces the deepagents summarization middleware on the
main agent and every subagent. It trims each model request to ``max_tokens``
with the same ``trim_messages`` settings as ``react``, so the graph never
summarizes history. Policy events report ``node="model"``, with the model name in
``extra["model"]`` and the subagent name in ``extra["agent"]``.

Without a ``backend``, files live in graph state. Skills load only when
``compile`` is given ``skills``. A tool result above ``tool_result_max_tokens``
(default 20,000, counted with the same tiktoken counter as the history trim)
is written to ``/large_tool_results/`` on that backend;
``tool_result_max_tokens=None`` leaves every result in the thread.

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
