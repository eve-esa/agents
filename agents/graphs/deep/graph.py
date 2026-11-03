"""Deep agent graph: ``deepagents.create_deep_agent`` with EVE middleware.

``compile`` returns the ``create_deep_agent`` graph itself, so its ``model``
node is top-level and its tokens reach a runner that streams with
``subgraphs=False``.  Fault tolerance and tool errors come from
:mod:`.middleware`; files come from a deepagents backend.

Requires the ``deep`` extra: ``pip install "eve-esa-agents[deep]"``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.tools import BaseTool

from ..base import AgentGraph
from ..policies import DEFAULT_LLM_IDLE_TIMEOUT, DEFAULT_LLM_RUN_TIMEOUT
from ..react.graph import _DEFAULT_MAX_TOKENS, ReactAgent
from .middleware import build_eve_middleware

logger = logging.getLogger(__name__)


def default_backend() -> StateBackend:
    """Scratch files in graph state."""
    return StateBackend()


def _is_declarative(spec: Any) -> bool:
    return isinstance(spec, dict) and "graph_id" not in spec and "runnable" not in spec


class DeepAgent(AgentGraph):
    """Deep agent: planning, files and subagents on the ``create_agent`` loop.

    Fallback, retry, run cap and ``on_policy`` events come from
    :func:`.middleware.build_eve_middleware`, attached to the main agent and to
    every declarative subagent, including ``general-purpose``.

    The system prompt is the ``react`` prompt followed by ``addendum`` from this
    graph's ``prompts.yaml``.
    """

    name = "deep"

    def __init__(self) -> None:
        super().__init__()
        base = ReactAgent().prompts.get("system") or ""
        addendum = self.prompts.get("addendum") or ""
        self.prompts["system"] = "\n".join(p.strip() for p in (base, addendum) if p.strip())

    def compile(
        self,
        *,
        llm,
        tools: List[BaseTool],
        checkpointer: Any,
        history: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        fallback_llm=None,
        llm_run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
        llm_idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
        on_policy=None,
        backend: Any = None,
        skills: Optional[List[str]] = None,
        subagents: Optional[Sequence[Any]] = None,
        memory: Optional[List[str]] = None,
        permissions: Optional[List[FilesystemPermission]] = None,
        store: Any = None,
        **kwargs: Any,
    ):
        """Build the deep agent graph.

        *backend*, *skills*, *subagents*, *memory*, *permissions* and *store*
        are passed to ``create_deep_agent``.  Without *backend*, the graph uses
        :func:`default_backend`.  Skills load only when *skills* is passed.

        *llm_idle_timeout* is not enforced here; the provider client's read
        timeout covers stalled streams.  Each model request is trimmed to
        *max_tokens* like the ``react`` graph; the deepagents summarization
        middleware is replaced, so the graph never summarizes history.  Large
        tool results and human messages stay in the thread instead of being
        moved to ``/large_tool_results/`` or ``/conversation_history/``.
        """
        if backend is None:
            backend = default_backend()

        def eve_middleware(agent: Optional[str] = None):
            return build_eve_middleware(
                backend=backend,
                max_tokens=max_tokens,
                fallback_llm=fallback_llm,
                run_timeout=llm_run_timeout,
                on_policy=on_policy,
                agent=agent,
            )

        specs: List[Any] = []
        for spec in subagents or []:
            if _is_declarative(spec):
                merged = {m.name: m for m in eve_middleware(spec["name"])}
                merged.update({m.name: m for m in spec.get("middleware", [])})
                spec = {**spec, "middleware": list(merged.values())}
            specs.append(spec)
        if not any(s.get("name") == GENERAL_PURPOSE_SUBAGENT["name"] for s in specs):
            # The auto-added general-purpose subagent would not inherit the
            # EVE middleware, so declare it explicitly.
            general_purpose = {
                **GENERAL_PURPOSE_SUBAGENT,
                "middleware": eve_middleware(GENERAL_PURPOSE_SUBAGENT["name"]),
            }
            if skills is not None:
                general_purpose["skills"] = skills
            specs.insert(0, general_purpose)

        return create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=self.instruction_text(history=history, summary=summary),
            middleware=eve_middleware(),
            subagents=specs,
            skills=skills,
            memory=memory,
            permissions=permissions,
            backend=backend,
            checkpointer=checkpointer,
            store=store,
            name=self.name,
        )
