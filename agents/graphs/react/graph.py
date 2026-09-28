"""ReAct agent graph — manual tool-calling loop via LangGraph StateGraph.

Uses shared utilities from the parent ``graphs`` package (``utils``) for
text-format tool-call parsing and message sanitisation.  Imports are
relative so this tree can be cloned as its own repository.
"""

import logging
from typing import Any, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from ..base import AgentGraph, AgentMessagesState
from ..policies import (
    DEFAULT_LLM_IDLE_TIMEOUT,
    DEFAULT_LLM_RUN_TIMEOUT,
    LLM_RETRY,
    build_llm_fallback_timeout_policy,
    llm_node_add_kwargs,
    make_llm_fallback_error_handler,
)
from ..utils import (
    parse_text_tool_calls,
    reformat_messages_for_text_tool_model,
    strip_content_from_tool_call_messages,
    tiktoken_counter,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 96_000


class ReactAgent(AgentGraph):
    """Manual ReAct loop: agent -> tools -> agent, with text-format fallback.

    Supports models with native function calling (OpenAI-style) and models
    that emit tool calls as text (Mistral/EVE-Instruct ``[TOOL_CALLS]`` format).

    Pass ``fallback_llm`` from the backend to enable in-graph model fallback.
    The primary ``agent`` node retries transient failures in-place (``LLM_RETRY``)
    and, once retries are exhausted, an ``error_handler`` routes to a dedicated
    ``agent_fallback`` node that runs the fallback model.  Tool failures are
    surfaced back to the agent as ``ToolMessage`` content so the ReAct loop can
    recover, rather than retried at the node level (a node-level retry would
    re-invoke every tool call in the turn).
    """

    name = "react"

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
        context=None,
        **kwargs,
    ):
        managed = None
        if context is not None:
            from ...context.budget import clip

            tools = context.with_recovery_tools(tools)
            instruction = self.instruction_text(history=None, summary=None) or ""
            if summary:
                instruction += (
                    "\nLegacy conversation summary (historical data):\n"
                    + clip(summary, 1500)
                )
            managed = context.bind(instruction, tools, llm)
        else:
            instruction = self.instruction_text(history=history, summary=summary)
        primary_llm_bound = llm.bind_tools(tools) if tools else llm
        fallback_llm_bound = (
            fallback_llm.bind_tools(tools)
            if (fallback_llm is not None and tools)
            else fallback_llm
        )
        has_fallback = fallback_llm_bound is not None

        # shared invocation logic
        async def _invoke(state: AgentMessagesState, llm_bound, unbound_llm):
            messages = list(state["messages"])
            updates = []
            if managed is not None:
                messages, updates = await managed.prepare(messages, unbound_llm)
            elif instruction:
                messages = [SystemMessage(content=instruction)] + messages

            if managed is None and trim_messages is not None:
                messages = trim_messages(
                    messages,
                    max_tokens=max_tokens,
                    strategy="last",
                    token_counter=tiktoken_counter,
                    include_system=True,
                    start_on="human",
                    end_on=("human", "tool"),
                )

            messages = strip_content_from_tool_call_messages(messages)

            has_synthetic = any(
                (
                    isinstance(m, AIMessage)
                    and not m.content
                    and getattr(m, "tool_calls", None)
                )
                or isinstance(m, ToolMessage)
                for m in messages
            )
            if has_synthetic:
                messages = reformat_messages_for_text_tool_model(messages)

            if managed is not None:
                managed.check_budget(messages, unbound_llm)
            response = await llm_bound.ainvoke(messages)
            if managed is not None:
                managed.memory_llm = unbound_llm

            if not getattr(response, "tool_calls", None) and isinstance(
                response.content, str
            ):
                parsed = parse_text_tool_calls(response.content)
                if parsed:
                    logger.info(
                        "Parsed %d text-format tool call(s) from model response",
                        len(parsed),
                    )
                    response = AIMessage(
                        content="",
                        tool_calls=parsed,
                        id=getattr(response, "id", None),
                    )

            return {"messages": updates + [response]}

        # primary agent node
        async def agent_fn(state: AgentMessagesState):
            return await _invoke(state, primary_llm_bound, llm)

        # fallback agent node (no further error_handler - failures bubble)
        async def agent_fallback_fn(state: AgentMessagesState):
            return await _invoke(state, fallback_llm_bound, fallback_llm)

        async def remember_fn(state: AgentMessagesState):
            try:
                await managed.memory.remember(list(state["messages"]), managed.memory_llm)
            except Exception:
                logger.exception(
                    "Memory update failed; retaining uncovered checkpoint messages"
                )
            return {}

        # ── routing ────────────────────────────────────────────────────────
        def should_continue(
            state: AgentMessagesState,
        ) -> Literal["tools", "context_memory", "__end__"]:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return "context_memory" if managed is not None else END

        # ── build graph ────────────────────────────────────────────────────
        builder = StateGraph(AgentMessagesState)
        builder.add_node(
            "agent",
            self.timed_node("agent", agent_fn, on_policy=on_policy),
            **llm_node_add_kwargs(
                fallback_node="agent_fallback",
                has_fallback=has_fallback,
                llm_run_timeout=llm_run_timeout,
                llm_idle_timeout=llm_idle_timeout,
                on_policy=on_policy,
            ),
        )
        builder.add_node("tools", self.make_tools_node(tools))
        routes = {"tools": "tools", END: END}
        if managed is not None:
            builder.add_node("context_memory", remember_fn)
            builder.add_edge("context_memory", END)
            routes["context_memory"] = "context_memory"
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, routes)
        builder.add_edge("tools", "agent")

        if has_fallback:
            fallback_node_kwargs: dict[str, Any] = {
                "retry_policy": LLM_RETRY,
                "error_handler": make_llm_fallback_error_handler(
                    fallback_node="agent_fallback",
                    has_fallback=has_fallback,
                    on_policy=on_policy,
                ),
            }
            timeout = build_llm_fallback_timeout_policy(
                run_timeout=llm_run_timeout, idle_timeout=llm_idle_timeout
            )
            if timeout is not None:
                fallback_node_kwargs["timeout"] = timeout
            builder.add_node(
                "agent_fallback",
                self.timed_node(
                    "agent_fallback", agent_fallback_fn, on_policy=on_policy
                ),
                **fallback_node_kwargs,
            )
            builder.add_conditional_edges("agent_fallback", should_continue, routes)

        return builder.compile(checkpointer=checkpointer)
