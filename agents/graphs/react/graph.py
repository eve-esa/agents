"""ReAct agent graph — manual tool-calling loop via LangGraph StateGraph.

Uses shared utilities from the parent ``graphs`` package (``utils``) for
text-format tool-call parsing and message sanitisation.  Imports are
relative so this tree can be cloned as its own repository.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
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


def _forced_tool_query(messages: List[Any], tool_name: str) -> Optional[str]:
    """Text of the last human message, or None when this turn already called *tool_name*."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            return str(content).strip() or None
        if isinstance(msg, AIMessage) and any(
            tc["name"] == tool_name for tc in (msg.tool_calls or [])
        ):
            return None
    return None


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

    A run with ``force_first_tool`` in ``config["configurable"]`` naming a bound
    tool calls that tool once with ``{"query": <last human message>}`` before
    the model answers; the call goes through the same ``tools`` node.
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
        **kwargs,
    ):
        instruction = self.instruction_text(history=history, summary=summary)
        primary_llm_bound = llm.bind_tools(tools) if tools else llm
        fallback_llm_bound = (
            fallback_llm.bind_tools(tools)
            if (fallback_llm is not None and tools)
            else fallback_llm
        )
        has_fallback = fallback_llm_bound is not None

        # shared invocation logic
        async def _invoke(state: AgentMessagesState, llm_bound):
            messages = list(state["messages"])
            # Resolve at invocation time so cached graphs and resumed threads
            # do not keep the date on which they were compiled.
            now = datetime.now(timezone.utc)
            runtime_context = (
                "## Clock\n"
                f"Current UTC timestamp: {now.isoformat(timespec='seconds')}\n"
                f"Current UTC date: {now.date().isoformat()}\n"
                f"Current UTC year: {now.year}\n"
                "Use this clock for the current date/year and relative dates, "
                "not the knowledge cutoff or dates in conversation history. "
                "For local requests, interpret this instant in the requested "
                "location's timezone.\n"
            )
            system_instruction = runtime_context
            if instruction:
                system_instruction += "\n" + instruction
            messages = [SystemMessage(content=system_instruction)] + messages

            if trim_messages is not None:
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

            response = await llm_bound.ainvoke(messages)

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

            return {"messages": [response]}

        # primary agent node
        async def agent_fn(state: AgentMessagesState):
            return await _invoke(state, primary_llm_bound)

        # fallback agent node (no further error_handler - failures bubble)
        async def agent_fallback_fn(state: AgentMessagesState):
            return await _invoke(state, fallback_llm_bound)

        tool_names = {t.name for t in tools}

        def _forced_tool(state: AgentMessagesState, config: RunnableConfig):
            name = (config.get("configurable") or {}).get("force_first_tool")
            if not name or name not in tool_names:
                return None, None
            return name, _forced_tool_query(state["messages"], name)

        # synthetic tool call, executed by the tools node like a model call
        def force_tool_fn(state: AgentMessagesState, config: RunnableConfig):
            name, query = _forced_tool(state, config)
            call = {
                "name": name,
                "args": {"query": query},
                # 9 alphanumerics: the only id shape Mistral accepts.
                "id": uuid.uuid4().hex[:9],
                "type": "tool_call",
            }
            return {"messages": [AIMessage(content="", tool_calls=[call])]}

        # ── routing ────────────────────────────────────────────────────────
        def route_start(
            state: AgentMessagesState, config: RunnableConfig
        ) -> Literal["force_tool", "agent"]:
            _, query = _forced_tool(state, config)
            return "force_tool" if query else "agent"

        def should_continue(state: AgentMessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return END

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
        builder.add_node("force_tool", force_tool_fn)
        builder.add_conditional_edges(
            START, route_start, {"force_tool": "force_tool", "agent": "agent"}
        )
        builder.add_edge("force_tool", "tools")
        builder.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
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
            builder.add_conditional_edges(
                "agent_fallback", should_continue, {"tools": "tools", END: END}
            )

        return builder.compile(checkpointer=checkpointer)
