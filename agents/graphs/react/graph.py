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
    build_llm_timeout_policy,
    llm_node_add_kwargs,
)
from ..utils import (
    parse_text_tool_calls,
    reformat_messages_for_text_tool_model,
    strip_content_from_tool_call_messages,
    tiktoken_counter,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 16_384


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
        streaming: bool = True,
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
        node_idle_timeout = llm_idle_timeout if streaming else None

        # shared invocation logic
        async def _invoke(state: AgentMessagesState, llm_bound):
            messages = list(state["messages"])
            if instruction:
                messages = [SystemMessage(content=instruction)] + messages

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

            if streaming:
                response = None
                async for chunk in llm_bound.astream(messages):
                    if response is None:
                        response = chunk
                    else:
                        response = response + chunk
            else:
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

        # ── routing ────────────────────────────────────────────────────────
        def should_continue(state: AgentMessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return END

        # ── build graph ────────────────────────────────────────────────────
        builder = StateGraph(AgentMessagesState)
        builder.add_node(
            "agent",
            self.timed_node("agent", agent_fn),
            **llm_node_add_kwargs(
                fallback_node="agent_fallback",
                has_fallback=has_fallback,
                llm_run_timeout=llm_run_timeout,
                llm_idle_timeout=node_idle_timeout,
            ),
        )
        builder.add_node("tools", self.make_tools_node(tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "agent")

        if has_fallback:
            fallback_node_kwargs: dict[str, Any] = {"retry_policy": LLM_RETRY}
            timeout = build_llm_timeout_policy(
                run_timeout=llm_run_timeout, idle_timeout=node_idle_timeout
            )
            if timeout is not None:
                fallback_node_kwargs["timeout"] = timeout
            builder.add_node(
                "agent_fallback",
                self.timed_node("agent_fallback", agent_fallback_fn),
                **fallback_node_kwargs,
            )
            builder.add_conditional_edges(
                "agent_fallback", should_continue, {"tools": "tools", END: END}
            )

        return builder.compile(checkpointer=checkpointer)
