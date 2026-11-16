"""ReAct agent graph: manual tool-calling loop via LangGraph StateGraph.

Uses shared utilities from the parent ``graphs`` package (``utils``) for
text-format tool-call parsing and message sanitisation.  Imports are
relative so this tree can be cloned as its own repository.
"""

import logging
import time
from typing import Any, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
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
    mark_fallback,
)
from ..utils import (
    MALFORMED_TOOL_CALL_ANSWER,
    MALFORMED_TOOL_CALL_RETRIES,
    clock_context,
    malformed_reply_errors,
    prepare_model_messages,
    structure_model_reply,
    tiktoken_counter,
    tool_names,
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
        bound_names = tool_names(tools)

        # shared invocation logic
        async def _invoke(state: AgentMessagesState, llm_bound):
            messages = list(state["messages"])
            system_instruction = clock_context()
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

            # Tool history stays structured (assistant tool_calls plus tool
            # messages): the provider renders it in the model's own format.
            # Old text calls become structured, every call gets one result,
            # ids get the shape JSC accepts. A reply whose only calls have
            # malformed arguments is answered with an error result and the
            # model tries again, so the turn never ends empty.
            produced: List[Any] = []
            for _ in range(MALFORMED_TOOL_CALL_RETRIES + 1):
                response = await llm_bound.ainvoke(
                    prepare_model_messages(messages + produced, bound_names)
                )
                structured = structure_model_reply(response, bound_names)
                if structured is not response and not response.tool_calls:
                    logger.info(
                        "Structured %d tool call(s) from a text or malformed reply",
                        len(structured.tool_calls),
                    )
                produced.append(structured)
                errors = malformed_reply_errors(structured)
                if errors is None:
                    break
                logger.warning(
                    "Model sent %d tool call(s) with malformed arguments",
                    len(errors),
                )
                produced.extend(errors)
            else:
                produced.append(AIMessage(content=MALFORMED_TOOL_CALL_ANSWER))

            return {"messages": produced}

        # primary agent node
        async def agent_fn(state: AgentMessagesState):
            return await _invoke(state, primary_llm_bound)

        # fallback agent node (no further error_handler - failures bubble)
        async def agent_fallback_fn(state: AgentMessagesState):
            start = time.perf_counter()
            result = await _invoke(state, fallback_llm_bound)
            mark_fallback(
                result["messages"], round(time.perf_counter() - start, 3)
            )
            return result

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
        builder.add_edge(START, "agent")
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
