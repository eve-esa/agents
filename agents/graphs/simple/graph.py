"""Single-node chat graph — one LLM call per turn, no tools.

Useful for smoke tests and for runs where MCP tools must not be used even if
the model supports function calling.
"""

from typing import Any, List, Optional

from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from ..base import AgentGraph, AgentMessagesState
from ..policies import (
    DEFAULT_LLM_IDLE_TIMEOUT,
    DEFAULT_LLM_RUN_TIMEOUT,
    LLM_RETRY,
    build_llm_fallback_timeout_policy,
    llm_node_add_kwargs,
)
from ..utils import tiktoken_counter

_DEFAULT_MAX_TOKENS = 16_384


class SimpleChatAgent(AgentGraph):
    """MessagesState graph: ``START -> agent -> END`` (no tool node).

    Pass ``fallback_llm`` to enable in-graph model fallback.  The primary
    ``agent`` node retries transient failures in-place (``LLM_RETRY``) and,
    once retries are exhausted, an ``error_handler`` routes to a dedicated
    ``agent_fallback`` node that runs the fallback model.  Failures on
    ``agent_fallback`` bubble up unhandled.
    """

    name = "simple"

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
        **kwargs,
    ):
        instruction = self.instruction_text(history=history, summary=summary)
        has_fallback = fallback_llm is not None
        # Deliberately ignore *tools* — this graph never binds or invokes tools.

        def _build_messages(state: AgentMessagesState):
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
            return messages

        async def _invoke(state: AgentMessagesState, llm_client):
            messages = _build_messages(state)
            response = await llm_client.ainvoke(messages)
            return {"messages": [response]}

        async def agent_fn(state: AgentMessagesState):
            return await _invoke(state, llm)

        async def agent_fallback_fn(state: AgentMessagesState):
            return await _invoke(state, fallback_llm)

        builder = StateGraph(AgentMessagesState)
        builder.add_node(
            "agent",
            self.timed_node("agent", agent_fn),
            **llm_node_add_kwargs(
                fallback_node="agent_fallback",
                has_fallback=has_fallback,
                llm_run_timeout=llm_run_timeout,
                llm_idle_timeout=llm_idle_timeout,
            ),
        )
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)

        if has_fallback:
            fallback_node_kwargs: dict[str, Any] = {"retry_policy": LLM_RETRY}
            timeout = build_llm_fallback_timeout_policy(
                run_timeout=llm_run_timeout, idle_timeout=llm_idle_timeout
            )
            if timeout is not None:
                fallback_node_kwargs["timeout"] = timeout
            builder.add_node(
                "agent_fallback",
                self.timed_node("agent_fallback", agent_fallback_fn),
                **fallback_node_kwargs,
            )
            builder.add_edge("agent_fallback", END)

        return builder.compile(checkpointer=checkpointer)
