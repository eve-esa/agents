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
    llm_node_add_kwargs,
)
from ..utils import tiktoken_counter

_DEFAULT_MAX_TOKENS = 16_384


class SimpleChatAgent(AgentGraph):
    """MessagesState graph: ``START -> agent -> END`` (no tool node)."""

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

        async def agent_fn(state: AgentMessagesState):
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

            llm_to_use = (
                fallback_llm
                if state.get("use_fallback_llm") and fallback_llm is not None
                else llm
            )
            # A successful attempt clears the fallback flag so it scopes to the
            # failed attempt's retry rather than pinning the thread to fallback.
            response = await llm_to_use.ainvoke(messages)
            return {"messages": [response], "use_fallback_llm": False}

        builder = StateGraph(AgentMessagesState)
        builder.add_node(
            "agent",
            self.timed_node("agent", agent_fn),
            **llm_node_add_kwargs(
                node="agent",
                has_fallback=has_fallback,
                llm_run_timeout=llm_run_timeout,
                llm_idle_timeout=llm_idle_timeout,
            ),
        )
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        return builder.compile(checkpointer=checkpointer)
