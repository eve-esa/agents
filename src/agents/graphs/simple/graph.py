"""Single-node chat graph — one LLM call per turn, no tools.

Useful for smoke tests and for runs where MCP tools must not be used even if
the model supports function calling.
"""

from typing import Any, List, Optional

from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from ..base import AgentGraph
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
        conversation_prefix: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        instruction = self.instruction_text(conversation_prefix=conversation_prefix)
        # Deliberately ignore *tools* — this graph never binds or invokes tools.

        async def agent_fn(state: MessagesState):
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

            response = await llm.ainvoke(messages)
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("agent", self.timed_node("agent", agent_fn))
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        return builder.compile(checkpointer=checkpointer)
