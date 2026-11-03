"""Fallback answers from react and simple carry the shared marker."""

from __future__ import annotations

import asyncio
from typing import Any, List

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.policies import FALLBACK_LATENCY_KEY, FALLBACK_MARKER
from agents.graphs.react.graph import ReactAgent
from agents.graphs.simple.graph import SimpleChatAgent


class ScriptedModel(FakeMessagesListChatModel):
    calls: List[Any] = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return await super()._agenerate(messages, *args, **kwargs)


class FailingModel(ScriptedModel):
    async def _agenerate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        raise ConnectionError("provider down")


def _model(cls=ScriptedModel, *responses: AIMessage):
    return cls(responses=list(responses) or [AIMessage("unused")], calls=[])


async def _run(graph) -> List[Any]:
    out = await graph.ainvoke(
        {"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "t"}}
    )
    return out["messages"]


def _compile(agent, llm, **kwargs):
    kwargs.setdefault("tools", [])
    return agent.compile(llm=llm, checkpointer=InMemorySaver(), **kwargs)


def test_react_fallback_answer_is_marked() -> None:
    primary = _model(FailingModel)
    fallback = _model(ScriptedModel, AIMessage("from fallback"))
    messages = asyncio.run(
        _run(_compile(ReactAgent(), primary, fallback_llm=fallback))
    )
    answer = messages[-1]
    assert answer.content == "from fallback"
    assert answer.response_metadata[FALLBACK_MARKER] is True
    assert answer.response_metadata[FALLBACK_LATENCY_KEY] >= 0


def test_react_primary_answer_has_no_marker() -> None:
    messages = asyncio.run(
        _run(_compile(ReactAgent(), _model(ScriptedModel, AIMessage("ok"))))
    )
    assert FALLBACK_MARKER not in messages[-1].response_metadata
    assert FALLBACK_LATENCY_KEY not in messages[-1].response_metadata


def test_simple_fallback_answer_is_marked() -> None:
    primary = _model(FailingModel)
    fallback = _model(ScriptedModel, AIMessage("from fallback"))
    messages = asyncio.run(
        _run(_compile(SimpleChatAgent(), primary, fallback_llm=fallback))
    )
    answer = messages[-1]
    assert answer.content == "from fallback"
    assert answer.response_metadata[FALLBACK_MARKER] is True
    assert answer.response_metadata[FALLBACK_LATENCY_KEY] >= 0
