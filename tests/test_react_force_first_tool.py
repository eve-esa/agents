"""ReactAgent ``force_first_tool``: one tool call before the model answers."""

import asyncio
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.react.graph import ReactAgent

RETRIEVE = "eve_retrieval_retrieve"


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted replies in order, then a plain answer."""

    replies: List[AIMessage] = []
    log: Any = None
    systems: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append(("model", len(messages)))
        self.systems.append(messages[0].content)
        reply = self.replies.pop(0) if self.replies else AIMessage(content="answer")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def make_tools(log):
    @tool(RETRIEVE)
    async def retrieve(query: str) -> str:
        """Search the knowledge base."""
        log.append((RETRIEVE, query))
        return "documents"

    @tool("geocode_search")
    async def geocode(query: str) -> str:
        """Find a place."""
        log.append(("geocode_search", query))
        return "place"

    return [retrieve, geocode]


def build(log, replies=None, tools=None, systems=None):
    model = ScriptedChatModel(
        replies=list(replies or []), log=log, systems=[] if systems is None else systems
    )
    return ReactAgent().compile(
        llm=model,
        tools=make_tools(log) if tools is None else tools,
        checkpointer=InMemorySaver(),
        llm_run_timeout=None,
        llm_idle_timeout=None,
    )


def run(graph, text, force: Optional[str], thread="t1"):
    configurable = {"thread_id": thread}
    if force is not None:
        configurable["force_first_tool"] = force
    return asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": configurable},
        )
    )


def forced_calls(messages, name=RETRIEVE):
    return [
        tc
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
        if tc["name"] == name
    ]


def test_forced_tool_runs_once_before_the_model_with_the_user_text():
    log: list = []
    graph = build(log)
    out = run(graph, "What is the Doppler effect?", RETRIEVE)

    assert log[0] == (RETRIEVE, "What is the Doppler effect?")
    assert [e for e in log if e[0] == RETRIEVE] == [(RETRIEVE, "What is the Doppler effect?")]
    assert log[1][0] == "model"
    calls = forced_calls(out["messages"])
    assert len(calls) == 1
    assert calls[0]["args"] == {"query": "What is the Doppler effect?"}
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == [calls[0]["id"]]
    assert out["messages"][-1].content == "answer"


def test_forced_tool_not_bound_leaves_the_turn_to_the_model():
    log: list = []
    graph = build(log, tools=make_tools(log)[1:])
    out = run(graph, "hello", RETRIEVE)

    assert log[0][0] == "model"
    assert not any(e[0] == RETRIEVE for e in log)
    assert forced_calls(out["messages"]) == []


def test_no_flag_keeps_todays_loop():
    log: list = []
    replies = [
        AIMessage(
            content="",
            tool_calls=[{"name": "geocode_search", "args": {"query": "Rome"}, "id": "abc123def"}],
        )
    ]
    graph = build(log, replies=replies)

    async def node_order():
        order = []
        async for update in graph.astream(
            {"messages": [HumanMessage(content="Where is Rome?")]},
            config={"configurable": {"thread_id": "t1"}},
            stream_mode="updates",
        ):
            order.extend(update.keys())
        return order

    assert asyncio.run(node_order()) == ["agent", "tools", "agent"]
    assert log[0][0] == "model"
    assert [e for e in log if e[0] != "model"] == [("geocode_search", "Rome")]


def test_model_may_call_any_tool_after_the_forced_call():
    log: list = []
    replies = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": RETRIEVE, "args": {"query": "again"}, "id": "abc123def"},
                {"name": "geocode_search", "args": {"query": "Rome"}, "id": "ghi456jkl"},
            ],
        )
    ]
    graph = build(log, replies=replies)
    out = run(graph, "first", RETRIEVE)

    tools_run = [e for e in log if e[0] != "model"]
    assert tools_run == [(RETRIEVE, "first"), (RETRIEVE, "again"), ("geocode_search", "Rome")]
    assert out["messages"][-1].content == "answer"


def test_forced_call_happens_on_every_human_turn_with_a_unique_id():
    log: list = []
    graph = build(log)
    run(graph, "What is SAR?", RETRIEVE)
    out = run(graph, "and in 2020?", RETRIEVE)

    assert [e for e in log if e[0] == RETRIEVE] == [
        (RETRIEVE, "What is SAR?"),
        (RETRIEVE, "What is SAR? and in 2020?"),
    ]
    calls = forced_calls(out["messages"])
    assert [c["args"]["query"] for c in calls] == ["What is SAR?", "What is SAR? and in 2020?"]
    assert calls[0]["id"] != calls[1]["id"]
    assert all(c["id"].isalnum() and len(c["id"]) == 9 for c in calls)


def test_forced_call_reads_text_parts_of_a_multipart_message():
    log: list = []
    graph = build(log)
    content = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
    run(graph, content, RETRIEVE)

    assert log[0] == (RETRIEVE, "part one part two")


def test_forced_call_is_streamed_as_a_message_of_its_own_node():
    log: list = []
    graph = build(log)

    async def streamed():
        seen = []
        async for chunk, meta in graph.astream(
            {"messages": [HumanMessage(content="Doppler")]},
            config={"configurable": {"thread_id": "t1", "force_first_tool": RETRIEVE}},
            stream_mode="messages",
        ):
            seen.append((meta.get("langgraph_node"), type(chunk).__name__, bool(getattr(chunk, "tool_calls", None))))
        return seen

    seen = asyncio.run(streamed())
    assert seen[0] == ("force_tool", "AIMessage", True)
    assert seen[1][:2] == ("tools", "ToolMessage")


def test_follow_up_query_is_capped_and_keeps_the_current_question_whole():
    log: list = []
    graph = build(log)
    run(graph, "x" * 2000, None)
    current = "and what about the 2020 floods in Pakistan?"
    run(graph, current, RETRIEVE)

    query = [e for e in log if e[0] == RETRIEVE][0][1]
    assert query.endswith(" " + current)
    assert len(query) == 500


def test_model_is_told_only_when_the_call_was_forced():
    log: list = []
    systems: list = []
    graph = build(log, systems=systems)
    run(graph, "Doppler", RETRIEVE, thread="forced")
    run(graph, "Doppler", None, thread="free")

    assert len(systems) == 2
    assert f"{RETRIEVE} already ran for this question" in systems[0]
    assert "already ran for this question" not in systems[1]


def test_note_is_gone_on_the_next_unforced_turn_of_the_same_thread():
    log: list = []
    systems: list = []
    graph = build(log, systems=systems)
    run(graph, "turn one", RETRIEVE)
    run(graph, "turn two", None)

    assert "already ran for this question" in systems[0]
    assert "already ran for this question" not in systems[1]


def test_retry_of_the_same_question_is_not_prepended():
    log: list = []
    graph = build(log)
    run(graph, "What is SAR?", RETRIEVE)
    run(graph, "What is SAR?", RETRIEVE)

    assert [e[1] for e in log if e[0] == RETRIEVE] == ["What is SAR?", "What is SAR?"]


def test_long_current_question_is_capped_too():
    log: list = []
    graph = build(log)
    run(graph, "previous question", None)
    run(graph, "y" * 2000, RETRIEVE)

    assert [e[1] for e in log if e[0] == RETRIEVE] == ["y" * 500]
