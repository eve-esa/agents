"""Tool history sent to the model stays structured; text tool calls still parse.

The model sees earlier tool calls as assistant ``tool_calls`` and ``tool``
messages, the OpenAI chat completions form every provider renders with its
own template. A text copy of that history taught EVE-Instruct to answer in
the copied form, which no parser read.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, List

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs import utils
from agents.graphs.react.graph import ReactAgent

_PROVIDER_ID = re.compile(r"^[A-Za-z0-9]{9}$")


class ScriptedModel(FakeMessagesListChatModel):
    calls: List[Any] = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return await super()._agenerate(messages, *args, **kwargs)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _model(*responses: AIMessage) -> ScriptedModel:
    return ScriptedModel(responses=list(responses), calls=[])


def _run_react(llm) -> List[Any]:
    graph = ReactAgent().compile(llm=llm, tools=[add], checkpointer=InMemorySaver())
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("Compute 2 + 3 + 4.")]},
            {"configurable": {"thread_id": "t"}},
        )
    )
    return out["messages"]


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _text(message: Any) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def assert_structured(sent: List[Any], rounds: int) -> None:
    """*sent* carries *rounds* tool calls as structured messages, no text copy."""
    calls = [m for m in sent if isinstance(m, AIMessage) and m.tool_calls]
    results = [m for m in sent if isinstance(m, ToolMessage)]
    assert len(calls) == rounds
    assert len(results) == rounds
    assert [m for m in sent if isinstance(m, HumanMessage)] == [sent[1]]
    for message in sent:
        assert "[TOOL_CALLS]" not in _text(message)
        assert "[TOOL_RESULTS]" not in _text(message)
    # Every tool result follows the call it answers, with the same id.
    for call, result in zip(calls, results):
        assert sent.index(result) == sent.index(call) + 1
        assert result.tool_call_id == call.tool_calls[0]["id"]


# ─── react graph ──────────────────────────────────────────────────────────────


def test_react_history_after_one_and_two_tool_calls_stays_structured() -> None:
    llm = _model(
        _call("add", {"a": 2, "b": 3}, "AbC123xyZ"),
        _call("add", {"a": 5, "b": 4}, "QwE456rtY"),
        AIMessage("The sum is 9."),
    )
    messages = _run_react(llm)

    assert messages[-1].content == "The sum is 9."
    assert len(llm.calls) == 3
    assert_structured(llm.calls[1], rounds=1)
    assert_structured(llm.calls[2], rounds=2)
    # Provider ids that already fit every provider are sent unchanged.
    assert [m.tool_calls[0]["id"] for m in llm.calls[2] if isinstance(m, AIMessage) and m.tool_calls] == [
        "AbC123xyZ",
        "QwE456rtY",
    ]


def test_react_text_calls_get_provider_ids_on_the_request_only() -> None:
    # Both replies parse to the id call_0_add: the parser counts per reply.
    llm = _model(
        AIMessage('[TOOL_CALLS]add[ARGS]{"a": 2, "b": 3}'),
        AIMessage('[TOOL_CALLS] [{"name": "add", "arguments": {"a": 5, "b": 4}}]'),
        AIMessage("The sum is 9."),
    )
    messages = _run_react(llm)

    assert messages[-1].content == "The sum is 9."
    assert len(llm.calls) == 3
    assert_structured(llm.calls[2], rounds=2)
    sent_ids = [m.tool_calls[0]["id"] for m in llm.calls[2] if isinstance(m, AIMessage) and m.tool_calls]
    assert all(_PROVIDER_ID.match(i) for i in sent_ids)
    assert len(set(sent_ids)) == 2
    # The thread keeps the ids the parser minted; only the request copy changes.
    kept = [m.tool_calls[0]["id"] for m in messages if isinstance(m, AIMessage) and m.tool_calls]
    assert kept == ["call_0_add", "call_0_add"]
    assert [m.content for m in messages if isinstance(m, ToolMessage)] == ["5", "9"]


def test_react_answer_that_mentions_the_marker_is_kept() -> None:
    answer = "The [TOOL_CALLS] token tells the server that a call follows."
    messages = _run_react(_model(AIMessage(answer)))
    assert messages[-1].content == answer
    assert not messages[-1].tool_calls


# ─── parse_text_tool_calls ────────────────────────────────────────────────────


def _parsed(content: str) -> List[tuple]:
    return [(c["name"], c["args"]) for c in utils.parse_text_tool_calls(content)]


def test_parses_name_and_object() -> None:
    assert _parsed('[TOOL_CALLS]search{"q": "ndvi"}') == [("search", {"q": "ndvi"})]


def test_parses_small_3_2_form_with_args_marker() -> None:
    assert _parsed('[TOOL_CALLS]search[ARGS]{"q": "ndvi"}') == [("search", {"q": "ndvi"})]


def test_parses_one_marker_per_call() -> None:
    content = '[TOOL_CALLS]a[ARGS]{"x": 1}[TOOL_CALLS]b[ARGS]{"y": 2}'
    assert _parsed(content) == [("a", {"x": 1}), ("b", {"y": 2})]
    ids = [c["id"] for c in utils.parse_text_tool_calls(content)]
    assert ids == ["call_0_a", "call_1_b"]


def test_parses_json_array_form_from_the_live_trace() -> None:
    content = '[TOOL_CALLS] [{"name": "execute", "arguments": {"command": "python compute_ndvi.py"}}]'
    assert _parsed(content) == [("execute", {"command": "python compute_ndvi.py"})]


def test_parses_json_array_with_several_calls_and_string_arguments() -> None:
    content = (
        '[TOOL_CALLS] [{"name": "a", "arguments": {"x": 1}}, '
        '{"name": "b", "arguments": "{\\"y\\": 2}"}, {"name": "c", "args": {"z": 3}}]'
    )
    assert _parsed(content) == [("a", {"x": 1}), ("b", {"y": 2}), ("c", {"z": 3})]


def test_brace_inside_a_string_argument() -> None:
    content = '[TOOL_CALLS]execute{"command": "echo \'}\' > out.txt"}'
    assert _parsed(content) == [("execute", {"command": "echo '}' > out.txt"})]


def test_invalid_json_escape_in_a_shell_command() -> None:
    content = '[TOOL_CALLS]execute[ARGS]{"command": "find . \\( -name x \\)"}'
    assert _parsed(content) == [("execute", {"command": "find . \\( -name x \\)"})]


def test_prose_that_mentions_the_marker_is_not_a_call() -> None:
    assert _parsed("The [TOOL_CALLS] token tells the server that a call follows.") == []
    assert _parsed("Calls start with [TOOL_CALLS] [1] in the old format.") == []
    assert _parsed("No marker at all.") == []


# ─── normalize_tool_call_ids ──────────────────────────────────────────────────


def test_provider_ids_are_left_alone_and_input_is_not_mutated() -> None:
    history = [
        HumanMessage("q"),
        _call("add", {"a": 1, "b": 2}, "AbC123xyZ"),
        ToolMessage(content="3", tool_call_id="AbC123xyZ"),
    ]
    out = utils.normalize_tool_call_ids(history)
    assert out[1].tool_calls[0]["id"] == "AbC123xyZ"
    assert out[2].tool_call_id == "AbC123xyZ"


def test_parser_ids_become_provider_ids_and_stay_paired() -> None:
    history = [
        HumanMessage("q"),
        _call("add", {"a": 2, "b": 3}, "call_0_add"),
        ToolMessage(content="5", tool_call_id="call_0_add"),
        _call("add", {"a": 5, "b": 4}, "call_0_add"),
        ToolMessage(content="9", tool_call_id="call_0_add"),
    ]
    out = utils.normalize_tool_call_ids(history)
    first, second = out[1].tool_calls[0]["id"], out[3].tool_calls[0]["id"]
    assert _PROVIDER_ID.match(first) and _PROVIDER_ID.match(second)
    assert first != second
    assert out[2].tool_call_id == first
    assert out[4].tool_call_id == second
    # Same history, same ids: the request is reproducible.
    again = utils.normalize_tool_call_ids(history)
    assert again[1].tool_calls[0]["id"] == first
    # The caller's messages keep their ids.
    assert history[1].tool_calls[0]["id"] == "call_0_add"
    assert history[2].tool_call_id == "call_0_add"
    assert out[2].content == "5" and out[4].content == "9"


# ─── answer_dangling_tool_calls ───────────────────────────────────────────────
# Mistral answers 400 "Not the same number of function calls and responses"
# when a call has no tool result, for example after a Stop during a tool.


def test_dangling_call_gets_an_error_result_on_the_request() -> None:
    history = [
        HumanMessage("q1"),
        _call("add", {"a": 2, "b": 3}, "AbC123xyZ"),
        HumanMessage("q2"),
    ]
    out = utils.answer_dangling_tool_calls(history)
    assert len(out) == 4
    assert isinstance(out[2], ToolMessage)
    assert out[2].tool_call_id == "AbC123xyZ"
    assert out[2].status == "error"
    assert out[3] is history[2]
    assert len(history) == 3


def test_only_the_missing_result_is_added_after_the_others() -> None:
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "add", "args": {"a": 1, "b": 1}, "id": "AAAAAAAA1"},
            {"name": "add", "args": {"a": 2, "b": 2}, "id": "AAAAAAAA2"},
        ],
    )
    history = [HumanMessage("q"), call, ToolMessage(content="2", tool_call_id="AAAAAAAA1")]
    out = utils.answer_dangling_tool_calls(history)
    assert [m.tool_call_id for m in out if isinstance(m, ToolMessage)] == ["AAAAAAAA1", "AAAAAAAA2"]
    assert out[2].content == "2"


def test_invalid_tool_call_is_answered_too() -> None:
    garbled = AIMessage(
        content="",
        invalid_tool_calls=[{"name": " [", "args": "{bad", "id": "BbBbBbBb1", "error": None}],
    )
    out = utils.answer_dangling_tool_calls([HumanMessage("q"), garbled, HumanMessage("again")])
    assert [m.tool_call_id for m in out if isinstance(m, ToolMessage)] == ["BbBbBbBb1"]


def test_answered_history_is_unchanged() -> None:
    history = [
        HumanMessage("q"),
        _call("add", {"a": 1, "b": 2}, "AbC123xyZ"),
        ToolMessage(content="3", tool_call_id="AbC123xyZ"),
    ]
    out = utils.answer_dangling_tool_calls(history)
    assert all(a is b for a, b in zip(out, history)) and len(out) == len(history)


def test_react_answers_a_call_left_open_by_the_previous_turn() -> None:
    llm = _model(AIMessage("7 + 1 is 8."))
    graph = ReactAgent().compile(llm=llm, tools=[add], checkpointer=InMemorySaver())
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage("Compute 2 + 3."),
                    _call("add", {"a": 2, "b": 3}, "AbC123xyZ"),
                    HumanMessage("Stop. Compute 7 + 1."),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    sent = llm.calls[0]
    call_at = next(i for i, m in enumerate(sent) if isinstance(m, AIMessage) and m.tool_calls)
    assert isinstance(sent[call_at + 1], ToolMessage)
    assert sent[call_at + 1].tool_call_id == "AbC123xyZ"
