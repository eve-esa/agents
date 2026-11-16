"""Malformed tool calls, old text-form history and the shared parser vectors.

A structured call whose arguments are not JSON (a shell ``\\(``) arrives from
ChatOpenAI as ``invalid_tool_calls`` with no ``tool_calls``; the graphs repair
it with the escape repair of the text parser, or answer it with an error
result and let the model try again, so the turn never ends empty.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs import utils
from agents.graphs.deep import DeepAgent
from agents.graphs.react.graph import ReactAgent
from agents.graphs.text_tool_call_vectors import TEXT_TOOL_CALL_VECTORS


class ScriptedModel(FakeMessagesListChatModel):
    calls: List[Any] = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return await super()._agenerate(messages, *args, **kwargs)


@tool
def execute(command: str) -> str:
    """Run a shell command."""
    return f"ran {command}"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


SHELL_ARGS = '{"command": "find . \\( -name x \\)"}'
SHELL_COMMAND = "find . \\( -name x \\)"


def _invalid(args: str, name: str = "execute", call_id: str = "AbC123xyZ") -> AIMessage:
    return AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": name, "args": args, "id": call_id, "error": "bad json", "type": "invalid_tool_call"}
        ],
    )


def _model(*responses: AIMessage) -> ScriptedModel:
    return ScriptedModel(responses=list(responses), calls=[])


def _run(graph, messages: List[Any]) -> List[Any]:
    out = asyncio.run(graph.ainvoke({"messages": messages}, {"configurable": {"thread_id": "t"}}))
    return out["messages"]


def _react(llm, tools=(execute, add)):
    return ReactAgent().compile(llm=llm, tools=list(tools), checkpointer=InMemorySaver())


def _deep(llm, tools=(execute, add)):
    return DeepAgent().compile(llm=llm, tools=list(tools), checkpointer=InMemorySaver())


def _text(message: Any) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


GRAPHS = [pytest.param(_react, id="react"), pytest.param(_deep, id="deep")]


# ─── Parser vectors ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("vector", TEXT_TOOL_CALL_VECTORS, ids=[v["name"] for v in TEXT_TOOL_CALL_VECTORS])
def test_parser_vectors(vector) -> None:
    calls = utils.parse_text_tool_calls(vector["content"], allowed_names=vector["tools"])
    assert [[c["name"], c["args"]] for c in calls] == vector["expected"]


# ─── 1. Invalid structured calls ──────────────────────────────────────────────


@pytest.mark.parametrize("build", GRAPHS)
def test_shell_escaped_structured_call_is_repaired_and_run(build) -> None:
    llm = _model(_invalid(SHELL_ARGS), AIMessage("Done."))
    messages = _run(build(llm), [HumanMessage("Find x.")])
    call = next(m for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    assert call.tool_calls[0]["name"] == "execute"
    assert call.tool_calls[0]["args"] == {"command": SHELL_COMMAND}
    assert call.tool_calls[0]["id"] == "AbC123xyZ"
    assert not call.invalid_tool_calls
    assert any(isinstance(m, ToolMessage) and m.content == f"ran {SHELL_COMMAND}" for m in messages)
    assert messages[-1].content == "Done."


@pytest.mark.parametrize("build", GRAPHS)
def test_unrepairable_call_gets_a_malformed_error_and_the_model_retries(build) -> None:
    llm = _model(_invalid('{"command": "ls'), AIMessage("Sorry, here is the answer."))
    messages = _run(build(llm), [HumanMessage("List.")])
    errors = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(errors) == 1
    assert errors[0].tool_call_id == "AbC123xyZ"
    assert "malformed" in errors[0].content
    assert errors[0].status == "error"
    assert messages[-1].content == "Sorry, here is the answer."
    # The retry request carries the bad call as a structured call plus its error.
    retry = llm.calls[1]
    call = next(m for m in retry if isinstance(m, AIMessage) and m.tool_calls)
    assert call.tool_calls[0]["args"] == {}
    assert not call.invalid_tool_calls
    assert retry[retry.index(call) + 1].tool_call_id == call.tool_calls[0]["id"]


@pytest.mark.parametrize("build", GRAPHS)
def test_turn_never_ends_empty_when_every_retry_is_malformed(build) -> None:
    bad = [_invalid("{oops", call_id=f"Bad{n}xxxxx"[:9]) for n in range(5)]
    llm = _model(*bad)
    messages = _run(build(llm), [HumanMessage("List.")])
    assert len(llm.calls) == 1 + utils.MALFORMED_TOOL_CALL_RETRIES
    last = messages[-1]
    assert isinstance(last, AIMessage) and not last.tool_calls
    assert "malformed" in _text(last)
    # Every bad call has its error result, so the thread stays well formed.
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    asked = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.invalid_tool_calls}
    assert asked and asked <= answered


def test_reply_helper_promotes_and_keeps_metadata() -> None:
    reply = _invalid(SHELL_ARGS).model_copy(update={"usage_metadata": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}})
    fixed = utils.structure_model_reply(reply, allowed_names={"execute"})
    assert fixed.tool_calls[0]["args"] == {"command": SHELL_COMMAND}
    assert fixed.usage_metadata["total_tokens"] == 5
    assert reply.invalid_tool_calls and not reply.tool_calls


# ─── 2. Strip considers invalid calls ─────────────────────────────────────────


def test_strip_also_clears_text_next_to_invalid_calls() -> None:
    message = _invalid("{bad").model_copy(update={"content": "Let me run it."})
    out = utils.strip_content_from_tool_call_messages([message])
    assert out[0].content == ""
    assert out[0].invalid_tool_calls == message.invalid_tool_calls
    assert message.content == "Let me run it."


def test_request_copy_sends_no_invalid_calls() -> None:
    history = [HumanMessage("q"), _invalid("{bad").model_copy(update={"content": "Running."}), HumanMessage("again")]
    out = utils.prepare_model_messages(history)
    call = next(m for m in out if isinstance(m, AIMessage))
    assert call.content == "" and not call.invalid_tool_calls
    assert call.tool_calls[0]["name"] == "execute" and call.tool_calls[0]["args"] == {}
    assert isinstance(out[out.index(call) + 1], ToolMessage)


def test_invalid_call_with_a_garbled_name_is_dropped_from_the_request() -> None:
    history = [
        HumanMessage("q"),
        _invalid("{bad", name=" ["),
        ToolMessage(content="err", tool_call_id="AbC123xyZ"),
        HumanMessage("again"),
    ]
    out = utils.prepare_model_messages(history)
    assert [type(m).__name__ for m in out] == ["HumanMessage", "HumanMessage"]


# ─── 3. Old text-form calls in history ────────────────────────────────────────


F1_REPLY = '[TOOL_CALLS] [{"name": "add", "arguments": {"a": 2, "b": 3}}]'


@pytest.mark.parametrize("build", GRAPHS)
def test_old_text_call_in_history_is_never_sent_as_text(build) -> None:
    llm = _model(AIMessage("5."))
    _run(
        build(llm),
        [HumanMessage("Compute 2 + 3."), AIMessage(F1_REPLY), HumanMessage("Well?")],
    )
    sent = llm.calls[0]
    assert not any("[TOOL_CALLS]" in _text(m) for m in sent)
    call = next(m for m in sent if isinstance(m, AIMessage) and m.tool_calls)
    assert call.tool_calls[0]["name"] == "add"
    result = sent[sent.index(call) + 1]
    assert isinstance(result, ToolMessage) and result.tool_call_id == call.tool_calls[0]["id"]


def test_old_text_call_with_its_results_keeps_the_pairing() -> None:
    history = [
        HumanMessage("Compute 2 + 3."),
        AIMessage(F1_REPLY),
        ToolMessage(content="5", tool_call_id="call_0_add"),
        HumanMessage("And?"),
    ]
    out = utils.prepare_model_messages(history, allowed_names={"add"})
    call = out[1]
    assert call.content == "" and call.tool_calls[0]["name"] == "add"
    assert out[2].content == "5" and out[2].tool_call_id == call.tool_calls[0]["id"]
    assert len(out) == 4


def test_old_unparseable_marker_text_is_blanked() -> None:
    history = [HumanMessage("q"), AIMessage("[TOOL_CALLS]broken{"), HumanMessage("again")]
    out = utils.prepare_model_messages(history, allowed_names={"add"})
    assert not any("[TOOL_CALLS]" in _text(m) for m in out)
    assert all(_text(m) for m in out if isinstance(m, AIMessage))


def test_answer_that_mentions_the_marker_mid_text_is_kept() -> None:
    answer = AIMessage("Mistral writes [TOOL_CALLS] before a call.")
    out = utils.prepare_model_messages([HumanMessage("q"), answer], allowed_names={"add"})
    assert out[1].content == answer.content


# ─── 4. Nits ──────────────────────────────────────────────────────────────────


def test_react_unbound_name_after_marker_is_an_answer() -> None:
    llm = _model(AIMessage('The [TOOL_CALLS] token {"a": 1} starts a call.'))
    messages = _run(_react(llm), [HumanMessage("What is the marker?")])
    assert messages[-1].content.startswith("The [TOOL_CALLS] token")
    assert not any(isinstance(m, ToolMessage) for m in messages)


def test_dangling_invalid_call_is_told_its_arguments_were_malformed() -> None:
    out = utils.answer_dangling_tool_calls([HumanMessage("q"), _invalid("{bad"), HumanMessage("again")])
    error = next(m for m in out if isinstance(m, ToolMessage))
    assert "malformed" in error.content
    assert "did not complete" not in error.content


def test_orphan_and_duplicate_tool_messages_are_dropped() -> None:
    call = AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "AbC123xyZ"}])
    history = [
        HumanMessage("q"),
        ToolMessage(content="orphan", tool_call_id="ZzZ999zzZ"),
        call,
        ToolMessage(content="3", tool_call_id="AbC123xyZ"),
        ToolMessage(content="3 again", tool_call_id="AbC123xyZ"),
        HumanMessage("next"),
        ToolMessage(content="late", tool_call_id="AbC123xyZ"),
    ]
    out = utils.answer_dangling_tool_calls(history)
    assert [m.content for m in out if isinstance(m, ToolMessage)] == ["3"]
    assert len(out) == 4
