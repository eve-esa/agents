"""Behaviour checks for the deep agent graph (needs the ``deep`` extra)."""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

pytest.importorskip("deepagents")

import deepagents.graph as deepagents_graph  # noqa: E402
import deepagents.middleware.subagents as deepagents_subagents  # noqa: E402
from deepagents.backends import StateBackend  # noqa: E402
from deepagents.middleware.filesystem import (  # noqa: E402
    FilesystemMiddleware,
    FilesystemPermission,
)
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agents.graphs.deep import DeepAgent  # noqa: E402
from agents.graphs.deep import graph as deep_graph  # noqa: E402
from langchain.agents.middleware import ModelRequest, ModelResponse  # noqa: E402

from agents.graphs.deep.middleware import (  # noqa: E402
    DEFAULT_TOOL_RESULT_MAX_TOKENS,
    EveCompatMiddleware,
    EveSummarizationMiddleware,
)
from agents.graphs.policies import FALLBACK_LATENCY_KEY, FALLBACK_MARKER  # noqa: E402
from agents.graphs.react.graph import ReactAgent  # noqa: E402


class ScriptedModel(FakeMessagesListChatModel):
    """Returns *responses* in order and records the messages of every call."""

    calls: List[Any] = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return await super()._agenerate(messages, *args, **kwargs)


class FailingModel(ScriptedModel):
    error: Any = ConnectionError("provider down")

    async def _agenerate(self, messages, *args, **kwargs):
        self.calls.append(messages)
        raise self.error


class SlowModel(ScriptedModel):
    async def _agenerate(self, messages, *args, **kwargs):
        await asyncio.sleep(1)
        return await super()._agenerate(messages, *args, **kwargs)


def _model(cls=ScriptedModel, *responses: AIMessage, **kwargs):
    return cls(responses=list(responses) or [AIMessage("unused")], calls=[], **kwargs)


class PolicyLog:
    def __init__(self) -> None:
        self.events: List[tuple] = []
        self.extras: List[dict] = []

    async def __call__(self, *, policy, node=None, attempt=None, error=None, extra=None, **_):
        self.events.append((policy, attempt, (extra or {}).get("model")))
        self.extras.append(dict(extra or {}))

    @property
    def policies(self) -> List[str]:
        return [e[0] for e in self.events]


async def _run(graph, text: str = "hi") -> List[Any]:
    out = await graph.ainvoke(
        {"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": "t"}}
    )
    return out["messages"]


def _compile(llm, **kwargs):
    kwargs.setdefault("tools", [])
    return DeepAgent().compile(llm=llm, checkpointer=InMemorySaver(), **kwargs)


def test_prompt_is_react_prompt_plus_addendum() -> None:
    agent = DeepAgent()
    react_system = ReactAgent().prompts["system"].strip()
    assert agent.name == "deep"
    assert agent.prompts["system"].startswith(react_system)
    assert "## Files and Subagents" in agent.prompts["system"]


def test_transient_error_retries_then_falls_back_with_marker() -> None:
    log = PolicyLog()
    primary = _model(FailingModel)
    fallback = _model(ScriptedModel, AIMessage("from fallback"))
    messages = asyncio.run(
        _run(_compile(primary, fallback_llm=fallback, on_policy=log))
    )

    assert len(primary.calls) == 2
    assert log.policies == ["retry", "error_handler", "fallback"]
    assert log.events[0][1] == 2
    answer = messages[-1]
    assert answer.content == "from fallback"
    assert answer.response_metadata[FALLBACK_MARKER] is True
    assert answer.response_metadata[FALLBACK_LATENCY_KEY] >= 0
    # The primary failed after a retry with backoff, so it lost measurable time.
    assert log.extras[1]["failed_after_s"] > 0
    assert log.extras[2]["failed_after_s"] == log.extras[1]["failed_after_s"]


def test_non_transient_error_skips_retry() -> None:
    log = PolicyLog()
    primary = _model(FailingModel, error=ValueError("bad request"))
    fallback = _model(ScriptedModel, AIMessage("from fallback"))
    asyncio.run(_run(_compile(primary, fallback_llm=fallback, on_policy=log)))

    assert len(primary.calls) == 1
    assert log.policies == ["error_handler", "fallback"]


def test_without_fallback_the_error_is_reported_and_raised() -> None:
    log = PolicyLog()
    primary = _model(FailingModel, error=ValueError("bad request"))
    with pytest.raises(ValueError, match="bad request"):
        asyncio.run(_run(_compile(primary, on_policy=log)))
    assert log.policies == ["error_handler"]


def test_primary_answer_has_no_fallback_marker() -> None:
    messages = asyncio.run(_run(_compile(_model(ScriptedModel, AIMessage("ok")))))
    assert FALLBACK_MARKER not in messages[-1].response_metadata
    assert FALLBACK_LATENCY_KEY not in messages[-1].response_metadata


def test_run_timeout_is_reported_as_timeout() -> None:
    log = PolicyLog()
    primary = _model(SlowModel, AIMessage("too late"), AIMessage("too late"))
    fallback = _model(ScriptedModel, AIMessage("from fallback"))
    messages = asyncio.run(
        _run(
            _compile(
                primary, fallback_llm=fallback, on_policy=log, llm_run_timeout=0.05
            )
        )
    )
    assert log.policies == ["retry", "timeout", "fallback"]
    assert messages[-1].content == "from fallback"


def test_tool_error_becomes_error_tool_message() -> None:
    @tool
    def broken(x: str) -> str:
        """Always fails."""
        raise RuntimeError("tool exploded")

    llm = _model(
        ScriptedModel,
        AIMessage(content="", tool_calls=[{"name": "broken", "args": {"x": "1"}, "id": "c1"}]),
        AIMessage("recovered"),
    )
    messages = asyncio.run(_run(_compile(llm, tools=[broken])))
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert tool_message.content == "Tool error: tool exploded"
    assert messages[-1].content == "recovered"


def test_text_tool_calls_are_parsed() -> None:
    @tool
    def ping(x: str) -> str:
        """Echo."""
        return f"pong {x}"

    llm = _model(
        ScriptedModel,
        AIMessage(content='[TOOL_CALLS]ping{"x": "1"}'),
        AIMessage("done"),
    )
    messages = asyncio.run(_run(_compile(llm, tools=[ping])))
    call = next(m for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    assert call.tool_calls[0]["name"] == "ping"
    assert call.tool_calls[0]["args"] == {"x": "1"}
    assert any(isinstance(m, ToolMessage) and m.content == "pong 1" for m in messages)
    # The second model call sees the tool history in text form.
    assert not any(isinstance(m, ToolMessage) for m in llm.calls[1])


def test_system_prompt_has_clock() -> None:
    llm = _model(ScriptedModel, AIMessage("ok"))
    asyncio.run(_run(_compile(llm)))
    system = llm.calls[0][0]
    text = system.content if isinstance(system.content, str) else str(system.content)
    assert "## Clock" in text
    assert text.index("## Clock") < text.index("You are EVE")
    assert "eo-literature-review" not in text


def _captured_create_agent(monkeypatch) -> List[dict]:
    captured: List[dict] = []
    real = deepagents_graph.create_agent

    def spy(model, **kwargs):
        captured.append(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(deepagents_graph, "create_agent", spy)
    return captured


def test_middleware_stack_snapshot(monkeypatch) -> None:
    """Fails when a deepagents upgrade reorders the stack around EVE middleware."""
    captured = _captured_create_agent(monkeypatch)
    _compile(_model(), fallback_llm=_model())
    names = [m.name for m in captured[-1]["middleware"]]
    assert names == [
        "FilesystemMiddleware",
        "SubAgentMiddleware",
        "_ParentNotifierMiddleware",
        "SummarizationMiddleware",
        "PatchToolCallsMiddleware",
        "EveFallbackMiddleware",
        "EveRetryMiddleware",
        "EveCompatMiddleware",
        "ToolErrorMiddleware",
        "AnthropicPromptCachingMiddleware",
        "UnsupportedContentMiddleware",
        "ModelRefusalMiddleware",
    ]


def test_every_agent_summarizes_with_the_eve_middleware(monkeypatch) -> None:
    captured = _captured_create_agent(monkeypatch)
    real = deepagents_subagents.create_agent

    def spy(model, **kwargs):
        captured.append(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(deepagents_subagents, "create_agent", spy)
    custom = {"name": "researcher", "description": "Researches.", "system_prompt": "Research."}
    _compile(_model(), subagents=[custom])
    assert len(captured) >= 3
    for kwargs in captured:
        slot = [m for m in kwargs["middleware"] if m.name == "SummarizationMiddleware"]
        assert len(slot) == 1
        assert isinstance(slot[0], EveSummarizationMiddleware)


def test_every_agent_offloads_large_tool_results(monkeypatch) -> None:
    captured = _captured_create_agent(monkeypatch)
    real = deepagents_subagents.create_agent

    def spy(model, **kwargs):
        captured.append(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(deepagents_subagents, "create_agent", spy)
    custom = {"name": "researcher", "description": "Researches.", "system_prompt": "Research."}
    backend = deep_graph.default_backend()
    _compile(_model(), subagents=[custom], backend=backend)
    assert len(captured) >= 3
    for kwargs in captured:
        slot = [m for m in kwargs["middleware"] if m.name == "FilesystemMiddleware"]
        assert len(slot) == 1
        assert slot[0].backend is backend
        assert slot[0]._tool_token_limit_before_evict == DEFAULT_TOOL_RESULT_MAX_TOKENS
        assert slot[0]._human_message_token_limit_before_evict is None


def _large_search():
    big = "chunk " * 30_000

    @tool
    def search(q: str) -> str:
        """Returns a large result."""
        return big

    return big, search


def test_large_tool_result_is_offloaded() -> None:
    big, search = _large_search()
    llm = _model(
        ScriptedModel,
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "s1"}]),
        AIMessage("done"),
    )
    graph = _compile(llm, tools=[search])
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "t"}}
        )
    )
    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "/large_tool_results/s1" in result.content
    assert big not in result.content
    assert out["files"]["/large_tool_results/s1"]["content"] == big


def test_dense_tool_result_is_offloaded_by_tiktoken() -> None:
    """CJK is one token per character. The character estimate would keep this.

    120 characters is under the deepagents budget of 4 characters per token
    (200 at a limit of 50) and over 50 tiktoken tokens.  Forty lines means the
    head-and-tail preview cannot hold the whole result.
    """
    dense = "你你\n" * 40

    @tool
    def search(q: str) -> str:
        """Returns a dense result."""
        return dense

    llm = _model(
        ScriptedModel,
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "s1"}]),
        AIMessage("done"),
    )
    graph = _compile(llm, tools=[search], tool_result_max_tokens=50)
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "t"}}
        )
    )
    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "/large_tool_results/s1" in result.content
    assert dense not in result.content
    assert out["files"]["/large_tool_results/s1"]["content"] == dense


def test_subagent_filesystem_keeps_backend_and_offload_limit(monkeypatch) -> None:
    captured = _captured_create_agent(monkeypatch)
    real = deepagents_subagents.create_agent

    def spy(model, **kwargs):
        captured.append(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(deepagents_subagents, "create_agent", spy)
    other = deep_graph.StateBackend()
    permission = FilesystemPermission(operations=["read"], paths=["/notes/**"], mode="allow")
    custom = {
        "name": "researcher",
        "description": "Researches.",
        "system_prompt": "Research.",
        "middleware": [
            FilesystemMiddleware(
                backend=other,
                tool_token_limit_before_evict=None,
                human_message_token_limit_before_evict=5,
                permissions=[permission],
            )
        ],
    }
    _compile(_model(), subagents=[custom], tool_result_max_tokens=1234)
    researcher = next(kwargs for kwargs in captured if kwargs.get("system_prompt") == "Research.")
    slot = [m for m in researcher["middleware"] if m.name == "FilesystemMiddleware"]
    assert len(slot) == 1
    assert slot[0].backend is other
    assert slot[0]._permissions == [permission]
    assert slot[0]._tool_token_limit_before_evict == 1234
    assert slot[0]._human_message_token_limit_before_evict is None


def test_tool_result_offload_can_be_turned_off() -> None:
    big, search = _large_search()
    llm = _model(
        ScriptedModel,
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "s1"}]),
        AIMessage("done"),
    )
    graph = _compile(llm, tools=[search], tool_result_max_tokens=None)
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "t"}}
        )
    )
    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert result.content == big
    assert not any("large_tool_results" in path for path in out.get("files") or {})


def test_offloaded_tool_result_is_read_in_pages() -> None:
    lines = [f"line-{i:05d}" for i in range(25_000)]
    big = "\n".join(lines)
    path = "/large_tool_results/s1"

    @tool
    def search(q: str) -> str:
        """Returns a large result."""
        return big

    llm = _model(
        ScriptedModel,
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "s1"}]),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": path, "offset": 10, "limit": 3},
                    "id": "r1",
                }
            ],
        ),
        AIMessage("done"),
    )
    graph = _compile(llm, tools=[search])
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "t"}}
        )
    )
    page = next(m for m in out["messages"] if getattr(m, "tool_call_id", None) == "r1")
    text = page.content if isinstance(page.content, str) else str(page.content)
    assert "line-00010" in text
    assert "line-00012" in text
    assert "line-00009" not in text
    assert "line-00013" not in text
    assert len(text) < len(big)


def _text(message: Any) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _joined(messages: List[Any]) -> str:
    return "\n".join(_text(message) for message in messages)


# Small enough that one long turn crosses 0.85 of it, and large enough that
# the system prompt plus the built-in tool schemas stay under that line.
_WINDOW = 20_000


def test_short_history_is_not_summarized() -> None:
    llm = _model(ScriptedModel, AIMessage("ok"))
    graph = _compile(llm)
    out = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage("new question")]},
            {"configurable": {"thread_id": "t"}},
        )
    )
    assert len(llm.calls) == 1
    sent = llm.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert "been summarized" not in _joined(sent)
    assert [m.content for m in sent[1:]] == ["new question"]
    assert any(getattr(m, "content", None) == "new question" for m in out["messages"])


def test_over_the_trigger_summarizes_and_keeps_the_tail() -> None:
    llm = _model(ScriptedModel, AIMessage("kept the decision"), AIMessage("ok"))
    graph = _compile(
        llm,
        context_window=_WINDOW,
        recall_note="Use search_history to recover a dropped fact.",
    )
    old = "oldfact " * 8_000
    newer = "newerfact " * 8_000
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(old, additional_kwargs={"eve_message_id": "id-old"}),
                    AIMessage("old answer"),
                    HumanMessage(newer, additional_kwargs={"eve_message_id": "id-new"}),
                    AIMessage("second answer"),
                    HumanMessage(
                        "new question", additional_kwargs={"eve_message_id": "id-tail"}
                    ),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    assert len(llm.calls) == 2
    sent = llm.calls[1]
    body = _joined(sent)
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], HumanMessage)
    assert sent[1].additional_kwargs.get("lc_source") == "summarization"
    assert "kept the decision" in _text(sent[1])
    assert "Summarized message_id id-old through id-new." in _text(sent[1])
    assert "Use search_history to recover a dropped fact." in _text(sent[1])
    assert "/conversation_history/" not in _text(sent[1])
    assert "new question" in body
    assert old not in body
    assert newer not in body
    stored = _joined(
        graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    )
    assert old in stored
    assert newer in stored


def test_the_summary_message_carries_the_text_and_range() -> None:
    llm = _model(ScriptedModel, AIMessage("first summary"), AIMessage("ok"))
    graph = _compile(llm, context_window=_WINDOW)
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage("oldfact " * 8_000, additional_kwargs={"eve_message_id": "a"}),
                    AIMessage("x"),
                    HumanMessage("newerfact " * 8_000, additional_kwargs={"eve_message_id": "b"}),
                    AIMessage("y"),
                    HumanMessage("q", additional_kwargs={"eve_message_id": "c"}),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    event = graph.get_state({"configurable": {"thread_id": "t"}}).values[
        "_summarization_event"
    ]
    extra = event["summary_message"].additional_kwargs
    assert extra["eve_summary"] == "first summary"
    assert extra["eve_summary_first_message_id"] == "a"
    assert extra["eve_summary_last_message_id"] == "b"


def test_a_second_summary_keeps_the_start_of_the_first_range() -> None:
    llm = _model(
        ScriptedModel,
        AIMessage("first summary"),
        AIMessage("ok"),
        AIMessage("second summary"),
        AIMessage("ok again"),
    )
    graph = _compile(llm, context_window=_WINDOW)
    config = {"configurable": {"thread_id": "t"}}
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage("oldfact " * 8_000, additional_kwargs={"eve_message_id": "a"}),
                    AIMessage("x"),
                    HumanMessage("newerfact " * 8_000, additional_kwargs={"eve_message_id": "b"}),
                    AIMessage("y"),
                    HumanMessage("q", additional_kwargs={"eve_message_id": "c"}),
                ]
            },
            config,
        )
    )
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage("laterfact " * 14_000, additional_kwargs={"eve_message_id": "d"}),
                ]
            },
            config,
        )
    )
    extra = graph.get_state(config).values["_summarization_event"][
        "summary_message"
    ].additional_kwargs
    assert extra["eve_summary"] == "second summary"
    assert extra["eve_summary_first_message_id"] == "a"
    assert extra["eve_summary_last_message_id"] in {"c", "d"}


def _summarizer_prompt(llm) -> str:
    """Text the summarization call received: the first recorded call."""
    first = llm.calls[0]
    return "\n".join(str(m.content) for m in first)


def test_the_summarizer_uses_the_eve_prompt() -> None:
    llm = _model(ScriptedModel, AIMessage("summary"), AIMessage("ok"))
    graph = _compile(llm, context_window=_WINDOW)
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage("oldfact " * 8_000, additional_kwargs={"eve_message_id": "a"}),
                    AIMessage("x"),
                    HumanMessage("q", additional_kwargs={"eve_message_id": "b"}),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    prompt = _summarizer_prompt(llm)
    assert "the Earth Observation assistant of ESA" in prompt
    assert "## Facts and values" in prompt
    assert "SESSION INTENT" not in prompt
    assert "oldfact" in prompt


def test_system_messages_are_not_summarized() -> None:
    llm = _model(ScriptedModel, AIMessage("summary"), AIMessage("ok"))
    graph = _compile(llm, context_window=_WINDOW)
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    SystemMessage("RULE-ZX9: always cite sources."),
                    HumanMessage("oldfact " * 8_000, additional_kwargs={"eve_message_id": "a"}),
                    AIMessage("x"),
                    HumanMessage("q", additional_kwargs={"eve_message_id": "b"}),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    prompt = _summarizer_prompt(llm)
    assert "oldfact" in prompt
    assert "RULE-ZX9" not in prompt


def test_a_prompt_without_the_messages_slot_is_refused() -> None:
    with pytest.raises(ValueError):
        EveSummarizationMiddleware(
            _model(), StateBackend(), context_window=_WINDOW, summary_prompt="no slot"
        )


def test_old_write_file_args_are_cut_before_summarizing() -> None:
    llm = _model(ScriptedModel, AIMessage("ok"))
    graph = _compile(llm, context_window=_WINDOW, truncate_args_max_chars=2_000)
    old_body = "Z" * 80_000
    recent_body = "Q" * 5_000
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "old.py", "content": old_body},
                                "id": "w1",
                            }
                        ],
                    ),
                    ToolMessage(content="wrote old", tool_call_id="w1"),
                    HumanMessage("recent"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "new.py", "content": recent_body},
                                "id": "w2",
                            }
                        ],
                    ),
                    ToolMessage(content="wrote new", tool_call_id="w2"),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    assert len(llm.calls) == 1
    # EveCompat rewrites tool calls as text before the model sees them.
    body = _joined(llm.calls[0])
    assert "been summarized" not in body
    assert "Z" * 80_000 not in body
    assert "Z" * 20 in body
    assert "argument truncated" in body
    assert recent_body in body


def test_a_tool_batch_is_not_split() -> None:
    llm = _model(ScriptedModel, AIMessage("batch summary"), AIMessage("ok"))
    graph = _compile(llm, context_window=_WINDOW, summary_keep_fraction=0.02)
    old = "oldfact " * 12_000
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(old, additional_kwargs={"eve_message_id": "id-old"}),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "search", "args": {}, "id": "c1"},
                            {"name": "search", "args": {}, "id": "c2"},
                        ],
                    ),
                    ToolMessage(content="result-one", tool_call_id="c1"),
                    ToolMessage(content="result-two", tool_call_id="c2"),
                    HumanMessage("new question"),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    body = _joined(llm.calls[-1])
    assert old not in body
    assert "result-one" in body
    assert "result-two" in body
    calls = [message for message in llm.calls[-1] if "[TOOL_CALLS]" in _text(message)]
    assert len(calls) == 1
    assert _text(calls[0]).count('"name": "search"') == 2


def test_an_offloaded_result_stays_a_stub_in_the_summary_input() -> None:
    llm = _model(ScriptedModel, AIMessage("saw the stub"), AIMessage("ok"))
    graph = _compile(llm, context_window=_WINDOW)
    payload = "PAYLOAD " * 4_000
    stub = "Result saved to /large_tool_results/s1. Preview: station."
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        "lookup", additional_kwargs={"eve_message_id": "id-tool"}
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "s1"}],
                    ),
                    ToolMessage(content=stub, tool_call_id="s1", name="search"),
                    HumanMessage("oldfact " * 8_000),
                    HumanMessage("new question"),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    summary_input = _joined(llm.calls[0])
    assert stub in summary_input
    assert payload not in summary_input


def test_general_purpose_subagent_gets_eve_middleware(monkeypatch) -> None:
    captured: List[dict] = []
    real = deep_graph.create_deep_agent

    def spy(**kwargs):
        captured.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(deep_graph, "create_deep_agent", spy)
    custom = {"name": "researcher", "description": "Researches.", "system_prompt": "Research."}
    _compile(_model(), fallback_llm=_model(), subagents=[custom])

    specs = {s["name"]: s for s in captured[-1]["subagents"]}
    assert set(specs) == {"general-purpose", "researcher"}
    for spec in specs.values():
        names = [m.name for m in spec["middleware"]]
        assert names[:6] == [
            "FilesystemMiddleware",
            "SummarizationMiddleware",
            "EveFallbackMiddleware",
            "EveRetryMiddleware",
            "EveCompatMiddleware",
            "ToolErrorMiddleware",
        ]
    assert "skills" not in specs["general-purpose"]
    assert captured[-1]["skills"] is None


def test_a_later_turn_does_not_send_the_agent_name_to_the_provider() -> None:
    """Mistral rejects ``name`` on an assistant message. The thread keeps it."""
    stored = AIMessage(content="previous answer", name="deep")
    seen: dict = {}

    def handler(request: ModelRequest) -> ModelResponse:
        seen["messages"] = request.messages
        return ModelResponse(result=[AIMessage(content="ok")])

    EveCompatMiddleware().wrap_model_call(
        ModelRequest(
            model=_model(),
            messages=[HumanMessage("again"), stored],
        ),
        handler,
    )

    sent = seen["messages"][1]
    assert sent.name is None
    assert sent.content == "previous answer"
    assert stored.name == "deep"
