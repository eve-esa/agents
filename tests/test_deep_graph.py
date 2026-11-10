"""Behaviour checks for the deep agent graph (needs the ``deep`` extra)."""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

pytest.importorskip("deepagents")

import deepagents.graph as deepagents_graph  # noqa: E402
import deepagents.middleware.subagents as deepagents_subagents  # noqa: E402
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
from agents.graphs.deep.middleware import (  # noqa: E402
    DEFAULT_TOOL_RESULT_MAX_TOKENS,
    EveTrimMiddleware,
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
    ]


def test_every_agent_trims_instead_of_summarizing(monkeypatch) -> None:
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
        assert isinstance(slot[0], EveTrimMiddleware)


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


def test_history_is_trimmed_to_max_tokens() -> None:
    llm = _model(ScriptedModel, AIMessage("ok"))
    graph = _compile(llm, max_tokens=20_000)
    old = "old " * 40_000
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    HumanMessage(old),
                    AIMessage("old answer"),
                    HumanMessage("new question"),
                ]
            },
            {"configurable": {"thread_id": "t"}},
        )
    )
    sent = llm.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert "You are EVE" in str(sent[0].content)
    assert [m.content for m in sent[1:]] == ["new question"]


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
