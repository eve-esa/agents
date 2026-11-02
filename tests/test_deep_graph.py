"""Behaviour checks for the deep agent graph (needs the ``deep`` extra)."""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

pytest.importorskip("deepagents")

import deepagents.graph as deepagents_graph  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agents.graphs.deep import DeepAgent  # noqa: E402
from agents.graphs.deep import graph as deep_graph  # noqa: E402
from agents.graphs.policies import FALLBACK_MARKER  # noqa: E402
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

    async def __call__(self, *, policy, node=None, attempt=None, error=None, extra=None, **_):
        self.events.append((policy, attempt, (extra or {}).get("model")))

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
    assert "## Files, Subagents and Skills" in agent.prompts["system"]


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


def test_system_prompt_has_clock_and_packaged_skill() -> None:
    llm = _model(ScriptedModel, AIMessage("ok"))
    asyncio.run(_run(_compile(llm)))
    system = llm.calls[0][0]
    text = system.content if isinstance(system.content, str) else str(system.content)
    assert "## Clock" in text
    assert text.index("## Clock") < text.index("You are EVE")
    assert "eo-literature-review" in text


def test_skills_are_readable_and_read_only() -> None:
    llm = _model(
        ScriptedModel,
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/skills/eo-literature-review/SKILL.md"},
                    "id": "r1",
                },
                {
                    "name": "write_file",
                    "args": {"file_path": "/skills/evil/SKILL.md", "content": "x"},
                    "id": "w1",
                },
            ],
        ),
        AIMessage("done"),
    )
    messages = asyncio.run(_run(_compile(llm)))
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    assert "Earth Observation literature review" in str(results["r1"].content)
    assert "denied" in str(results["w1"].content).lower()
    assert not (deep_graph.PACKAGED_SKILLS_DIR / "evil").exists()


def test_skills_dir_env_override(tmp_path, monkeypatch) -> None:
    skill = tmp_path / "custom-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: custom-skill\ndescription: A custom test skill.\n---\nBody\n"
    )
    monkeypatch.setenv(deep_graph.SKILLS_DIR_ENV, str(tmp_path))
    llm = _model(ScriptedModel, AIMessage("ok"))
    asyncio.run(_run(_compile(llm)))
    text = str(llm.calls[0][0].content)
    assert "custom-skill" in text
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
        "SkillsMiddleware",
        "AnthropicPromptCachingMiddleware",
        "UnsupportedContentMiddleware",
    ]


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
        assert names[:4] == [
            "EveFallbackMiddleware",
            "EveRetryMiddleware",
            "EveCompatMiddleware",
            "ToolErrorMiddleware",
        ]
    assert specs["general-purpose"]["skills"] == ["/skills/"]
