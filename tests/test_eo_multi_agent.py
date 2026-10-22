"""Checks for the EO multi-agent graph: team, plan, verifier and one full run."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agents.graphs.eo_multi_agent.graph import EOMultiAgent
from agents.graphs.eo_multi_agent.plan import Playbook, parse_plan, usable_playbooks
from agents.graphs.eo_multi_agent.team import (
    GENERALIST,
    Specialist,
    assign_tools,
    owner,
)
from agents.graphs.eo_multi_agent.verify import clean_links, unsupported_numbers

TEAM = [
    Specialist("fire", "", ("effis", "wildfire_flood")),
    Specialist("wild", "", ("wildfire",)),
    Specialist("research", "", ("*",)),
]


# ── team ──────────────────────────────────────────────────────────────────────


def test_owner_prefers_longest_server_and_catch_all() -> None:
    assert owner("wildfire_flood_burn_severity", TEAM) == "fire"
    assert owner("wildfire_other_tool", TEAM) == "wild"
    assert owner("geocode_search", TEAM) == "research"
    assert owner("geocode_search", TEAM[:2]) is None


def test_assign_tools_gives_generalist_everything() -> None:
    @tool("effis_get_fires")
    def fires() -> str:
        """Find fires."""
        return ""

    @tool("geocode_search")
    def geocode() -> str:
        """Find places."""
        return ""

    groups = assign_tools([fires, geocode], TEAM)
    assert [t.name for t in groups["fire"]] == ["effis_get_fires"]
    assert [t.name for t in groups["research"]] == ["geocode_search"]
    assert len(groups[GENERALIST]) == 2
    assert assign_tools([], TEAM) == {}


# ── plan ──────────────────────────────────────────────────────────────────────


def test_parse_plan_reads_fenced_json_and_drops_unknown_agents() -> None:
    reply = 'Here is the plan:\n```json\n{"stages": [[{"agent": "fire", "task": "find it"}, {"agent": "ghost", "task": "x"}]]}\n```'
    assert parse_plan(reply, ["fire"]) == [[{"agent": "fire", "task": "find it"}]]


def test_parse_plan_empty_is_valid_and_garbage_is_not() -> None:
    assert parse_plan('{"stages": []}', ["fire"]) == []
    assert parse_plan("I would call the fire tools.", ["fire"]) is None
    assert parse_plan('{"stages": [[{"agent": "ghost", "task": "x"}]]}', ["fire"]) is None


def test_parse_plan_expands_a_playbook_the_request_can_use() -> None:
    supply = Playbook("supply", "buy decision", [[{"agent": "forest", "task": "screen"}],
                                                  [{"agent": "forest", "task": "decide"}]])
    fire = Playbook("fire", "mudslides", [[{"agent": "fire", "task": "find"}]])
    usable = usable_playbooks([supply, fire], ["forest"])
    assert usable == [supply]
    assert parse_plan('{"playbook": "supply"}', ["forest"], usable) == supply.stages
    assert parse_plan('{"playbook": "fire"}', ["forest"], usable) is None


# ── verifier ──────────────────────────────────────────────────────────────────


def test_unsupported_numbers_accepts_rounding_formats_and_percent() -> None:
    evidence = '{"area_ha": 18271.4, "likelihood": 0.493, "spi": -2.1}'
    evidence += ' {"longitude": -7.737252}'
    answer = "About 18,300 ha (18.271 ha), 49 % likely, SPI −2.1, at 7.7373° W, in 5 farms since 2020."
    assert unsupported_numbers(answer, evidence) == []
    assert unsupported_numbers("Burned area: 25,000 ha.", evidence) == ["25,000"]


def test_clean_links_keeps_real_artifacts_only() -> None:
    evidence = '![map](/artifacts/abc123 "MCP: forest/map")'
    answer = (
        "See ![map](/artifacts/abc123) and ![x](https://bad.example/x.png).\n"
        "[map](/artifacts/abc123) [fake](/artifacts/zzz) [raw](resource://x)"
    )
    cleaned = clean_links(answer, evidence)
    assert "![" not in cleaned
    assert "[map](/artifacts/abc123)" in cleaned
    assert "fake" in cleaned and "/artifacts/zzz" not in cleaned
    assert "resource://" not in cleaned


# ── full run ──────────────────────────────────────────────────────────────────


class ScriptedModel(BaseChatModel):
    """Chat model whose reply is computed from the prompt by ``script``."""

    script: Callable[[List[Any]], AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script(messages))])


def _call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "x"}])


def cocoa_script(messages: List[Any]) -> AIMessage:
    """Plays every role of a two-stage cocoa audit, like use case C."""
    system = str(messages[0].content)
    seen = " ".join(str(m.content) for m in messages[1:])
    if "planner of EVE's EO multi-agent" in system:
        return AIMessage(content=json.dumps({"stages": [
            [{"agent": "forest", "task": "Check farm-01 for deforestation."},
             {"agent": "climate", "task": "Get the drought index of farm-01."}],
            [{"agent": "forest", "task": "Build the decision table for farm-01."}],
        ]}))
    if "You are the forest specialist" in system and "Build the decision table" in system:
        if "[TOOL_RESULTS]" not in seen:
            return _call("forest_compliance_decide", plot="farm-01", drought="high")
        return AIMessage(content="farm-01: blocked.")
    if "You are the forest specialist" in system:
        if "[TOOL_RESULTS]" not in seen:
            return _call("forest_compliance_check", plot="farm-01")
        return AIMessage(content="farm-01: high risk, 12.5 ha cleared after 2020.")
    if "You are the climate specialist" in system:
        if "[TOOL_RESULTS]" not in seen:
            return _call("climate_risk_drought", plot="farm-01")
        return AIMessage(content="farm-01: drought high, SPI -2.1.")
    if "Previous draft" in seen:
        return AIMessage(content="farm-01 is blocked: 12.5 ha cleared, SPI -2.1.")
    return AIMessage(content="farm-01 is blocked: 99.9 ha cleared. ![map](https://bad.example/m.png)")


@tool("forest_compliance_check")
def forest_check(plot: str) -> str:
    """Check one plot against the 2020 forest map."""
    return json.dumps({"plot": plot, "risk": "high", "cleared_ha": 12.5})


@tool("forest_compliance_decide")
def forest_decide(plot: str, drought: str) -> str:
    """Decide the supply status of one plot."""
    return json.dumps({"plot": plot, "status": "blocked", "drought": drought})


@tool("climate_risk_drought")
def drought(plot: str) -> str:
    """Drought index of one plot."""
    return json.dumps({"plot": plot, "level": "high", "spi_3": -2.1})


def _run(script: Callable[[List[Any]], AIMessage], tools: list):
    agent = EOMultiAgent()
    graph = agent.compile(
        llm=ScriptedModel(script=script), tools=tools, checkpointer=InMemorySaver()
    )

    async def go():
        streamed, final = [], None
        async for mode, payload in graph.astream(
            {"messages": [HumanMessage(content="Can I buy from farm-01?")]},
            config={"configurable": {"thread_id": "t"}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, _ = payload
                if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                    streamed.append(chunk.content)
        final = await graph.aget_state({"configurable": {"thread_id": "t"}})
        return streamed, final.values

    return asyncio.run(go())


def test_cocoa_audit_runs_stages_and_fixes_invented_numbers() -> None:
    streamed, state = _run(cocoa_script, [forest_check, forest_decide, drought])

    tools_called = [m.name for m in state["messages"] if isinstance(m, ToolMessage)]
    assert sorted(tools_called[:2]) == ["climate_risk_drought", "forest_compliance_check"]
    assert tools_called[2] == "forest_compliance_decide"
    assert len(state["reports"]) == 3
    assert state["revisions"] == 1

    answer = state["messages"][-1].content
    assert answer == "farm-01 is blocked: 12.5 ha cleared, SPI -2.1."
    # Only the published answer reaches the user: no plan, report or draft.
    assert streamed == [answer]


def test_unusable_plan_falls_back_to_generalist() -> None:
    def script(messages: List[Any]) -> AIMessage:
        system = str(messages[0].content)
        seen = " ".join(str(m.content) for m in messages[1:])
        if "planner of EVE's EO multi-agent" in system:
            return AIMessage(content="Sure, I will ask the forest team.")
        if "You are the generalist specialist" in system:
            if "[TOOL_RESULTS]" not in seen:
                return _call("forest_compliance_check", plot="farm-01")
            return AIMessage(content="farm-01: high risk, 12.5 ha cleared.")
        return AIMessage(content="farm-01 has 12.5 ha cleared after 2020.")

    _, state = _run(script, [forest_check])
    assert list(state["reports"]) == ["Stage 1.1 · generalist"]
    assert state["messages"][-1].content == "farm-01 has 12.5 ha cleared after 2020."


def test_no_tools_answers_directly() -> None:
    _, state = _run(lambda messages: AIMessage(content="Hello, I am EVE."), [])
    assert state["plan"] == []
    assert state["messages"][-1].content == "Hello, I am EVE."
