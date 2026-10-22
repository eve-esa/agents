"""EO multi-agent: a planner, domain specialists and a verifier in one graph.

One ReAct agent with every tool works well with a large model. With a small
one (the Mistral free tier offers only ministral) it loses track in long tool
chains: it stops halfway, rewrites image links and invents values. This graph
splits the work so each model call has one small job::

    planner ─┬─> specialist (forest)  ─┐
             ├─> specialist (climate) ─┼─> advance ─> next stage … ─> reporter
             └─> …    in parallel     ─┘                                 │
                                         publish <─ verifier (rule-based) ┘
                                                    └─> reporter, once, to fix numbers

- **planner** picks a tested playbook or splits the request into stages of
  tasks (``plan.py``).
- **specialist** runs one task as a small ReAct loop with only its own tools
  (``team.py``) and writes a factual report. Tasks of one stage run in parallel.
- **reporter** writes the answer from the reports, without tools.
- **verifier** drops broken image links and sends the draft back once when it
  holds numbers no tool returned (``verify.py``).
- **publish** emits the checked answer as the turn's only text.

Streaming: every internal model call is tagged ``nostream`` and internal text
lives in state keys other than ``messages``, so the user sees the tool calls
and results as they happen and then one answer. Tool calls run through the
same tools the backend passed in, so MCP interceptors (artifacts, latency)
still see every call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import string
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..base import AgentGraph, AgentMessagesState
from ..policies import (
    DEFAULT_LLM_IDLE_TIMEOUT,
    DEFAULT_LLM_RUN_TIMEOUT,
    emit_on_policy,
    is_transient_llm_error,
)
from ..react.graph import ReactAgent
from ..utils import (
    parse_text_tool_calls,
    reformat_messages_for_text_tool_model,
    strip_content_from_tool_call_messages,
)
from .plan import Plan, describe_playbooks, load_playbooks, parse_plan, usable_playbooks
from .team import GENERALIST, GENERALIST_ROLE, assign_tools, describe_team, load_team
from .verify import clean_links, unsupported_numbers

logger = logging.getLogger(__name__)

MAX_SPECIALIST_STEPS = 8
MAX_REVISIONS = 1
LLM_ATTEMPTS = 3
FALLBACK_ANSWER = "Sorry, I could not produce an answer to this request."


def _merge_reports(old: Optional[Dict[str, str]], new: Optional[Dict[str, str]]):
    """Reducer for ``reports``: parallel specialists add keys, ``None`` resets."""
    if new is None:
        return {}
    return {**(old or {}), **new}


class MultiAgentState(AgentMessagesState):
    """Graph state. Only ``messages`` reaches the user; the rest is internal."""

    plan: Plan
    stage: int
    reports: Annotated[Dict[str, str], _merge_reports]
    draft: str
    problems: List[str]
    revisions: int


def _fill(template: str, **values: Any) -> str:
    """``str.format`` for templates that also contain literal JSON braces."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "").strip()


def _clock() -> str:
    now = datetime.now(timezone.utc)
    return (
        "## Clock\n"
        f"Current UTC timestamp: {now.isoformat(timespec='seconds')}\n"
        f"Current UTC date: {now.date().isoformat()}\n"
        "Use this clock for the current date and for relative dates.\n"
    )


def _call_id() -> str:
    """Nine alphanumeric characters, the tool-call id format Mistral accepts."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(9))


def _turn(messages: List[Any]) -> tuple[str, List[Any]]:
    """The current question and the messages added since it was asked."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return _text(messages[index]), list(messages[index + 1 :])
    return "", list(messages)


def _render_reports(reports: Dict[str, str]) -> str:
    return "\n\n".join(f"### {key}\n{text}" for key, text in reports.items())


def _for_model(messages: List[Any]) -> List[Any]:
    """Same message clean-up as the ReAct graph before every model call."""
    messages = strip_content_from_tool_call_messages(messages)
    if any(isinstance(m, ToolMessage) for m in messages):
        messages = reformat_messages_for_text_tool_model(messages)
    return messages


class EOMultiAgent(AgentGraph):
    """Planner, parallel domain specialists, reporter and rule-based verifier.

    Select it in the backend with ``AGENT_GRAPH_TYPE=eo_multi_agent`` (or per request
    with ``agent="eo_multi_agent"``). The team is declared in ``prompts.yaml``; the
    EVE persona for the final answer is the ReAct graph's system prompt.
    """

    name = "eo_multi_agent"

    def __init__(self) -> None:
        super().__init__()
        self.prompts.setdefault("system", ReactAgent().prompts.get("system", ""))
        self.team = load_team(self.prompts.get("team", {}))
        self.playbooks = load_playbooks(self.prompts.get("playbooks", {}))

    # ── model calls ───────────────────────────────────────────────────────────

    async def _ask(
        self,
        node: str,
        messages: List[Any],
        llm: Any,
        fallback_llm: Any,
        *,
        timeout: Optional[float],
        on_policy: Any,
    ) -> AIMessage:
        """One model call with retries on transient errors, then the fallback model.

        Node-level retry policies would re-run a specialist's tool calls, so
        the retries live here, around the model call only.
        """
        config = {"tags": [TAG_NOSTREAM]}
        last_error: Optional[BaseException] = None
        for model, attempts in ((llm, LLM_ATTEMPTS), (fallback_llm, LLM_ATTEMPTS)):
            if model is None:
                continue
            if model is fallback_llm and last_error is not None:
                await emit_on_policy(on_policy, node=node, policy="fallback", error=last_error)
            for attempt in range(1, attempts + 1):
                try:
                    call = model.ainvoke(messages, config=config)
                    return await (asyncio.wait_for(call, timeout) if timeout else call)
                except Exception as exc:  # noqa: BLE001 - retried or re-raised below
                    last_error = exc
                    if attempt == attempts or not (
                        is_transient_llm_error(exc) or isinstance(exc, asyncio.TimeoutError)
                    ):
                        break
                    await emit_on_policy(on_policy, node=node, policy="retry", attempt=attempt + 1)
                    await asyncio.sleep(min(8.0, 1.0 * 2 ** (attempt - 1)))
            logger.warning("Model call in %s failed: %s", node, last_error)
        raise last_error or RuntimeError(f"No model available for {node}")

    # ── graph ─────────────────────────────────────────────────────────────────

    def compile(
        self,
        *,
        llm,
        tools: List[BaseTool],
        checkpointer: Any,
        history: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        fallback_llm=None,
        llm_run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
        llm_idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
        on_policy=None,
        **kwargs,
    ):
        tools = list(tools or [])
        groups = assign_tools(tools, self.team)
        roles = {spec.name: spec.role for spec in self.team}
        roles[GENERALIST] = GENERALIST_ROLE
        planned = [name for name in groups if name != GENERALIST]
        playbooks = usable_playbooks(self.playbooks, planned)
        instruction = self.instruction_text(history=history, summary=summary) or ""
        history_text = self.format_history(history or [], summary) or ""
        tools_nodes = {name: self.make_tools_node(group) for name, group in groups.items()}
        ask = dict(timeout=llm_run_timeout, on_policy=on_policy)

        def bound(model: Any, name: str) -> Any:
            return model.bind_tools(groups[name]) if model is not None else None

        async def planner(state: MultiAgentState):
            reset = {"stage": 0, "reports": None, "draft": "", "problems": [], "revisions": 0}
            question, _ = _turn(state["messages"])
            if not planned:
                return {**reset, "plan": []}
            system = _clock()
            if history_text:
                system += "\n" + history_text + "\n"
            system += "\n" + _fill(
                self.prompts["planner"],
                team=describe_team(self.team, groups),
                playbooks=describe_playbooks(playbooks),
            )
            try:
                reply = await self._ask(
                    "planner", [SystemMessage(content=system), HumanMessage(content=question)],
                    llm, fallback_llm, **ask,
                )
                plan = parse_plan(_text(reply), planned, playbooks)
            except Exception as exc:  # noqa: BLE001 - the generalist takes over
                logger.warning("Planner failed: %s", exc)
                plan = None
            if plan is None:
                logger.info("No usable plan; the generalist runs the request")
                plan = [[{"agent": GENERALIST, "task": "Answer the user's request in full."}]]
            logger.info("Plan: %s", plan)
            return {**reset, "plan": plan}

        def dispatch(state: MultiAgentState):
            plan, stage = state.get("plan") or [], state.get("stage", 0)
            if stage >= len(plan):
                return "reporter"
            question, _ = _turn(state["messages"])
            reports = state.get("reports") or {}
            return [
                Send("specialist", {
                    "agent": task["agent"],
                    "task": task["task"],
                    "key": f"Stage {stage + 1}.{index + 1} · {task['agent']}",
                    "question": question,
                    "reports": dict(reports),
                })
                for index, task in enumerate(plan[stage])
            ]

        async def specialist(job: Dict[str, Any]):
            name = job["agent"]
            context = (
                _fill(self.prompts["context"], reports=_render_reports(job["reports"]))
                if job["reports"] else ""
            )
            system = _clock() + "\n"
            if history_text:
                system += history_text + "\n\n"
            system += _fill(
                self.prompts["specialist"], name=name, role=roles.get(name, ""),
                task=job["task"], context=context,
            )
            # The task first, the user's message as data: given the whole request, a specialist that holds
            # only some of the tools may refuse it instead of doing its part (seen with EVE-Instruct).
            request = _fill(self.prompts["specialist_request"], task=job["task"], question=job["question"])
            private: List[Any] = [SystemMessage(content=system), HumanMessage(content=request)]
            public: List[Any] = []
            primary, fallback = bound(llm, name), bound(fallback_llm, name)
            report = ""
            try:
                for _ in range(MAX_SPECIALIST_STEPS):
                    reply = await self._ask(f"specialist:{name}", _for_model(private), primary, fallback, **ask)
                    calls = list(getattr(reply, "tool_calls", None) or []) or parse_text_tool_calls(_text(reply))
                    if not calls:
                        report = _text(reply)
                        break
                    call = AIMessage(content="", tool_calls=[{**c, "id": _call_id()} for c in calls])
                    results = (await tools_nodes[name]({"messages": [call]}))["messages"]
                    private += [call, *results]
                    public += [call, *results]
                else:
                    private.append(HumanMessage(content="Stop calling tools. Write your report now."))
                    reply = await self._ask(f"specialist:{name}", _for_model(private), llm, fallback_llm, **ask)
                    report = _text(reply)
            except Exception as exc:  # noqa: BLE001 - one failed specialist must not sink the turn
                logger.warning("Specialist %s failed: %s", name, exc)
                report = f"(The {name} specialist failed: {exc})"
            return {"messages": public, "reports": {job["key"]: report or "(no report)"}}

        def advance(state: MultiAgentState):
            return {"stage": state.get("stage", 0) + 1}

        async def reporter(state: MultiAgentState):
            question, _ = _turn(state["messages"])
            reports = state.get("reports") or {}
            system = _clock() + "\n" + instruction + "\n\n"
            if reports:
                system += self.prompts["reporter"]
                user = f"{question}\n\n## Specialist reports\n{_render_reports(reports)}"
            else:
                system += self.prompts["reporter_direct"]
                user = question
            if state.get("problems"):
                user += "\n\n" + _fill(
                    self.prompts["revision"],
                    numbers=", ".join(state["problems"]), draft=state.get("draft", ""),
                )
            try:
                reply = await self._ask(
                    "reporter", [SystemMessage(content=system), HumanMessage(content=user)],
                    llm, fallback_llm, **ask,
                )
                draft = _text(reply)
            except Exception as exc:  # noqa: BLE001 - keep the last draft if there is one
                logger.warning("Reporter failed: %s", exc)
                draft = state.get("draft", "")
            return {"draft": draft, "problems": []}

        def verifier(state: MultiAgentState):
            question, turn = _turn(state["messages"])
            reports = state.get("reports") or {}
            evidence = "\n".join(
                [question, *reports.values()]
                + [_text(m) for m in turn if isinstance(m, ToolMessage)]
                + [json.dumps(c.get("args", {})) for m in turn for c in getattr(m, "tool_calls", None) or []]
            )
            draft = clean_links(state.get("draft", ""), evidence)
            problems = unsupported_numbers(draft, evidence) if reports else []
            revisions = state.get("revisions", 0)
            if problems and revisions < MAX_REVISIONS:
                logger.info("Verifier sends the draft back for numbers %s", problems)
                return {"draft": draft, "problems": problems, "revisions": revisions + 1}
            if problems:
                logger.warning("Publishing with unsupported numbers %s", problems)
            return {"draft": draft, "problems": []}

        def after_verifier(state: MultiAgentState):
            return "reporter" if state.get("problems") else "publish"

        def publish(state: MultiAgentState):
            return {"messages": [AIMessage(content=state.get("draft") or FALLBACK_ANSWER)]}

        builder = StateGraph(MultiAgentState)
        for node_name, fn in (
            ("planner", planner), ("specialist", specialist), ("advance", advance),
            ("reporter", reporter), ("verifier", verifier), ("publish", publish),
        ):
            builder.add_node(node_name, self.timed_node(node_name, fn, on_policy=on_policy))
        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", dispatch, ["specialist", "reporter"])
        builder.add_edge("specialist", "advance")
        builder.add_conditional_edges("advance", dispatch, ["specialist", "reporter"])
        builder.add_edge("reporter", "verifier")
        builder.add_conditional_edges("verifier", after_verifier, ["reporter", "publish"])
        builder.add_edge("publish", END)
        return builder.compile(checkpointer=checkpointer)
