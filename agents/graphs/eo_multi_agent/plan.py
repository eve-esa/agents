"""Parse the planner's reply into stages of specialist tasks.

The planner answers with JSON, either naming a playbook::

    {"playbook": "supply_check"}

or writing its own stages::

    {"stages": [[{"agent": "forest", "task": "..."},
                 {"agent": "climate", "task": "..."}],
                [{"agent": "forest", "task": "..."}]]}

Tasks in one stage run in parallel; a stage starts when the one before it has
finished. Playbooks are tested recipes for known requests, declared in
``prompts.yaml``: choosing one is a much easier job for a small model than
composing the stages, and the recipe gets the order right every time. Small
models wrap the JSON in prose or code fences, so the parser looks for the
outermost object instead of requiring a clean reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

MAX_STAGES = 3
MAX_TASKS_PER_STAGE = 3

Plan = List[List[Dict[str, str]]]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Playbook:
    """A tested plan for a known kind of request."""

    name: str
    when: str
    stages: Plan

    def agents(self) -> set:
        return {task["agent"] for stage in self.stages for task in stage}


def load_playbooks(config: Mapping[str, Any]) -> List[Playbook]:
    """Build the playbooks from the ``playbooks`` mapping of ``prompts.yaml``."""
    playbooks = []
    for name, spec in (config or {}).items():
        spec = spec or {}
        stages = [
            [{"agent": str(t["agent"]), "task": " ".join(str(t["task"]).split())} for t in stage]
            for stage in spec.get("stages", [])
        ]
        if stages:
            playbooks.append(Playbook(str(name), " ".join(str(spec.get("when", "")).split()), stages))
    return playbooks


def usable_playbooks(playbooks: List[Playbook], agents: Iterable[str]) -> List[Playbook]:
    """The playbooks whose specialists all have tools in this request."""
    available = set(agents)
    return [p for p in playbooks if p.agents() <= available]


def describe_playbooks(playbooks: List[Playbook]) -> str:
    return "\n".join(f"- {p.name}: {p.when}" for p in playbooks) or "(none for this request)"


def _json_object(text: str) -> Optional[Any]:
    """Return the first JSON object in *text*, or ``None``."""
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


def _tasks(stage: Any, agents: Iterable[str]) -> List[Dict[str, str]]:
    """Keep the well-formed tasks of one stage that name a known specialist."""
    known = set(agents)
    if isinstance(stage, dict):
        stage = [stage]
    tasks = []
    for item in stage if isinstance(stage, list) else []:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        task = str(item.get("task", "")).strip()
        if agent in known and task:
            tasks.append({"agent": agent, "task": task})
    return tasks[:MAX_TASKS_PER_STAGE]


def parse_plan(
    text: str, agents: Iterable[str], playbooks: Iterable[Playbook] = ()
) -> Optional[Plan]:
    """Turn the planner's reply into stages; ``None`` when it is unusable.

    A named playbook wins over any stages in the same reply. An empty list is
    a valid plan: the request needs no tool. A reply that lists stages but none
    with a usable task is unusable, because the planner clearly wanted tools
    and named something that does not exist.
    """
    data = _json_object(str(text or ""))
    if not isinstance(data, dict):
        return None
    chosen = str(data.get("playbook") or "").strip()
    for playbook in playbooks:
        if playbook.name == chosen:
            return [list(stage) for stage in playbook.stages]
    if not isinstance(data.get("stages"), list):
        return None
    agents = list(agents)
    stages = [tasks for tasks in (_tasks(s, agents) for s in data["stages"]) if tasks]
    if data["stages"] and not stages:
        return None
    return stages[:MAX_STAGES]
