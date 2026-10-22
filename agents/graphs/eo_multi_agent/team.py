"""The EO multi-agent team: which specialist owns which tools.

MCP tools reach the graph named ``<server>_<tool>``. A specialist owns every
tool whose name starts with one of its servers; the longest matching server
wins, so ``wildfire_flood`` is not mistaken for a server called ``wildfire``.
Tools no server claims go to the specialist whose servers include ``"*"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

CATCH_ALL = "*"

# Built in, never offered to the planner: owns every tool and runs the whole
# request as one task when no usable plan comes back.
GENERALIST = "generalist"
GENERALIST_ROLE = (
    "Generalist: answers the request with every tool available, step by step."
)


@dataclass(frozen=True)
class Specialist:
    """One member of the team, as declared under ``team`` in ``prompts.yaml``."""

    name: str
    role: str
    servers: Tuple[str, ...]


def load_team(config: Mapping[str, Any]) -> List[Specialist]:
    """Build the specialists from the ``team`` mapping of ``prompts.yaml``."""
    team = []
    for name, spec in (config or {}).items():
        servers = tuple(str(s) for s in (spec or {}).get("servers", []))
        role = str((spec or {}).get("role", "")).strip()
        if servers and name != GENERALIST:
            team.append(Specialist(name=str(name), role=role, servers=servers))
    return team


def tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def owner(name: str, team: List[Specialist]) -> Optional[str]:
    """Return the specialist owning tool *name*, or ``None`` if nobody does."""
    best, best_len = None, -1
    for spec in team:
        for server in spec.servers:
            if server != CATCH_ALL and name.startswith(server + "_") and len(server) > best_len:
                best, best_len = spec.name, len(server)
    if best is None:
        best = next((s.name for s in team if CATCH_ALL in s.servers), None)
    return best


def assign_tools(tools: List[Any], team: List[Specialist]) -> Dict[str, List[Any]]:
    """Group *tools* by owning specialist; specialists without tools are left out.

    The generalist always gets every tool, so a request still has a way through
    when the planner fails.
    """
    groups: Dict[str, List[Any]] = {}
    for tool in tools:
        name = owner(tool_name(tool), team)
        if name is not None:
            groups.setdefault(name, []).append(tool)
    if tools:
        groups[GENERALIST] = list(tools)
    return groups


def short_tool_name(name: str, servers: Tuple[str, ...]) -> str:
    """``forest_compliance_eudr_check_plots`` -> ``eudr_check_plots``."""
    for server in sorted(servers, key=len, reverse=True):
        if server != CATCH_ALL and name.startswith(server + "_"):
            return name[len(server) + 1 :]
    return name


def first_sentence(text: str, limit: int = 120) -> str:
    """First sentence of a tool description, cut to *limit* characters."""
    text = " ".join(str(text or "").split())
    end = text.find(". ")
    sentence = text if end < 0 else text[: end + 1]
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"


def describe_team(team: List[Specialist], groups: Dict[str, List[Any]]) -> str:
    """Team roster for the planner: each specialist with its role and tools."""
    lines = []
    for spec in team:
        tools = groups.get(spec.name)
        if not tools:
            continue
        lines.append(f"- {spec.name}: {spec.role}")
        for tool in tools:
            short = short_tool_name(tool_name(tool), spec.servers)
            summary = first_sentence(getattr(tool, "description", ""))
            lines.append(f"    - {short}: {summary}" if summary else f"    - {short}")
    return "\n".join(lines)
