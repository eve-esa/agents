"""Shared utilities for agent graphs — no backend dependencies.

Contains the clock prompt section, text-format tool-call parsing
(Mistral/EVE-Instruct), message reformatting and token counting.

Only depends on standard library + langchain-core.
Safe to import from standalone scripts/notebooks.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_TEXT_TOOL_CALL_MARKER = "[TOOL_CALLS]"


# ─── Clock ────────────────────────────────────────────────────────────────────


def clock_context(now: Optional[datetime] = None) -> str:
    """Return the ``## Clock`` system-prompt section for the current UTC instant.

    Call it on every model invocation, not at compile time, so cached graphs
    and resumed threads do not keep the date on which they were compiled.
    """
    now = now or datetime.now(timezone.utc)
    return (
        "## Clock\n"
        f"Current UTC timestamp: {now.isoformat(timespec='seconds')}\n"
        f"Current UTC date: {now.date().isoformat()}\n"
        f"Current UTC year: {now.year}\n"
        "Use this clock for the current date/year and relative dates, "
        "not the knowledge cutoff or dates in conversation history. "
        "For local requests, interpret this instant in the requested "
        "location's timezone.\n"
    )


# ─── Text-format tool-call parsing (Mistral / EVE-Instruct) ───────────────────


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse ``[TOOL_CALLS]tool_name{"key": "val"} ...`` into structured dicts.

    Returns a list compatible with ``AIMessage.tool_calls``.
    """
    if _TEXT_TOOL_CALL_MARKER not in content:
        return []

    text = content[
        content.index(_TEXT_TOOL_CALL_MARKER) + len(_TEXT_TOOL_CALL_MARKER) :
    ].strip()
    calls: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\n,":
            i += 1
        if i >= len(text):
            break
        m = re.match(r"([A-Za-z_]\w*)", text[i:])
        if not m:
            break
        name = m.group(1)
        i += m.end()
        while i < len(text) and text[i] in " \t":
            i += 1
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            args = json.loads(text[i : j + 1])
                        except Exception:
                            args = {}
                        calls.append(
                            {
                                "id": f"call_{len(calls)}_{name}",
                                "name": name,
                                "args": args,
                                "type": "tool_call",
                            }
                        )
                        i = j + 1
                        break
                j += 1
            else:
                break
        else:
            calls.append(
                {
                    "id": f"call_{len(calls)}_{name}",
                    "name": name,
                    "args": {},
                    "type": "tool_call",
                }
            )
    return calls


# ─── Message history sanitisation ─────────────────────────────────────────────


def reformat_messages_for_text_tool_model(
    messages: List[Any],
    *,
    AIMessage: Any = None,
    ToolMessage: Any = None,
    HumanMessage: Any = None,
) -> List[Any]:
    """Convert structured tool_calls/ToolMessages back to plain-text format.

    Pass LangChain message classes explicitly if calling from a context where
    they may not be installed, or leave as None to auto-import.
    """
    if AIMessage is None or ToolMessage is None or HumanMessage is None:
        from langchain_core.messages import (
            AIMessage as _AI,
            HumanMessage as _H,
            ToolMessage as _T,
        )

        AIMessage = AIMessage or _AI
        ToolMessage = ToolMessage or _T
        HumanMessage = HumanMessage or _H

    result: List[Any] = []
    for msg in messages:
        if (
            isinstance(msg, AIMessage)
            and not msg.content
            and getattr(msg, "tool_calls", None)
        ):
            calls = [
                {"name": tc["name"], "arguments": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
            result.append(AIMessage(content=f"[TOOL_CALLS] {json.dumps(calls)}"))
        elif isinstance(msg, ToolMessage):
            result.append(HumanMessage(content=f"[TOOL_RESULTS]\n{msg.content}"))
        else:
            result.append(msg)
    return result


def strip_content_from_tool_call_messages(
    messages: List[Any],
    *,
    AIMessage: Any = None,
) -> List[Any]:
    """Strip text content from AIMessages that also carry tool_calls.

    Some APIs (Mistral) reject assistant messages with both non-empty
    content AND tool_calls.
    """
    if AIMessage is None:
        from langchain_core.messages import AIMessage as _AI

        AIMessage = _AI

    return [
        AIMessage(content="", tool_calls=m.tool_calls, id=m.id)
        if (isinstance(m, AIMessage) and m.content and getattr(m, "tool_calls", None))
        else m
        for m in messages
    ]


# ─── Token counting ───────────────────────────────────────────────────────────


def tiktoken_counter(messages: List[Any]) -> int:
    """Approximate token count using tiktoken cl100k_base."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            total += len(enc.encode(str(content))) + 4
        return total
    except Exception:
        return sum(len(str(getattr(m, "content", m))) // 4 for m in messages)
