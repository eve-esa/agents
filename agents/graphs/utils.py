"""Shared utilities for agent graphs, no backend dependencies.

Contains the clock prompt section, text-format tool-call parsing
(Mistral/EVE-Instruct), tool call id normalisation, message clean-up and
token counting.

Only depends on standard library + langchain-core.
Safe to import from standalone scripts/notebooks.
"""

import hashlib
import json
import re
import string
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

_ARGS_MARKER = "[ARGS]"
_TOOL_NAME = re.compile(r"[A-Za-z_][\w-]*")


def _call_args(item: Dict[str, Any]) -> Dict[str, Any]:
    """Arguments under ``arguments`` or ``args``; a string is one more JSON object."""
    args = item.get("arguments", item.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def _relax_json_escapes(text: str) -> tuple[str, List[int]]:
    """Keep valid JSON escapes and double the rest.

    A shell command such as ``\\(`` is not a JSON escape, and ``json.loads``
    would reject the whole call. Also returns where a backslash was inserted.
    """
    inserted: List[int] = []

    def _escape(match: "re.Match[str]") -> str:
        if match.group(1) in '"\\/bfnrtu':
            return match.group(0)
        inserted.append(match.start())
        return "\\\\" + match.group(1)

    return re.sub(r"\\(.)", _escape, text), inserted


def _decode_json_value(text: str) -> tuple[Any, int]:
    """Decode the JSON value at the start of *text*; return it and its end in *text*."""
    try:
        return json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        repaired, inserted = _relax_json_escapes(text)
        if not inserted:
            raise
        value, end = json.JSONDecoder().raw_decode(repaired)
        shift = sum(1 for k, pos in enumerate(inserted) if pos + k < end)
        return value, end - shift


def _tool_call(index: int, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": f"call_{index}_{name}", "name": name, "args": args, "type": "tool_call"}


def _json_array_calls(text: str) -> Optional[tuple[List[Dict[str, str]], int]]:
    """``[{"name": ..., "arguments": {...}}, ...]`` at the start of *text*.

    Mistral tokenizer V2/V3 form. Returns the calls and the end offset, or
    ``None`` when *text* does not start with such an array.
    """
    if not text.startswith("["):
        return None
    try:
        payload, end = _decode_json_value(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    calls = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        if not isinstance(name, str) or not name:
            return None
        calls.append({"name": name, "args": _call_args(item)})
    return calls, end


def _named_call(text: str) -> Optional[tuple[Dict[str, Any], int]]:
    """``name{...}`` or ``name[ARGS]{...}`` at the start of *text*.

    The arguments object is required, so a word after the marker in prose is
    not taken for a call.
    """
    match = _TOOL_NAME.match(text)
    if not match:
        return None
    i = match.end()
    while i < len(text) and text[i] in " \t":
        i += 1
    if text.startswith(_ARGS_MARKER, i):
        i += len(_ARGS_MARKER)
        while i < len(text) and text[i] in " \t":
            i += 1
    if not text.startswith("{", i):
        return None
    try:
        args, length = _decode_json_value(text[i:])
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return {"name": match.group(0), "args": args}, i + length


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse ``[TOOL_CALLS]`` text into structured tool calls.

    Fallback for a model reply that carries its tool calls as text instead of
    structured ``tool_calls``. Reads the three Mistral forms, several calls
    each:

    - ``[TOOL_CALLS]name{...}``: the Small 3.2 form with ``[ARGS]`` dropped on
      decode;
    - ``[TOOL_CALLS]name[ARGS]{...}``: the Small 3.2 form, one marker per call;
    - ``[TOOL_CALLS] [{"name": ..., "arguments": {...}}]``: tokenizer V2/V3.

    Parsing stops at the first text that is none of these. Returns ``[]`` when
    no call parses, so prose that mentions the marker is left alone. Ids are
    ``call_<n>_<name>``, counted per reply. Returns a list compatible with
    ``AIMessage.tool_calls``.
    """
    if _TEXT_TOOL_CALL_MARKER not in content:
        return []

    text = content[content.index(_TEXT_TOOL_CALL_MARKER) :]
    found: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\n,":
            i += 1
        if text.startswith(_TEXT_TOOL_CALL_MARKER, i):
            i += len(_TEXT_TOOL_CALL_MARKER)
            while i < len(text) and text[i] in " \t\n":
                i += 1
        array = _json_array_calls(text[i:])
        if array is not None:
            found.extend(array[0])
            i += array[1]
            continue
        named = _named_call(text[i:])
        if named is None:
            break
        found.append(named[0])
        i += named[1]
    return [_tool_call(n, call["name"], call["args"]) for n, call in enumerate(found)]


# ─── Message history sanitisation ─────────────────────────────────────────────

# JSC serves EVE-Instruct on vLLM with the Mistral tokenizer: it keeps the last
# 9 characters of a longer tool call id and answers 400 "Tool call id was ...
# but must be a-z, A-Z, 0-9, with a length of 9" for anything else.
_PROVIDER_TOOL_CALL_ID = re.compile(r"[A-Za-z0-9]{9}")
_ID_ALPHABET = string.ascii_letters + string.digits


def _provider_tool_call_id(seed: str) -> str:
    number = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    chars = []
    for _ in range(9):
        number, rest = divmod(number, len(_ID_ALPHABET))
        chars.append(_ID_ALPHABET[rest])
    return "".join(chars)


def normalize_tool_call_ids(
    messages: List[Any],
    *,
    AIMessage: Any = None,
    ToolMessage: Any = None,
) -> List[Any]:
    """Give every tool call id the ``[A-Za-z0-9]{9}`` shape all providers accept.

    Ids from a provider reply already have it and are kept. Ids minted by
    :func:`parse_text_tool_calls` (``call_<n>_<name>``) do not, and repeat
    across replies, so each occurrence gets its own id derived from the
    original and its position; a tool result takes the id of the call before
    it. Returns new messages for the model request; *messages* is unchanged.
    """
    if AIMessage is None or ToolMessage is None:
        from langchain_core.messages import AIMessage as _AI, ToolMessage as _T

        AIMessage = AIMessage or _AI
        ToolMessage = ToolMessage or _T

    seen: Dict[str, int] = {}
    current: Dict[str, str] = {}

    def _renamed(call_id: Optional[str]) -> Optional[str]:
        if not call_id or _PROVIDER_TOOL_CALL_ID.fullmatch(call_id):
            return call_id
        occurrence = seen.get(call_id, 0)
        seen[call_id] = occurrence + 1
        current[call_id] = _provider_tool_call_id(f"{call_id}#{occurrence}")
        return current[call_id]

    result: List[Any] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and (msg.tool_calls or msg.invalid_tool_calls):
            tool_calls = [{**tc, "id": _renamed(tc.get("id"))} for tc in msg.tool_calls]
            invalid = [
                {**tc, "id": _renamed(tc.get("id"))} for tc in msg.invalid_tool_calls
            ]
            if tool_calls != msg.tool_calls or invalid != msg.invalid_tool_calls:
                msg = msg.model_copy(
                    update={"tool_calls": tool_calls, "invalid_tool_calls": invalid}
                )
        elif isinstance(msg, ToolMessage):
            call_id = msg.tool_call_id
            if call_id and not _PROVIDER_TOOL_CALL_ID.fullmatch(call_id):
                new_id = current.get(call_id) or _provider_tool_call_id(f"{call_id}#0")
                msg = msg.model_copy(update={"tool_call_id": new_id})
        result.append(msg)
    return result


def answer_dangling_tool_calls(
    messages: List[Any],
    *,
    AIMessage: Any = None,
    ToolMessage: Any = None,
) -> List[Any]:
    """Add an error tool result for every call that has none.

    A Stop or a failure during a tool leaves a call without a result in the
    thread. Mistral answers 400 "Not the same number of function calls and
    responses" for such history. The result goes after the results the call
    did get. Returns new messages for the model request; *messages* is
    unchanged.
    """
    if AIMessage is None or ToolMessage is None:
        from langchain_core.messages import AIMessage as _AI, ToolMessage as _T

        AIMessage = AIMessage or _AI
        ToolMessage = ToolMessage or _T

    result: List[Any] = []
    pending: List[Dict[str, Any]] = []

    def _close() -> None:
        for call in pending:
            name = call.get("name") or "unknown"
            result.append(
                ToolMessage(
                    content=f"Tool call {name} did not complete: no result was recorded.",
                    tool_call_id=call["id"],
                    name=name,
                    status="error",
                )
            )
        pending.clear()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            pending[:] = [c for c in pending if c["id"] != msg.tool_call_id]
            result.append(msg)
            continue
        _close()
        result.append(msg)
        if isinstance(msg, AIMessage):
            pending.extend(
                c for c in (*msg.tool_calls, *msg.invalid_tool_calls) if c.get("id")
            )
    _close()
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
