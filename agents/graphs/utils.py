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
from typing import Any, Dict, Iterable, List, Optional

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


def parse_text_tool_calls(
    content: str, allowed_names: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Parse ``[TOOL_CALLS]`` text into structured tool calls.

    Public and stable: the backend stream flush reuses it. Fallback for a
    model reply that carries its tool calls as text instead of structured
    ``tool_calls``. Accepted forms, several calls each, text before the first
    marker ignored:

    - ``[TOOL_CALLS]name{...}``: the Small 3.2 form with ``[ARGS]`` dropped on
      decode;
    - ``[TOOL_CALLS]name[ARGS]{...}``: the Small 3.2 form, one marker per call;
    - ``[TOOL_CALLS] [{"name": ..., "arguments": {...}}]``: tokenizer V2/V3
      (``args`` for ``arguments``, or arguments as a JSON string, also read).

    Arguments are one JSON object; a backslash that is not a JSON escape (a
    shell ``\\(``) is kept as a literal backslash. A bare name without an
    object is not a call. Parsing stops at the first text that is none of
    these. With *allowed_names*, a call to any other name is dropped, so
    prose such as ``the [TOOL_CALLS] token {...}`` is not a call. Returns
    ``[]`` when no call parses. Ids are ``call_<n>_<name>``, counted per
    reply. The list is compatible with ``AIMessage.tool_calls``. Test vectors:
    :data:`agents.graphs.text_tool_call_vectors.TEXT_TOOL_CALL_VECTORS`.
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
    if allowed_names is not None:
        allowed = set(allowed_names)
        found = [call for call in found if call["name"] in allowed]
    return [_tool_call(n, call["name"], call["args"]) for n, call in enumerate(found)]


def repair_tool_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    """Arguments of a structured call as a dict, or ``None`` when malformed.

    Applies the escape repair of :func:`parse_text_tool_calls`, so a shell
    ``\\(`` that made the provider's JSON invalid is read back. Empty
    arguments are ``{}``.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return {}
    try:
        value, end = _decode_json_value(text)
    except json.JSONDecodeError:
        return None
    if text[end:].strip() or not isinstance(value, dict):
        return None
    return value


def tool_names(tools: Iterable[Any]) -> set:
    """Names of bound tools: ``BaseTool`` objects or OpenAI tool dicts."""
    names = set()
    for item in tools or ():
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name") or (item.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


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


MALFORMED_TOOL_CALL_RETRIES = 2
"""Extra model calls when a reply holds only calls with malformed arguments."""

MALFORMED_TOOL_CALL_ANSWER = (
    "I could not run the tool: the arguments of my tool call were malformed "
    "several times. Please try again or rephrase the request."
)


def _malformed_text(name: str) -> str:
    return (
        f"Tool call {name} was not run: its arguments were malformed (not valid "
        "JSON). Call the tool again with the arguments as one valid JSON object."
    )


def malformed_tool_call_results(
    calls: List[Dict[str, Any]], *, ToolMessage: Any = None
) -> List[Any]:
    """One error tool result per invalid call, telling the model why."""
    if ToolMessage is None:
        from langchain_core.messages import ToolMessage

    results = []
    for call in calls:
        name = call.get("name") or "unknown"
        results.append(
            ToolMessage(
                content=_malformed_text(name),
                tool_call_id=call.get("id") or "",
                name=name,
                status="error",
            )
        )
    return results


def malformed_reply_errors(message: Any) -> Optional[List[Any]]:
    """Error results when *message* has malformed calls and no runnable one.

    ``None`` otherwise. The caller appends the reply and these results and
    calls the model again, up to :data:`MALFORMED_TOOL_CALL_RETRIES` times.
    """
    if getattr(message, "tool_calls", None) or not getattr(
        message, "invalid_tool_calls", None
    ):
        return None
    return malformed_tool_call_results(message.invalid_tool_calls)


def structure_model_reply(
    message: Any,
    allowed_names: Optional[Iterable[str]] = None,
    *,
    AIMessage: Any = None,
) -> Any:
    """Give a model reply structured ``tool_calls`` wherever they can be read.

    An invalid structured call whose arguments the escape repair reads is
    promoted to ``tool_calls`` with its id. A reply with no call at all is
    parsed for ``[TOOL_CALLS]`` text (:func:`parse_text_tool_calls`, limited
    to *allowed_names*); its text is then cleared. Metadata is kept. Calls
    that stay malformed stay in ``invalid_tool_calls``.
    """
    if AIMessage is None:
        from langchain_core.messages import AIMessage

    if not isinstance(message, AIMessage):
        return message
    promoted: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for call in message.invalid_tool_calls:
        name = call.get("name")
        args = repair_tool_arguments(call.get("args"))
        if isinstance(name, str) and _TOOL_NAME.fullmatch(name) and args is not None:
            call_id = call.get("id") or _provider_tool_call_id(
                f"{name}#{call.get('args')}"
            )
            promoted.append(
                {"id": call_id, "name": name, "args": args, "type": "tool_call"}
            )
        else:
            remaining.append(call)
    if promoted:
        return message.model_copy(
            update={
                "tool_calls": [*message.tool_calls, *promoted],
                "invalid_tool_calls": remaining,
            }
        )
    if (
        not message.tool_calls
        and not message.invalid_tool_calls
        and isinstance(message.content, str)
    ):
        parsed = parse_text_tool_calls(message.content, allowed_names=allowed_names)
        if parsed:
            return message.model_copy(update={"tool_calls": parsed, "content": ""})
    return message


def answer_dangling_tool_calls(
    messages: List[Any],
    *,
    AIMessage: Any = None,
    ToolMessage: Any = None,
) -> List[Any]:
    """Pair every call with exactly one tool result.

    A Stop or a failure during a tool leaves a call without a result in the
    thread. Mistral answers 400 "Not the same number of function calls and
    responses" for such history. The added error result goes after the
    results the call did get; an invalid call is told its arguments were
    malformed. A tool result that answers no pending call (an orphan, a
    duplicate or a late one) is dropped. Returns new messages for the model
    request; *messages* is unchanged.
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
            content = (
                _malformed_text(name)
                if call.get("type") == "invalid_tool_call"
                else f"Tool call {name} did not complete: no result was recorded."
            )
            result.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=name,
                    status="error",
                )
            )
        pending.clear()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            if any(c["id"] == msg.tool_call_id for c in pending):
                pending[:] = [c for c in pending if c["id"] != msg.tool_call_id]
                result.append(msg)
            continue
        _close()
        result.append(msg)
        if isinstance(msg, AIMessage):
            pending.extend(
                {**c, "type": "tool_call"} for c in msg.tool_calls if c.get("id")
            )
            pending.extend(
                {**c, "type": "invalid_tool_call"}
                for c in msg.invalid_tool_calls
                if c.get("id")
            )
    _close()
    return result


def structure_text_tool_history(
    messages: List[Any],
    allowed_names: Optional[Iterable[str]] = None,
    *,
    AIMessage: Any = None,
) -> List[Any]:
    """Turn old ``[TOOL_CALLS]`` assistant text into structured calls.

    Threads from before structured history hold replies whose text is a call
    the graph did not parse. Sent as text, they teach the model that form.
    A reply that parses becomes structured calls (the results that follow
    pair by id, a missing one gets an error result later); a reply that
    starts with the marker and does not parse is dropped. Prose that only
    mentions the marker is kept. Returns new messages; *messages* is
    unchanged.
    """
    if AIMessage is None:
        from langchain_core.messages import AIMessage

    result: List[Any] = []
    for msg in messages:
        if (
            isinstance(msg, AIMessage)
            and not msg.tool_calls
            and not msg.invalid_tool_calls
            and isinstance(msg.content, str)
            and _TEXT_TOOL_CALL_MARKER in msg.content
        ):
            parsed = parse_text_tool_calls(msg.content, allowed_names=allowed_names)
            if parsed:
                msg = msg.model_copy(update={"tool_calls": parsed, "content": ""})
            elif msg.content.lstrip().startswith(_TEXT_TOOL_CALL_MARKER):
                continue
        result.append(msg)
    return result


def structure_invalid_tool_calls(
    messages: List[Any],
    *,
    AIMessage: Any = None,
    ToolMessage: Any = None,
) -> List[Any]:
    """Send invalid calls in history as structured calls the provider accepts.

    ``langchain_openai`` would send an invalid call with its raw, non-JSON
    arguments string. On the request copy its arguments become the repaired
    object, or ``{}``; the error result that answers it says why. A call
    whose name is not a tool name is dropped with its results, and so is an
    assistant message left with neither text nor calls.
    """
    if AIMessage is None or ToolMessage is None:
        from langchain_core.messages import AIMessage as _AI, ToolMessage as _T

        AIMessage = AIMessage or _AI
        ToolMessage = ToolMessage or _T

    dropped: set = set()
    result: List[Any] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in dropped:
            continue
        if isinstance(msg, AIMessage) and msg.invalid_tool_calls:
            calls = list(msg.tool_calls)
            for call in msg.invalid_tool_calls:
                name = call.get("name")
                if not (isinstance(name, str) and _TOOL_NAME.fullmatch(name)) or not call.get("id"):
                    dropped.add(call.get("id"))
                    continue
                args = repair_tool_arguments(call.get("args"))
                calls.append(
                    {"id": call["id"], "name": name, "args": args or {}, "type": "tool_call"}
                )
            if not calls and not msg.content:
                continue
            msg = msg.model_copy(update={"tool_calls": calls, "invalid_tool_calls": []})
        result.append(msg)
    return result


def prepare_model_messages(
    messages: List[Any], allowed_names: Optional[Iterable[str]] = None
) -> List[Any]:
    """The request copy of a thread, in the order each step needs.

    Old text calls become structured, every call gets exactly one result,
    invalid calls go out as structured calls, text next to calls is cleared
    and ids get the shape JSC accepts. *messages* is unchanged.
    """
    messages = structure_text_tool_history(messages, allowed_names)
    messages = answer_dangling_tool_calls(messages)
    messages = structure_invalid_tool_calls(messages)
    messages = strip_content_from_tool_call_messages(messages)
    return normalize_tool_call_ids(messages)


def strip_content_from_tool_call_messages(
    messages: List[Any],
    *,
    AIMessage: Any = None,
) -> List[Any]:
    """Strip text content from AIMessages that also carry tool calls.

    Some APIs (Mistral) reject assistant messages with both non-empty
    content AND tool_calls. Invalid calls count: ``langchain_openai`` sends
    them as ``tool_calls`` too. Returns copies; *messages* is unchanged.
    """
    if AIMessage is None:
        from langchain_core.messages import AIMessage as _AI

        AIMessage = _AI

    return [
        m.model_copy(update={"content": ""})
        if (
            isinstance(m, AIMessage)
            and m.content
            and (m.tool_calls or m.invalid_tool_calls)
        )
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
