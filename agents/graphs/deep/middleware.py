"""EVE middleware for the deep agent graph.

Built-in LangChain middleware does the work: ``ModelFallbackMiddleware``,
``ModelRetryMiddleware`` and ``ToolErrorMiddleware``.  The two subclasses below
keep the parent loops untouched and only wrap the ``handler`` the parent calls,
to add what the built-ins have no hook for: backend ``on_policy`` events, the
fallback marker and a per-attempt run cap.  ``EveCompatMiddleware`` keeps the
``ReactAgent`` prompt and text-tool-call behaviour, and
``EveSummarizationMiddleware`` replaces the deepagents summarization with
one whose trigger and tail are fractions of a known context window.
:func:`offload_filesystem` writes tool results over a tiktoken limit to
``/large_tool_results/``.  The same limit is the deepagents character cap on
one ``read_file`` page.  Human messages stay in the thread.

Events are emitted on the async path only (``astream`` / ``ainvoke``); the sync
path runs the parent behaviour unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from deepagents import FilesystemMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.middleware._message_eviction import (
    _aoffload_tool_message_content,
    _extract_text_from_message,
    _offload_tool_message_content,
)
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphBubbleUp

from ..policies import (
    DEFAULT_LLM_RUN_TIMEOUT,
    FALLBACK_LLM_TIMEOUT_MULTIPLIER,
    LLM_RETRY,
    emit_on_policy,
    is_timeout_error,
    mark_fallback,
)
from ..utils import (
    clock_context,
    parse_text_tool_calls,
    reformat_messages_for_text_tool_model,
    strip_content_from_tool_call_messages,
    tiktoken_counter,
)

logger = logging.getLogger(__name__)

# ``create_agent`` runs every model call in this node.
MODEL_NODE = "model"

PolicyCallback = Optional[Callable[..., Any]]


def _model_name(model: Any) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _policy_extra(model: Any, agent: Optional[str]) -> dict:
    extra = {"model": _model_name(model)}
    if agent:
        extra["agent"] = agent
    return extra


def _mark_fallback(response: Any, latency_s: float) -> None:
    messages = response.result if isinstance(response, ModelResponse) else [response]
    mark_fallback(messages, latency_s)


class EveFallbackMiddleware(ModelFallbackMiddleware):
    """``ModelFallbackMiddleware`` that reports each failed model to ``on_policy``.

    For every failed model it emits ``timeout`` or ``error_handler``; when
    another model follows it also emits ``fallback``.  These events carry
    ``extra["failed_after_s"]``, the seconds that model spent before failing,
    retries included.  Answers from a fallback model are stamped with
    :func:`agents.graphs.policies.mark_fallback`.  With no fallback
    models it still emits, then re-raises, like the ``react`` error handler.
    """

    def __init__(
        self,
        *models: BaseChatModel,
        on_policy: PolicyCallback = None,
        agent: Optional[str] = None,
    ) -> None:
        AgentMiddleware.__init__(self)
        self.models = list(models)
        self.on_policy = on_policy
        self.agent = agent

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        primary = request.model
        calls = 0

        async def observed(req: ModelRequest) -> ModelResponse:
            nonlocal calls
            calls += 1
            has_next = calls <= len(self.models)
            start = time.perf_counter()
            try:
                response = await handler(req)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                extra = _policy_extra(req.model, self.agent)
                extra["failed_after_s"] = round(time.perf_counter() - start, 3)
                await emit_on_policy(
                    self.on_policy,
                    node=MODEL_NODE,
                    policy="timeout" if is_timeout_error(exc) else "error_handler",
                    error=exc,
                    extra=extra,
                )
                if has_next:
                    logger.warning(
                        "Model %s failed (%s: %s), trying the next fallback model",
                        extra["model"],
                        type(exc).__name__,
                        exc,
                    )
                    await emit_on_policy(
                        self.on_policy,
                        node=MODEL_NODE,
                        policy="fallback",
                        error=exc,
                        extra=extra,
                    )
                raise
            if req.model is not primary:
                _mark_fallback(response, round(time.perf_counter() - start, 3))
            return response

        return await super().awrap_model_call(request, observed)


class EveRetryMiddleware(ModelRetryMiddleware):
    """``ModelRetryMiddleware`` with ``LLM_RETRY`` defaults, a run cap and retry events.

    Each attempt is capped at *run_timeout* seconds (times
    ``FALLBACK_LLM_TIMEOUT_MULTIPLIER`` for a model in *fallback_models*).  The
    cap raises ``TimeoutError``, which ``is_transient_llm_error`` retries.  From
    the second attempt on it emits ``retry``.  ``on_failure="error"`` so the
    last error reaches ``EveFallbackMiddleware``.

    There is no idle cap here: the provider client's read timeout covers it.
    """

    def __init__(
        self,
        *,
        run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
        fallback_models: Sequence[BaseChatModel] = (),
        on_policy: PolicyCallback = None,
        agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("max_retries", LLM_RETRY.max_attempts - 1)
        kwargs.setdefault("retry_on", LLM_RETRY.retry_on)
        kwargs.setdefault("on_failure", "error")
        kwargs.setdefault("initial_delay", LLM_RETRY.initial_interval)
        kwargs.setdefault("backoff_factor", LLM_RETRY.backoff_factor)
        kwargs.setdefault("max_delay", LLM_RETRY.max_interval)
        kwargs.setdefault("jitter", LLM_RETRY.jitter)
        super().__init__(**kwargs)
        self.run_timeout = run_timeout
        self.fallback_models = list(fallback_models)
        self.on_policy = on_policy
        self.agent = agent

    def _run_cap(self, model: Any) -> Optional[float]:
        if self.run_timeout is None:
            return None
        if any(model is fallback for fallback in self.fallback_models):
            return self.run_timeout * FALLBACK_LLM_TIMEOUT_MULTIPLIER
        return self.run_timeout

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        cap = self._run_cap(request.model)
        name = _model_name(request.model)
        attempt = 0

        async def attempt_call(req: ModelRequest) -> ModelResponse:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                await emit_on_policy(
                    self.on_policy,
                    node=MODEL_NODE,
                    policy="retry",
                    attempt=attempt,
                    extra=_policy_extra(req.model, self.agent),
                )
            start = time.perf_counter()
            try:
                async with asyncio.timeout(cap):
                    return await handler(req)
            finally:
                logger.info(
                    "Model %s attempt %d finished in %.3fs",
                    name,
                    attempt,
                    time.perf_counter() - start,
                )

        return await super().awrap_model_call(request, attempt_call)


class EveCompatMiddleware(AgentMiddleware):
    """Keep ``ReactAgent`` request and response handling.

    - Puts ``clock_context()`` before the system prompt on every call.
    - Strips text from assistant messages that also carry tool calls
      (Mistral rejects both), and rewrites tool history as text once any tool
      call is present.
    - Drops ``AIMessage.name`` on the request. ``create_deep_agent(name="deep")``
      stores that name, and Mistral rejects it on an assistant message.
    - Parses ``[TOOL_CALLS]`` text output into real tool calls.
    """

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        messages = _without_assistant_name(
            strip_content_from_tool_call_messages(list(request.messages))
        )
        has_tool_history = any(
            (isinstance(m, AIMessage) and not m.content and getattr(m, "tool_calls", None))
            or isinstance(m, ToolMessage)
            for m in messages
        )
        if has_tool_history:
            messages = reformat_messages_for_text_tool_model(messages)
        return request.override(
            system_message=_with_clock(request.system_message),
            messages=messages,
        )

    def _parse(self, response: ModelResponse) -> ModelResponse:
        result: List[Any] = []
        for message in response.result:
            if (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and isinstance(message.content, str)
            ):
                parsed = parse_text_tool_calls(message.content)
                if parsed:
                    logger.info(
                        "Parsed %d text-format tool call(s) from model response",
                        len(parsed),
                    )
                    message = AIMessage(
                        content="",
                        tool_calls=parsed,
                        id=message.id,
                        response_metadata=message.response_metadata,
                        usage_metadata=message.usage_metadata,
                    )
            result.append(message)
        return ModelResponse(
            result=result, structured_response=response.structured_response
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return self._parse(handler(self._prepare(request)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return self._parse(await handler(self._prepare(request)))


# The chat models have no ``profile.max_input_tokens``, so the fractions
# below are turned into token counts against this window.  Deep Agents would
# otherwise fall back to a 170,000 token trigger.
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_SUMMARY_TRIGGER_FRACTION = 0.85
DEFAULT_SUMMARY_KEEP_FRACTION = 0.10
DEFAULT_SUMMARY_MAX_TOKENS = 2_000
DEFAULT_TRUNCATE_ARGS_MAX_CHARS = 2_000
DEFAULT_MESSAGE_ID_KEY = "eve_message_id"

# Messages the in-flight summarization is about to replace.  ``wrap_model_call``
# partitions and then builds the summary message on the same task.
_summarized_messages: ContextVar[Optional[List[Any]]] = ContextVar(
    "eve_summarized_messages", default=None
)


def _token_budget(context_window: int, fraction: float) -> int:
    """Token count for *fraction* of *context_window*.

    Deep Agents rejects a token threshold of zero, so a fraction that rounds
    to zero is an error rather than a silent full-history summary.
    """
    if context_window <= 0:
        raise ValueError(f"context_window must be positive, got {context_window}.")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    tokens = int(context_window * fraction)
    if tokens <= 0:
        raise ValueError(
            f"context_window {context_window} times {fraction} rounds to 0 tokens."
        )
    return tokens


def _capped_summarizer(model: BaseChatModel, summary_max_tokens: int) -> BaseChatModel:
    """The graph model, with its completion capped at *summary_max_tokens*.

    A model with no ``max_tokens`` field is returned unchanged.  The copy does
    not replace the model that answers the user.
    """
    fields = getattr(type(model), "model_fields", None)
    if not isinstance(fields, dict) or "max_tokens" not in fields:
        return model
    return model.model_copy(update={"max_tokens": summary_max_tokens})


class EveSummarizationMiddleware(SummarizationMiddleware):
    """Summarize older history once the prompt fills most of the window.

    Its ``name`` replaces the deepagents ``SummarizationMiddleware`` in place,
    on the main agent and on every declarative subagent.  Trigger and keep are
    token counts, ``context_window`` times the fractions.  ``write_file`` and
    ``edit_file`` arguments in the older part are shortened first; the parent
    summarizes only when the prompt is still over the trigger.  The cut keeps
    each tool call with its results.  The thread keeps every message.

    The summary message names the ``message_id_key`` values of the first and
    last summarized human message.  When ``recall_note`` is set it is appended
    and the ``/conversation_history/`` path is left out: that file lives on
    the sandbox, which does not outlive the session.
    """

    def __init__(
        self,
        model: BaseChatModel,
        backend: Any,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        summary_trigger_fraction: float = DEFAULT_SUMMARY_TRIGGER_FRACTION,
        summary_keep_fraction: float = DEFAULT_SUMMARY_KEEP_FRACTION,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
        truncate_args_max_chars: int = DEFAULT_TRUNCATE_ARGS_MAX_CHARS,
        recall_note: Optional[str] = None,
        message_id_key: str = DEFAULT_MESSAGE_ID_KEY,
    ) -> None:
        trigger = ("tokens", _token_budget(context_window, summary_trigger_fraction))
        keep = ("tokens", _token_budget(context_window, summary_keep_fraction))
        super().__init__(
            model=_capped_summarizer(model, summary_max_tokens),
            backend=backend,
            trigger=trigger,
            keep=keep,
            # The constructor default keeps only the last 4,000 tokens of the
            # older part.  The whole older part is what the summary covers.
            trim_tokens_to_summarize=None,
            truncate_args_settings={
                "trigger": trigger,
                "keep": keep,
                "max_length": truncate_args_max_chars,
            },
        )
        self._recall_note = (recall_note or "").strip()
        self._message_id_key = message_id_key

    @property
    def name(self) -> str:
        # A subclass of the deepagents middleware reports its class name.
        # create_deep_agent only swaps the default when the names match.
        return "SummarizationMiddleware"

    def _partition_messages(
        self, conversation_messages: List[Any], cutoff_index: int
    ) -> tuple:
        parts = super()._partition_messages(conversation_messages, cutoff_index)
        _summarized_messages.set(list(parts[0]))
        return parts

    def _covered_ids(self, messages: Optional[List[Any]]) -> List[str]:
        found: List[str] = []
        for message in messages or []:
            if not isinstance(message, HumanMessage):
                continue
            value = (getattr(message, "additional_kwargs", None) or {}).get(
                self._message_id_key
            )
            if value:
                found.append(str(value))
        return found

    def _id_range_line(self, ids: List[str]) -> Optional[str]:
        if not ids:
            return None
        if ids[0] == ids[-1]:
            return (
                f"Summarized message_id {ids[0]}. "
                "Pass it to read_history to read the raw messages."
            )
        return (
            f"Summarized message_id {ids[0]} through {ids[-1]}. "
            "Pass either id to read_history to read the raw messages."
        )

    def _build_new_messages_with_path(
        self, summary: str, file_path: Optional[str]
    ) -> List[Any]:
        range_line = self._id_range_line(self._covered_ids(_summarized_messages.get()))
        # The history file is on the sandbox.  A recall note points at the
        # stored messages instead, which outlive that sandbox.
        if self._recall_note:
            file_path = None
        lines = ["You are in the middle of a conversation that has been summarized."]
        if file_path is not None:
            lines.append(
                "The full conversation history has been saved to "
                f"{file_path} should you need to refer back to it for details."
            )
        if range_line is not None:
            lines.append(range_line)
        if self._recall_note:
            lines.append(self._recall_note)
        lines.append(
            f"A condensed summary follows:\n\n<summary>\n{summary}\n</summary>"
        )
        return [
            HumanMessage(
                content="\n\n".join(lines),
                additional_kwargs={"lc_source": "summarization"},
            )
        ]


def _without_assistant_name(messages: List[Any]) -> List[Any]:
    """Copy assistant messages without ``name`` for the provider request.

    The thread keeps the original. A tool message keeps its name: that field
    is the tool, and the provider requires it.
    """
    cleaned: List[Any] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.name:
            cleaned.append(message.model_copy(update={"name": None}))
        else:
            cleaned.append(message)
    return cleaned


def _with_clock(system: Optional[SystemMessage]) -> SystemMessage:
    clock = clock_context()
    if system is None:
        return SystemMessage(content=clock)
    if isinstance(system.content, str):
        content: Any = f"{clock}\n{system.content}" if system.content else clock
    else:
        content = [{"type": "text", "text": clock}, *system.content]
    return system.model_copy(update={"content": content})


def tool_error_message(exc: Exception, request: ToolCallRequest) -> str:
    """``ToolErrorMiddleware`` handler: same text the ``react`` tools node returns."""
    logger.error("Tool %s failed: %s", request.tool_call["name"], exc)
    return f"Tool error: {exc}"


# Same default as deepagents ``FilesystemMiddleware``.
DEFAULT_TOOL_RESULT_MAX_TOKENS = 20_000


def _content_tokens(text: str) -> int:
    """Token count of *text* with the same counter the offload decision uses.

    :func:`tiktoken_counter` adds 4 per message.  Those are not part of the
    tool result, so they come back off.
    """
    return max(0, tiktoken_counter([HumanMessage(content=text)]) - 4)


class EveFilesystemMiddleware(FilesystemMiddleware):
    """``FilesystemMiddleware`` that offloads by :func:`tiktoken_counter`.

    deepagents decides from ``len(text) / 4``.  That leaves dense text, such
    as CJK, inline while it is already over the token budget the
    summarization trigger uses.  A result over *tool_result_max_tokens* real
    tokens is written to ``/large_tool_results/`` here.  A result under that
    count still goes through the parent, so a long ASCII result offloads on
    the character estimate too.

    The parent also uses this limit as the character cap of one ``read_file``
    page (about 4 characters per token).  ``None`` turns both off.  Human
    messages are never moved to ``/conversation_history/``.

    The instance name stays ``FilesystemMiddleware``, so it replaces the
    deepagents default in place.  ``create_deep_agent`` copies the agent's
    permissions onto it.  Harness-profile tool description overrides for the
    file tools are not applied.
    """

    def __init__(self, backend: Any, tool_result_max_tokens: Optional[int]) -> None:
        super().__init__(
            backend=backend,
            tool_token_limit_before_evict=tool_result_max_tokens,
            human_message_token_limit_before_evict=None,
        )

    @property
    def name(self) -> str:
        return "FilesystemMiddleware"

    def _offload_over_budget(self, message: ToolMessage, resolved_backend: Any, *, sync: bool):
        if not self._tool_token_limit_before_evict:
            return None
        content = _extract_text_from_message(message)
        if _content_tokens(content) <= self._tool_token_limit_before_evict:
            return None
        offload = _offload_tool_message_content if sync else _aoffload_tool_message_content
        return offload(message, content, resolved_backend, self._large_tool_results_prefix)

    def _process_large_message(self, message: ToolMessage, resolved_backend: Any):
        offloaded = self._offload_over_budget(message, resolved_backend, sync=True)
        if offloaded is None and _content_tokens(_extract_text_from_message(message)) > (
            self._tool_token_limit_before_evict or 0
        ):
            return message, False
        if offloaded is not None:
            return offloaded, True
        return super()._process_large_message(message, resolved_backend)

    async def _aprocess_large_message(self, message: ToolMessage, resolved_backend: Any):
        offloaded = self._offload_over_budget(message, resolved_backend, sync=False)
        if offloaded is not None:
            offloaded = await offloaded
        over = _content_tokens(_extract_text_from_message(message)) > (
            self._tool_token_limit_before_evict or 0
        )
        if offloaded is None and over:
            return message, False
        if offloaded is not None:
            return offloaded, True
        return await super()._aprocess_large_message(message, resolved_backend)


def offload_filesystem(
    backend: Any, tool_result_max_tokens: Optional[int]
) -> EveFilesystemMiddleware:
    """Filesystem middleware for one agent.  See :class:`EveFilesystemMiddleware`."""
    return EveFilesystemMiddleware(backend, tool_result_max_tokens)


def filesystem_keeping_offload(
    middleware: Any, tool_result_max_tokens: Optional[int]
) -> FilesystemMiddleware:
    """Keep *middleware*'s backend and permissions, and apply this graph's limit.

    A declarative subagent can pass its own ``FilesystemMiddleware``.  The
    merge replaces the EVE one by name, which would drop
    *tool_result_max_tokens* and the tiktoken check.  Rebuild it so the
    graph's limit still applies.
    """
    if (
        isinstance(middleware, EveFilesystemMiddleware)
        and middleware._tool_token_limit_before_evict == tool_result_max_tokens
        and middleware._human_message_token_limit_before_evict is None
    ):
        return middleware
    rebuilt = EveFilesystemMiddleware(
        backend=getattr(middleware, "backend", None),
        tool_result_max_tokens=tool_result_max_tokens,
    )
    permissions = list(getattr(middleware, "_permissions", []) or [])
    if permissions:
        return rebuilt._with_permissions(permissions)
    return rebuilt


def build_eve_middleware(
    *,
    backend: Any,
    model: BaseChatModel,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    summary_trigger_fraction: float = DEFAULT_SUMMARY_TRIGGER_FRACTION,
    summary_keep_fraction: float = DEFAULT_SUMMARY_KEEP_FRACTION,
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    truncate_args_max_chars: int = DEFAULT_TRUNCATE_ARGS_MAX_CHARS,
    recall_note: Optional[str] = None,
    message_id_key: str = DEFAULT_MESSAGE_ID_KEY,
    tool_result_max_tokens: Optional[int] = DEFAULT_TOOL_RESULT_MAX_TOKENS,
    fallback_llm: Optional[BaseChatModel] = None,
    run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    on_policy: PolicyCallback = None,
    agent: Optional[str] = None,
) -> List[AgentMiddleware]:
    """Return fresh EVE middleware instances.

    :func:`offload_filesystem` and ``EveSummarizationMiddleware`` take the
    filesystem and summarization slots of the deepagents stack; the rest run
    after the deepagents base stack, outermost first.  Build a new list for
    each agent (main and every subagent) so its policy events carry that
    agent's label in ``extra["agent"]``.
    """
    fallbacks = [fallback_llm] if fallback_llm is not None else []
    return [
        offload_filesystem(backend, tool_result_max_tokens),
        EveSummarizationMiddleware(
            model,
            backend,
            context_window=context_window,
            summary_trigger_fraction=summary_trigger_fraction,
            summary_keep_fraction=summary_keep_fraction,
            summary_max_tokens=summary_max_tokens,
            truncate_args_max_chars=truncate_args_max_chars,
            recall_note=recall_note,
            message_id_key=message_id_key,
        ),
        EveFallbackMiddleware(*fallbacks, on_policy=on_policy, agent=agent),
        EveRetryMiddleware(
            run_timeout=run_timeout,
            fallback_models=fallbacks,
            on_policy=on_policy,
            agent=agent,
        ),
        EveCompatMiddleware(),
        ToolErrorMiddleware(on_error=tool_error_message),
    ]
