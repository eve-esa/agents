"""EVE middleware for the deep agent graph.

Built-in LangChain middleware does the work: ``ModelFallbackMiddleware``,
``ModelRetryMiddleware`` and ``ToolErrorMiddleware``.  The two subclasses below
keep the parent loops untouched and only wrap the ``handler`` the parent calls,
to add what the built-ins have no hook for: backend ``on_policy`` events, the
fallback marker and a per-attempt run cap.  ``EveCompatMiddleware`` keeps the
``ReactAgent`` prompt and text-tool-call behaviour, and ``EveTrimMiddleware``
replaces the deepagents summarization with the ``ReactAgent`` history trim.
:func:`offload_filesystem` writes tool results over a token limit to
``/large_tool_results/`` and caps one ``read_file`` page at that same limit.
Human messages stay in the thread.

Events are emitted on the async path only (``astream`` / ``ainvoke``); the sync
path runs the parent behaviour unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from deepagents import FilesystemMiddleware
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
    SystemMessage,
    ToolMessage,
    trim_messages,
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
    - Parses ``[TOOL_CALLS]`` text output into real tool calls.
    """

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        messages = strip_content_from_tool_call_messages(list(request.messages))
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


class EveTrimMiddleware(AgentMiddleware):
    """Trim each model request to *max_tokens*, like ``ReactAgent``.

    Uses the same ``trim_messages`` settings as the ``react`` agent node, with
    the system prompt counted in the budget.  Only the request is trimmed; the
    thread state keeps every message.

    Its ``name`` replaces the deepagents ``SummarizationMiddleware`` in place,
    on the main agent and on every declarative subagent.
    """

    def __init__(self, max_tokens: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        # create_deep_agent swaps in custom middleware whose name matches a
        # default; this is the name its summarization middleware reports.
        return "SummarizationMiddleware"

    def _trim(self, request: ModelRequest) -> ModelRequest:
        system = request.system_message
        messages = ([system] if system is not None else []) + list(request.messages)
        trimmed = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            strategy="last",
            token_counter=tiktoken_counter,
            include_system=True,
            start_on="human",
            end_on=("human", "tool"),
        )
        if system is not None and trimmed and trimmed[0] is system:
            trimmed = trimmed[1:]
        return request.override(messages=trimmed)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._trim(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._trim(request))


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


# Same default as deepagents ``FilesystemMiddleware``.  About 80,000 characters.
DEFAULT_TOOL_RESULT_MAX_TOKENS = 20_000


def offload_filesystem(
    backend: Any, tool_result_max_tokens: Optional[int]
) -> FilesystemMiddleware:
    """``FilesystemMiddleware`` that offloads large tool results.

    A tool result above *tool_result_max_tokens* is written to
    ``/large_tool_results/`` and replaced in the thread by its path and a
    head-and-tail preview.  deepagents uses that same limit as the token cap
    on one ``read_file`` page (``filesystem.py``), so a page is truncated and
    never offloaded again.  ``None`` turns both off: every result stays whole
    and a page has no token cap.  Human messages are never moved to
    ``/conversation_history/``.

    *backend* must be the one passed to ``create_deep_agent``.  Its name
    replaces the default ``FilesystemMiddleware`` in place, and the agent's
    ``permissions`` are copied onto it.  Harness-profile tool description
    overrides for the file tools are not applied.
    """
    return FilesystemMiddleware(
        backend=backend,
        tool_token_limit_before_evict=tool_result_max_tokens,
        human_message_token_limit_before_evict=None,
    )


def build_eve_middleware(
    *,
    backend: Any,
    max_tokens: int,
    tool_result_max_tokens: Optional[int] = DEFAULT_TOOL_RESULT_MAX_TOKENS,
    fallback_llm: Optional[BaseChatModel] = None,
    run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    on_policy: PolicyCallback = None,
    agent: Optional[str] = None,
) -> List[AgentMiddleware]:
    """Return fresh EVE middleware instances.

    :func:`offload_filesystem` and ``EveTrimMiddleware`` take the filesystem
    and summarization slots of the deepagents stack; the rest run after the
    deepagents base stack, outermost first.  Build a new list for each agent
    (main and every subagent) so its policy events carry that agent's label in
    ``extra["agent"]``.
    """
    fallbacks = [fallback_llm] if fallback_llm is not None else []
    return [
        offload_filesystem(backend, tool_result_max_tokens),
        EveTrimMiddleware(max_tokens),
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
