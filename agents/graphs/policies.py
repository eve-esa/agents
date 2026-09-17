"""LangGraph fault-tolerance policies shared across agent graphs.

Requires ``langgraph>=1.2`` for :class:`~langgraph.types.TimeoutPolicy` and
node-level ``error_handler`` support.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Optional, Union

from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy, TimeoutPolicy

logger = logging.getLogger(__name__)

# Default LLM attempt caps — backends can override via ``compile(llm_run_timeout=…)``.
DEFAULT_LLM_RUN_TIMEOUT = 120.0
DEFAULT_LLM_IDLE_TIMEOUT = 30.0
FALLBACK_LLM_TIMEOUT_MULTIPLIER = 1.5


def is_transient_llm_error(exc: BaseException) -> bool:
    """Best-effort predicate: should this LLM-node failure be retried in-place?

    Kept dependency-light (no ``httpx`` / ``openai`` imports) so this module
    only depends on ``langgraph``: matches builtin transient types, common
    provider error class names, transient HTTP status codes, and node timeouts.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    if name in (
        "NodeTimeoutError",
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "ServiceUnavailableError",
        "InternalServerError",
    ):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (429, 502, 503, 504)


retry_on_transient = is_transient_llm_error


def is_node_timeout_error(exc: Any) -> bool:
    """True for LangGraph ``NodeTimeoutError`` (not a stdlib ``TimeoutError``)."""
    return type(exc).__name__ == "NodeTimeoutError"


async def emit_on_policy(on_policy, **kwargs) -> None:
    """Invoke an optional backend ``on_policy`` callback; never raise to the graph."""
    if on_policy is None:
        return
    try:
        result = on_policy(**kwargs)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("on_policy callback failed")


# Retry the primary LLM in-place on transient failures before the
# ``error_handler`` escalates to the fallback model.
LLM_RETRY = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=8.0,
    jitter=True,
    retry_on=is_transient_llm_error,
)


def build_llm_timeout_policy(
    *,
    run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
) -> Optional[TimeoutPolicy]:
    """Build a :class:`TimeoutPolicy` for an LLM node.

    Returns ``None`` only when both limits are ``None`` (no node-level timeout).
    """
    if run_timeout is None and idle_timeout is None:
        return None
    return TimeoutPolicy(run_timeout=run_timeout, idle_timeout=idle_timeout)


def build_llm_fallback_timeout_policy(
    *,
    run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
) -> Optional[TimeoutPolicy]:
    """Build a :class:`TimeoutPolicy` for a fallback LLM node (1.5× primary caps)."""
    scaled_run = (
        run_timeout * FALLBACK_LLM_TIMEOUT_MULTIPLIER
        if run_timeout is not None
        else None
    )
    scaled_idle = (
        idle_timeout * FALLBACK_LLM_TIMEOUT_MULTIPLIER
        if idle_timeout is not None
        else None
    )
    return build_llm_timeout_policy(
        run_timeout=scaled_run,
        idle_timeout=scaled_idle,
    )


def make_llm_fallback_error_handler(
    *,
    fallback_node: str,
    has_fallback: bool,
    on_policy: Optional[Callable[..., Any]] = None,
    log: logging.Logger = logger,
) -> Callable[[Any, NodeError], Awaitable[Union[Command, None]]]:
    """Return an ``error_handler`` that routes to *fallback_node* on exhausted retries.

    Fires after the node's ``retry_policy`` (:data:`LLM_RETRY`) is exhausted, so
    transient blips retry in-place first and only persistent failures escalate
    here.  If no fallback model was supplied, or the failed node *is* the
    fallback, the original exception is re-raised (no loop).

    When *on_policy* is set (backend Mongo writer), emits ``timeout`` or
    ``error_handler``, then ``fallback`` before the ``Command(goto=…)``.
    """

    async def handler(state: Any, error: NodeError):
        node = getattr(error, "node", None)
        exc = getattr(error, "error", error)
        policy = "timeout" if is_node_timeout_error(exc) else "error_handler"
        await emit_on_policy(
            on_policy,
            node=node,
            policy=policy,
            error=exc if isinstance(exc, Exception) else None,
        )
        if not has_fallback or node == fallback_node:
            raise exc
        log.warning(
            "Node %s failed (%s: %s), routing to fallback node %r",
            node,
            type(exc).__name__,
            exc,
            fallback_node,
        )
        await emit_on_policy(
            on_policy,
            node=node,
            policy="fallback",
            error=exc if isinstance(exc, Exception) else None,
        )
        return Command(goto=fallback_node)

    return handler


def llm_node_add_kwargs(
    *,
    fallback_node: str,
    has_fallback: bool,
    llm_run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    llm_idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
    on_policy: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """``add_node`` keyword args for a fault-tolerant LLM node.

    Combines an in-place :data:`LLM_RETRY` for transient failures with an
    ``error_handler`` that routes to *fallback_node* once retries are exhausted.
    """
    kwargs: dict[str, Any] = {
        "retry_policy": LLM_RETRY,
        "error_handler": make_llm_fallback_error_handler(
            fallback_node=fallback_node,
            has_fallback=has_fallback,
            on_policy=on_policy,
        ),
    }
    timeout = build_llm_timeout_policy(
        run_timeout=llm_run_timeout,
        idle_timeout=llm_idle_timeout,
    )
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs
