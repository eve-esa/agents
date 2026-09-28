
import contextvars
import logging
from dataclasses import dataclass

from mcp.types import CallToolResult, TextContent

from .budget import bounded_page, dumps

logger = logging.getLogger(__name__)


@dataclass
class Capture:
    raw: dict | None = None
    stored: dict | None = None


_capture = contextvars.ContextVar("tool_result_capture", default=None)


class CaptureResultInterceptor:

    def __init__(self, get_context):
        self.get_context = get_context

    async def __call__(self, request, handler):
        result = await handler(request)
        capture = _capture.get()
        ctx = self.get_context()
        if (
            capture is not None
            and ctx is not None
            and isinstance(result, CallToolResult)
        ):
            capture.raw = result.model_dump(mode="json", by_alias=True)
            try:
                name = "_".join(
                    filter(None, [getattr(request, "server_name", ""), request.name])
                )
                capture.stored = await ctx.results.save(
                    capture.raw,
                    name,
                    getattr(request, "args", {}) or {},
                )
            except Exception:
                logger.exception(
                    "Tool snapshot persistence failed; returning a bounded unavailable preview"
                )
        return result


class ProjectResultInterceptor:

    def __init__(self, get_context, on_projection=None):
        self.get_context = get_context
        self.on_projection = on_projection

    async def __call__(self, request, handler):
        ctx = self.get_context()
        if ctx is None:
            return await handler(request)
        capture = Capture()
        token = _capture.set(capture)
        try:
            result = await handler(request)
            if not isinstance(result, CallToolResult):
                return result
            if capture.stored:
                envelope = ctx.results.project(
                    capture.stored,
                    capture.raw,
                    result.model_dump(mode="json", by_alias=True),
                    ctx.settings.preview_tokens,
                )
                if self.on_projection is not None:
                    self.on_projection(request, capture.stored, capture.raw)
            else:
                envelope = bounded_page(
                    "\n".join(
                        b.text for b in result.content if isinstance(b, TextContent)
                    ),
                    0,
                    ctx.settings.preview_tokens,
                    status="error" if result.isError else "success",
                    storage_status="unavailable",
                    stored_result_complete=False,
                    preview_truncated=True,
                    warning="Full result could not be stored. No recovery ID is available.",
                )
                envelope.pop("next_offset", None)
            return result.model_copy(
                update={
                    "content": [TextContent(type="text", text=dumps(envelope))],
                    "structuredContent": None,
                }
            )
        finally:
            _capture.reset(token)
