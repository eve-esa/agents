"""Tools to search across previous conversations"""

import logging

from langchain_core.tools import tool

from .budget import dumps

logger = logging.getLogger(__name__)

RECOVERY_TOOL_NAMES = frozenset(
    {"find_results", "get_result", "search_result", "get_source", "search_memory"}
)


def recovery_tools(results, memory, on_source=None):

    async def run(action):
        try:
            return dumps(await action)
        except ValueError as exc:
            return dumps({"error": str(exc)})
        except Exception:
            logger.exception("Context recovery failed")
            return dumps(
                {
                    "error": "Stored context is temporarily unavailable. Do not assume its contents."
                }
            )

    @tool
    async def find_results(
        query: str = "", tool_name: str = "", offset: int = 0, limit: int = 5
    ) -> str:
        """Find old tool result IDs in this conversation by query/title/summary. Empty query lists recent results. Follow next_offset."""
        return await run(results.find(query, tool_name, offset, limit))

    @tool
    async def get_result(result_id: str, offset: int = 0, limit: int = 2000) -> str:
        """Read the original stored tool response. Offset is a character offset; limit is a token budget (256-4096). Follow next_offset."""
        return await run(results.read(result_id, offset, limit))

    @tool
    async def search_result(
        result_id: str, query: str, offset: int = 0, limit: int = 2000
    ) -> str:
        """Search a stored result for literal text, returning excerpts and source IDs. Offset counts matches; limit is tokens. Use get_source for citation evidence."""
        return await run(results.search(result_id, query, offset, limit))

    @tool
    async def get_source(
        result_id: str, source_id: str, offset: int = 0, limit: int = 2000
    ) -> str:
        """Read a saved retrieval passage and citation metadata, not necessarily the whole original document. Offset is characters; limit is tokens. Also restores the source to the answer's Sources panel."""

        async def read():
            page, document = await results.source(result_id, source_id, offset, limit)
            if on_source is not None:
                on_source(result_id, source_id, document)
            return page

        return await run(read())

    @tool
    async def search_memory(query: str, offset: int = 0, limit: int = 5) -> str:
        """Find historical decisions, constraints and findings omitted from active memory. Historical entries can be superseded; prefer newer corrections. Offset counts entries."""
        return await run(memory.search(query, offset, limit))

    return [find_results, get_result, search_result, get_source, search_memory]
