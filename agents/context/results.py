import hashlib
import json
import re

from .budget import bounded_page, clip, dumps, tokens


def source_records(raw: dict) -> list[dict]:
    """Stable result-local source IDs, retaining exact retrieved records."""
    records = []
    payloads = []
    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        payloads.append(structured)
    for block in raw.get("content", []):
        if block.get("type") == "text":
            try:
                payloads.append(json.loads(block.get("text", "")))
            except (ValueError, TypeError):
                pass
    seen = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for doc in payload.get("retrieved_docs", []) or []:
            if not isinstance(doc, dict):
                continue
            digest = hashlib.sha256(dumps(doc).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"source_id": f"s_{len(records) + 1}", "document": doc})
    return records


class ResultReader:
    def __init__(self, repository, document_for_model=None):
        self.repository = repository
        self.document_for_model = document_for_model or (lambda document: document)

    async def save(self, raw, tool_name, arguments):
        return await self.repository.save(raw, tool_name, arguments)

    async def load(self, result_id):
        return await self.repository.load(result_id)

    async def find(self, query="", tool_name="", offset=0, limit=5):
        return await self.repository.find(query, tool_name, offset, limit)

    def project(self, doc, raw, model_result, budget):
        return project_result(doc, raw, model_result, budget, self.document_for_model)

    async def read(self, result_id: str, offset: int = 0, limit: int = 2000) -> dict:
        _, raw = await self.load(result_id)
        return bounded_page(
            dumps(raw), offset, limit, result_id=result_id, format="raw_json"
        )

    async def source(
        self, result_id: str, source_id: str, offset: int = 0, limit: int = 2000
    ) -> tuple[dict, dict]:
        _, raw = await self.load(result_id)
        record = next(
            (s for s in source_records(raw) if s["source_id"] == source_id), None
        )
        if record is None:
            raise ValueError("Source not found in stored result")
        return bounded_page(
            dumps(self.document_for_model(record["document"])),
            offset,
            limit,
            result_id=result_id,
            source_id=source_id,
            scope="retrieved_passage",
        ), record["document"]

    async def search(
        self, result_id: str, query: str, offset: int = 0, limit: int = 2000
    ) -> dict:
        _, raw = await self.load(result_id)
        sources = source_records(raw)
        haystacks = [
            (s["source_id"], dumps(self.document_for_model(s["document"])))
            for s in sources
        ]
        if not haystacks:
            haystacks = [(None, dumps(raw))]
        matches = []
        if not query.strip():
            raise ValueError("Search query must not be empty")
        pattern = re.compile(re.escape(query[:500]), re.IGNORECASE)
        skipped = 0
        budget = max(256, min(limit, 4096))
        for source_id, text in haystacks:
            for match in pattern.finditer(text):
                if skipped < max(0, offset):
                    skipped += 1
                    continue
                item = {
                    "source_id": source_id,
                    "character_offset": match.start(),
                    "excerpt": clip(
                        text[max(0, match.start() - 100) : match.end() + 500],
                        min(120, budget - 160),
                    ),
                }
                if tokens(matches + [item]) > budget - 100:
                    return {
                        "result_id": result_id,
                        "matches": matches,
                        "next_offset": max(0, offset) + len(matches),
                        "truncated": True,
                    }
                matches.append(item)
        return {
            "result_id": result_id,
            "matches": matches,
            "next_offset": None,
            "truncated": False,
        }


def _source_preview(source, document_for_model):
    document = document_for_model(source["document"])
    text = document.get("text", "")
    if not isinstance(text, str):
        text = dumps(text)
    metadata, omitted = {}, []
    citation_fields = {"title", "url", "doi", "year", "source", "source_name", "page"}
    for key, value in (document.get("metadata") or {}).items():
        if key in citation_fields:
            # Omit long citations instead of clipping them into broken URLs/DOIs.
            if tokens(str(value)) > 120:
                omitted.append(key)
            else:
                metadata[key] = value
    return {
        "source_id": source["source_id"],
        "excerpt": clip(text, 300),
        "excerpt_truncated": tokens(text) > 300,
        "metadata": metadata,
        "omitted_metadata": omitted,
    }


def project_result(
    doc: dict, raw: dict, model_result: dict, budget: int, document_for_model=None
) -> dict:
    document_for_model = document_for_model or (lambda document: document)
    envelope = {
        "result_id": doc["_id"],
        "status": "error" if raw.get("isError") else "success",
        "summary": doc["summary"],
        "stored_result_complete": True,
    }
    sources = source_records(raw)
    if sources:
        envelope["top_results"] = []
        for source in sources:
            item = _source_preview(source, document_for_model)
            if (
                tokens({**envelope, "top_results": envelope["top_results"] + [item]})
                > budget - 150
            ):
                break
            envelope["top_results"].append(item)
        envelope["total_items"] = len(sources)
        envelope["shown_items"] = len(envelope["top_results"])
        envelope["preview_truncated"] = True  # Source excerpts may omit text.
    elif tokens(model_result) + tokens(envelope) < budget - 50:
        envelope["content"] = model_result.get("content", [])
        if model_result.get("structuredContent") is not None:
            envelope["structured_content"] = model_result["structuredContent"]
        envelope["preview_truncated"] = False
    else:
        # Artifact ingestion already replaced binary blocks with durable links.
        text = "\n".join(
            b.get("text", "")
            for b in model_result.get("content", [])
            if b.get("type") == "text"
        )
        if model_result.get("structuredContent") is not None:
            text += "\n" + dumps(model_result["structuredContent"])
        envelope["preview"] = clip(text, max(0, budget - tokens(envelope) - 100))
        envelope["preview_truncated"] = True
    while tokens(envelope) > budget:
        if envelope.get("top_results"):
            envelope["top_results"].pop()
            envelope["shown_items"] = len(envelope["top_results"])
        elif envelope.get("preview"):
            envelope["preview"] = envelope["preview"][: len(envelope["preview"]) // 2]
        else:
            break
    return envelope
