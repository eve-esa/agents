import json
from functools import lru_cache

import tiktoken

#cache this tokenizer for later use
@lru_cache(maxsize=1)
def encoding():
    return tiktoken.get_encoding("cl100k_base")


def dumps(value) -> str:
    """convert to JSON"""
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def tokens(value) -> int:
    return len(
        encoding().encode(
            value if isinstance(value, str) else dumps(value), disallowed_special=()
        )
    )


def clip(text: str, limit: int) -> str:
    """returns at most limit tokens from the start of a string"""
    ids = encoding().encode(text, disallowed_special=())[: max(0, limit)]
    return encoding().decode_bytes(ids).decode("utf-8", errors="ignore")


def bounded_page(text: str, offset: int, limit: int, **metadata) -> dict:
    """returns text + metadata"""
    limit = max(256, min(limit, 4096))
    offset = max(0, min(offset, len(text)))
    content = clip(text[offset:], max(0, limit - tokens(metadata) - 100))
    while True:
        end = offset + len(content)
        page = {
            **metadata,
            "content": content,
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "total_characters": len(text),
            "truncated": end < len(text),
        }
        if tokens(page) <= limit or not content:
            return page
        content = content[: max(1, len(content) * 3 // 4)]
