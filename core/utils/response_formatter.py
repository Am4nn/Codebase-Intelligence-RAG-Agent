"""Utilities for normalizing LLM/tool responses to plain text."""

from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)


def format_response(resp: object) -> str:
    """Normalize responses from various LLM and tool return formats into a string.

    Accepts strings, dicts (with `messages` or `content`), lists of message-like
    objects, and objects with attributes like `content`, `generations` or `text`.
    """
    try:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            if "messages" in resp and resp["messages"]:
                last = resp["messages"][-1]
                if hasattr(last, "content"):
                    return str(last.content)
                if isinstance(last, dict) and "content" in last:
                    return str(last["content"])
                return str(last)
            if "content" in resp:
                return str(resp["content"])
            return str(resp)
        if isinstance(resp, list):
            parts: list[str] = []
            for item in resp:
                if hasattr(item, "content"):
                    parts.append(str(item.content))
                elif isinstance(item, dict) and "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if hasattr(resp, "content"):
            return str(getattr(resp, "content"))
        if hasattr(resp, "generations"):
            try:
                gens = getattr(resp, "generations")
                first = gens[0]
                if isinstance(first, list):
                    gen = first[0]
                else:
                    gen = first
                if hasattr(gen, "text"):
                    return str(gen.text)
                if hasattr(gen, "content"):
                    return str(gen.content)
            except Exception:
                pass
        return str(resp)
    except Exception:
        logger.exception("Error formatting response")
        return str(resp)
