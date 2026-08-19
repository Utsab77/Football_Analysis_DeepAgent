"""Context management: summarize/externalize large tool outputs so the
agent receives compact, relevant context instead of raw dumps.

Phase 4 (Weeks 7-8).
"""
from __future__ import annotations

MAX_INLINE_CHARS = 2000


def compress(tool_name: str, raw_output: str) -> str:
    """Return a compact summary if `raw_output` is large; otherwise pass through.

    TODO: replace the naive truncation below with an actual summarization
    step (rule-based first, LLM-based later) once you have real tool
    outputs to test against.
    """
    if len(raw_output) <= MAX_INLINE_CHARS:
        return raw_output
    return raw_output[:MAX_INLINE_CHARS] + f"... [truncated, full output stored under '{tool_name}']"
