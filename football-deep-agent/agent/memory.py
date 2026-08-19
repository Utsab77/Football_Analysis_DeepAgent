"""Persistent memory: previous analyses, tool results, completed/failed tasks.

Phase 3 (Weeks 5-6). Start with a simple JSON-file-backed store; move to
PostgreSQL (+pgvector if you want semantic recall) once the interface
is stable.
"""
from __future__ import annotations

import json
from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "agent_memory.json"


def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {"analyses": [], "tool_results": [], "failed_operations": []}
    return json.loads(MEMORY_PATH.read_text())


def save_memory(memory: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(memory, indent=2, default=str))


def record_analysis(memory: dict, task: str, result: dict) -> dict:
    memory["analyses"].append({"task": task, "result": result})
    return memory
