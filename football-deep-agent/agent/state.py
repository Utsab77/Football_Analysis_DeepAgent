"""Shared state object passed through the agent loop.

Keep this a plain, serializable structure -- it gets logged to disk
after every run (see Phase 2 deliverable: save the complete execution log).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AgentState:
    task: str
    plan: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    result: dict | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def log_tool_call(self, tool_name: str, args: dict, result) -> None:
        self.tool_calls.append({"tool": tool_name, "args": args, "result": result})

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "plan": self.plan,
            "completed_steps": self.completed_steps,
            "tool_calls": self.tool_calls,
            "result": self.result,
            "started_at": self.started_at,
        }
