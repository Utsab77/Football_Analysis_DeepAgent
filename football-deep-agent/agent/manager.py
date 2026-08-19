"""Manager agent: the top-level loop -- perceive -> plan -> act -> observe -> repeat.

Phase 2 (Weeks 3-4): build this WITHOUT a framework first. Wire in an
LLM API directly (Anthropic or OpenAI, see .env.example) for the
planning/reasoning calls.
"""
from __future__ import annotations

from agent.state import AgentState
from agent.planner import make_plan


def run(task: str, tools: dict) -> AgentState:
    """Run one end-to-end agent task.

    `tools` is a dict of {tool_name: callable} -- see tools/ for the
    football-specific tool implementations to register here.
    """
    state = AgentState(task=task)
    state.plan = make_plan(task)

    # TODO (Phase 2): for each planned step, decide which tool to call,
    # call it, log the result to state, and update the plan. Start with
    # a hardcoded step sequence for one example task ("Analyze X vs Y and
    # predict the outcome") before generalizing to LLM-driven tool choice.

    return state
