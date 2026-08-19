"""Task planning: turn a user request into an ordered list of steps.

Phase 3 (Weeks 5-6) target:
    [ ] Retrieve team information
    [ ] Retrieve recent form
    [ ] Retrieve H2H
    [ ] Calculate features
    [ ] Run ML model
    [ ] Evaluate prediction
    [ ] Generate explanation

Start Phase 2 with a hardcoded plan for the single example task, then
replace with LLM-generated planning once the rest of the loop works.
"""
from __future__ import annotations

DEFAULT_ANALYSIS_PLAN = [
    "Retrieve team information",
    "Retrieve recent form",
    "Retrieve head-to-head",
    "Calculate features",
    "Run ML model",
    "Evaluate prediction",
    "Generate explanation",
]


def make_plan(task: str) -> list[str]:
    # TODO (Phase 3): replace with an LLM call that decomposes `task`
    # into steps, and support re-planning when a step fails.
    return list(DEFAULT_ANALYSIS_PLAN)
