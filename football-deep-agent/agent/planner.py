"""Task planning: turn a user request into an ordered list of steps.

Phase 3 (Weeks 5-6): 
- Hardcoded plans for known task types
- LLM-generated planning for complex requests
- Re-planning support when steps fail
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an analysis plan."""
    description: str
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: dict | None = None
    error: str | None = None


@dataclass
class AnalysisPlan:
    """A complete analysis plan with ordered steps."""
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step: int = 0
    
    def add_step(self, description: str, tool_name: str = None, tool_args: dict = None):
        """Add a step to the plan."""
        self.steps.append(PlanStep(
            description=description,
            tool_name=tool_name,
            tool_args=tool_args or {},
        ))
    
    def get_next_step(self) -> PlanStep | None:
        """Get the next pending step, or None if all done."""
        while self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            if step.status == StepStatus.PENDING:
                return step
            self.current_step += 1
        return None
    
    def mark_step_complete(self, step: PlanStep, result: dict):
        """Mark a step as completed with its result."""
        step.status = StepStatus.COMPLETED
        step.result = result
        self.current_step += 1
    
    def mark_step_failed(self, step: PlanStep, error: str):
        """Mark a step as failed with error message."""
        step.status = StepStatus.FAILED
        step.error = error
        self.current_step += 1
    
    def to_dict(self) -> dict:
        """Serialize plan to dict."""
        return {
            "task": self.task,
            "steps": [
                {
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "status": s.status.value,
                    "result": s.result,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "current_step": self.current_step,
        }


# Pre-defined task templates for known analysis types
MATCH_PREDICTION_TEMPLATE = [
    ("Retrieve team information", "get_team_stats", {}),
    ("Retrieve recent form", "get_recent_form", {}),
    ("Retrieve head-to-head", "get_head_to_head", {}),
    ("Calculate features", "calculate_features", {}),
    ("Run ML model", "run_prediction_model", {}),
    ("Evaluate prediction", "evaluate_prediction", {}),
    ("Generate explanation", "generate_explanation", {}),
]

FORM_ANALYSIS_TEMPLATE = [
    ("Retrieve team information", "get_team_stats", {}),
    ("Retrieve recent form", "get_recent_form", {}),
    ("Analyze home/away performance", "analyze_home_away", {}),
    ("Generate form report", "generate_form_report", {}),
]

H2H_ANALYSIS_TEMPLATE = [
    ("Retrieve head-to-head data", "get_head_to_head", {}),
    ("Analyze historical trends", "analyze_historical_trends", {}),
    ("Generate H2H report", "generate_h2h_report", {}),
]


def make_plan(task: str) -> AnalysisPlan:
    """Create an analysis plan for a given task.
    
    Starts with hardcoded templates, will be replaced with LLM-generated
    planning once the rest of the loop works.
    """
    plan = AnalysisPlan(task=task)
    
    task_lower = task.lower()
    
    # Select template based on task content
    if "predict" in task_lower or "vs" in task_lower or " vs " in task_lower:
        template = MATCH_PREDICTION_TEMPLATE
    elif "form" in task_lower or "recent" in task_lower:
        template = FORM_ANALYSIS_TEMPLATE
    elif "head-to-head" in task_lower or "h2h" in task_lower or "history" in task_lower:
        template = H2H_ANALYSIS_TEMPLATE
    else:
        # Default to match prediction template
        template = MATCH_PREDICTION_TEMPLATE
    
    # Add steps from template
    for desc, tool, args in template:
        plan.add_step(desc, tool, args)
    
    return plan


def replan_on_failure(plan: AnalysisPlan, failed_step: PlanStep, error: str) -> AnalysisPlan:
    """Re-plan when a step fails.
    
    Determines if the failure is recoverable and adjusts the plan accordingly.
    """
    # Check if this is a recoverable failure
    recoverable_errors = [
        "data not available",
        "timeout",
        "temporary failure",
        "rate limit",
    ]
    
    is_recoverable = any(err in error.lower() for err in recoverable_errors)
    
    if is_recoverable:
        # Add retry step
        retry_step = PlanStep(
            description=f"Retry: {failed_step.description}",
            tool_name=failed_step.tool_name,
            tool_args=failed_step.tool_args,
            status=StepStatus.PENDING,
        )
        plan.steps.insert(plan.current_step, retry_step)
    else:
        # Skip to next logical step or use fallback
        # Mark dependent steps as skipped
        for step in plan.steps[plan.current_step:]:
            if step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
    
    return plan


def extract_teams_from_task(task: str) -> tuple[str, str]:
    """Extract home and away team names from a task string.
    
    Handles formats like:
    - "Predict Arsenal vs Chelsea"
    - "Analyze Arsenal vs Chelsea match"
    - "Arsenal vs Chelsea prediction"
    """
    task_lower = task.lower()
    
    # Find "vs" or " vs " separator
    separators = [" vs ", " vs.", " versus "]
    for sep in separators:
        if sep in task_lower:
            parts = task.split(sep)
            if len(parts) == 2:
                # Clean up team names (remove common words)
                home = parts[0].strip()
                away = parts[1].strip()
                
                # Remove common prefixes/suffixes
                for word in ["predict", "analyze", "analysis", "match", "game", "prediction"]:
                    home = home.replace(word, "").strip()
                    away = away.replace(word, "").strip()
                
                return home, away
    
    return "", ""
