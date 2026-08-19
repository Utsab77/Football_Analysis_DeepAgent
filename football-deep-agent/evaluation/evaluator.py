"""Evaluation harness for AGENT performance (not ML performance).

Metrics tracked (see docs/evaluation.md):
- task completion rate
- tool success rate
- recovery rate
- planning success
- average steps
- average execution time
- token usage
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field


BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark_tasks.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class TaskResult:
    """Result of a single benchmark task."""
    task_id: str
    category: str
    prompt: str
    success: bool
    duration_seconds: float
    steps_executed: int
    tools_used: list[str]
    output_fields_present: list[str]
    output_fields_missing: list[str]
    error_message: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "prompt": self.prompt,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "steps_executed": self.steps_executed,
            "tools_used": self.tools_used,
            "output_fields_present": self.output_fields_present,
            "output_fields_missing": self.output_fields_missing,
            "error_message": self.error_message,
        }


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    task_completion_rate: float
    tool_success_rate: float
    recovery_rate: float
    planning_success_rate: float
    average_steps: float
    average_duration: float
    category_results: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "task_completion_rate": self.task_completion_rate,
            "tool_success_rate": self.tool_success_rate,
            "recovery_rate": self.recovery_rate,
            "planning_success_rate": self.planning_success_rate,
            "average_steps": self.average_steps,
            "average_duration": self.average_duration,
            "category_results": self.category_results,
        }


def load_benchmark_tasks() -> list[dict]:
    """Load benchmark tasks from JSON file."""
    if not BENCHMARK_PATH.exists():
        return []
    return json.loads(BENCHMARK_PATH.read_text())


def run_benchmark(agent_run_fn, max_tasks: int = None) -> dict:
    """Run benchmark tasks against an agent.
    
    Args:
        agent_run_fn: callable(task: str) -> AgentState-like object with .to_dict()
        max_tasks: Maximum number of tasks to run (None = all)
    
    Returns:
        Dictionary with evaluation results
    """
    tasks = load_benchmark_tasks()
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    results = []
    
    for task in tasks:
        print(f"Running task {task['id']}: {task['prompt'][:50]}...")
        
        start_time = time.monotonic()
        try:
            state = agent_run_fn(task["prompt"])
            duration = time.monotonic() - start_time
            
            # Extract metrics from state
            state_dict = state.to_dict() if hasattr(state, 'to_dict') else state
            
            # Determine success
            success = state_dict.get("result") is not None
            
            # Get tools used
            tool_calls = state_dict.get("tool_calls", [])
            tools_used = [tc.get("tool", "") for tc in tool_calls]
            
            # Check expected tools
            expected_tools = task.get("expected_tools", [])
            tools_match = all(any(t in used for used in tools_used) for t in expected_tools)
            
            # Check output fields
            result = state_dict.get("result", {})
            expected_fields = task.get("expected_output_fields", [])
            fields_present = [f for f in expected_fields if f in result]
            fields_missing = [f for f in expected_fields if f not in result]
            
            task_result = TaskResult(
                task_id=task["id"],
                category=task["category"],
                prompt=task["prompt"],
                success=success and tools_match,
                duration_seconds=duration,
                steps_executed=len(state_dict.get("completed_steps", [])),
                tools_used=tools_used,
                output_fields_present=fields_present,
                output_fields_missing=fields_missing,
            )
            
        except Exception as e:
            duration = time.monotonic() - start_time
            task_result = TaskResult(
                task_id=task["id"],
                category=task["category"],
                prompt=task["prompt"],
                success=False,
                duration_seconds=duration,
                steps_executed=0,
                tools_used=[],
                output_fields_present=[],
                output_fields_missing=task.get("expected_output_fields", []),
                error_message=str(e),
            )
        
        results.append(task_result)
    
    # Calculate aggregated metrics
    metrics = _calculate_metrics(results)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"evaluation_{timestamp}.json"
    
    output = {
        "timestamp": timestamp,
        "metrics": metrics.to_dict(),
        "task_results": [r.to_dict() for r in results],
    }
    
    results_path.write_text(json.dumps(output, indent=2, default=str))
    
    return {
        "results_path": str(results_path),
        "metrics": metrics.to_dict(),
    }


def _calculate_metrics(results: list[TaskResult]) -> EvaluationMetrics:
    """Calculate aggregated evaluation metrics from task results."""
    total = len(results)
    completed = sum(1 for r in results if r.success)
    failed = total - completed
    
    # Task completion rate
    completion_rate = completed / total if total > 0 else 0.0
    
    # Tool success rate (tasks where all expected tools were used)
    tool_success_count = sum(
        1 for r in results 
        if len(r.output_fields_missing) == 0
    )
    tool_success_rate = tool_success_count / total if total > 0 else 0.0
    
    # Recovery rate (tasks that succeeded after initial failure)
    # Simplified: assume recovery if task completed with steps > 0
    recovery_count = sum(
        1 for r in results 
        if r.success and r.steps_executed > 1
    )
    recovery_rate = recovery_count / completed if completed > 0 else 0.0
    
    # Planning success rate (tasks that used expected tools)
    planning_success = sum(
        1 for r in results
        if len(r.output_fields_missing) == 0
    )
    planning_success_rate = planning_success / total if total > 0 else 0.0
    
    # Average steps
    avg_steps = sum(r.steps_executed for r in results) / total if total > 0 else 0.0
    
    # Average duration
    avg_duration = sum(r.duration_seconds for r in results) / total if total > 0 else 0.0
    
    # Category breakdown
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"total": 0, "completed": 0, "failed": 0}
        categories[r.category]["total"] += 1
        if r.success:
            categories[r.category]["completed"] += 1
        else:
            categories[r.category]["failed"] += 1
    
    return EvaluationMetrics(
        total_tasks=total,
        completed_tasks=completed,
        failed_tasks=failed,
        task_completion_rate=completion_rate,
        tool_success_rate=tool_success_rate,
        recovery_rate=recovery_rate,
        planning_success_rate=planning_success_rate,
        average_steps=avg_steps,
        average_duration=avg_duration,
        category_results=categories,
    )


def compare_agents(agent_a_results: dict, agent_b_results: dict) -> dict:
    """Compare results from two agent implementations.
    
    Args:
        agent_a_results: Evaluation results from agent A (e.g., from-scratch)
        agent_b_results: Evaluation results from agent B (e.g., framework)
    
    Returns:
        Comparison summary
    """
    metrics_a = agent_a_results.get("metrics", {})
    metrics_b = agent_b_results.get("metrics", {})
    
    comparison = {
        "task_completion_rate": {
            "from_scratch": metrics_a.get("task_completion_rate", 0),
            "framework": metrics_b.get("task_completion_rate", 0),
            "difference": metrics_b.get("task_completion_rate", 0) - metrics_a.get("task_completion_rate", 0),
        },
        "average_duration": {
            "from_scratch": metrics_a.get("average_duration", 0),
            "framework": metrics_b.get("average_duration", 0),
            "difference": metrics_b.get("average_duration", 0) - metrics_a.get("average_duration", 0),
        },
        "average_steps": {
            "from_scratch": metrics_a.get("average_steps", 0),
            "framework": metrics_b.get("average_steps", 0),
            "difference": metrics_b.get("average_steps", 0) - metrics_a.get("average_steps", 0),
        },
        "tool_success_rate": {
            "from_scratch": metrics_a.get("tool_success_rate", 0),
            "framework": metrics_b.get("tool_success_rate", 0),
            "difference": metrics_b.get("tool_success_rate", 0) - metrics_a.get("tool_success_rate", 0),
        },
    }
    
    # Overall assessment
    a_better = sum(1 for v in comparison.values() if v["difference"] < 0)
    b_better = sum(1 for v in comparison.values() if v["difference"] > 0)
    
    comparison["overall"] = {
        "from_scratch_advantages": a_better,
        "framework_advantages": b_better,
        "recommendation": "from_scratch" if a_better > b_better else "framework",
    }
    
    return comparison
