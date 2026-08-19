"""Manager agent: the top-level loop -- perceive -> plan -> act -> observe -> repeat.

Phase 2 (Weeks 3-4): build this WITHOUT a framework first. Wire in an
LLM API directly for the planning/reasoning calls.

Phase 4+: Integrate sub-agents, context management, and result aggregation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.state import AgentState
from agent.planner import make_plan, AnalysisPlan, StepStatus, replan_on_failure, extract_teams_from_task
from agent.memory import WorkingMemory, get_working_memory, get_persistent_memory
from agent.context import compress, build_context_window, summarize_for_agent
from agent.aggregator import aggregate_results, validate_aggregation
from agent.critic import review

from tools.team_stats import get_team_stats
from tools.form import get_recent_form, get_head_to_head
from tools.league_position import get_league_position
from tools.prediction import run_prediction_model


# Tool registry
TOOL_REGISTRY = {
    "get_team_stats": get_team_stats,
    "get_recent_form": get_recent_form,
    "get_head_to_head": get_head_to_head,
    "get_league_position": get_league_position,
    "run_prediction_model": run_prediction_model,
}

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "execution_logs"


def run(task: str, tools: dict = None) -> AgentState:
    """Run one end-to-end agent task.
    
    Args:
        task: The analysis task to perform
        tools: Optional dict of {tool_name: callable} to use
    
    Returns:
        AgentState with complete analysis results
    """
    # Use provided tools or default registry
    available_tools = tools or TOOL_REGISTRY
    
    # Initialize state
    state = AgentState(task=task)
    
    # Initialize memory
    working_memory = get_working_memory()
    working_memory.clear()
    
    # Create plan
    state.plan = [step.description for step in make_plan(task).steps]
    plan = make_plan(task)
    
    # Extract teams for context
    home_team, away_team = extract_teams_from_task(task)
    
    # Execute plan steps
    max_retries = 3
    retry_count = 0
    
    while True:
        step = plan.get_next_step()
        if step is None:
            break
        
        # Execute step
        step.status = StepStatus.IN_PROGRESS
        result = _execute_step(step, home_team, away_team, working_memory)
        
        if result["success"]:
            plan.mark_step_complete(step, result["data"])
            working_memory.store_tool_result(
                step.tool_name or "unknown",
                step.tool_args,
                result["data"]
            )
            state.completed_steps.append(step.description)
            retry_count = 0
        else:
            retry_count += 1
            if retry_count <= max_retries:
                # Try re-planning
                plan = replan_on_failure(plan, step, result["error"])
                state.log_tool_call(
                    step.tool_name or "unknown",
                    step.tool_args,
                    {"error": result["error"], "retry": True}
                )
            else:
                plan.mark_step_failed(step, result["error"])
                state.log_tool_call(
                    step.tool_name or "unknown",
                    step.tool_args,
                    {"error": result["error"], "failed": True}
                )
    
    # Aggregate results
    aggregated = _aggregate_step_results(task, home_team, away_team, plan, working_memory)
    
    # Run critic
    state.result = aggregated.to_dict()
    verdict = review(state)
    
    if not verdict.passed:
        # Log critic feedback
        state.log_tool_call("critic", {}, {"reasons": verdict.reasons})
    
    # Record to persistent memory
    persistent_memory = get_persistent_memory()
    persistent_memory.record_analysis(
        task=task,
        result=aggregated.to_dict(),
        teams=[home_team, away_team] if home_team and away_team else []
    )
    
    # Save execution log
    _save_execution_log(state)
    
    return state


def _execute_step(
    step, 
    home_team: str, 
    away_team: str, 
    working_memory: WorkingMemory
) -> dict:
    """Execute a single plan step."""
    try:
        # Map step descriptions to tool calls
        tool_name = step.tool_name
        args = step.tool_args.copy()
        
        # Add default arguments based on step type
        if tool_name == "get_team_stats":
            # Alternate between teams
            if "team" not in args:
                step_num = len([s for s in step.description.split() if "team" in s.lower()])
                args["team"] = home_team if step_num % 2 == 0 else away_team
            if "before_date" not in args:
                args["before_date"] = datetime.now(timezone.utc).isoformat()
        
        elif tool_name == "get_recent_form":
            if "team" not in args:
                args["team"] = home_team
            if "before_date" not in args:
                args["before_date"] = datetime.now(timezone.utc).isoformat()
        
        elif tool_name == "get_head_to_head":
            args["team_a"] = home_team
            args["team_b"] = away_team
            if "before_date" not in args:
                args["before_date"] = datetime.now(timezone.utc).isoformat()
        
        elif tool_name == "get_league_position":
            if "team" not in args:
                args["team"] = home_team
            if "before_date" not in args:
                args["before_date"] = datetime.now(timezone.utc).isoformat()
        
        elif tool_name == "run_prediction_model":
            args["home_team"] = home_team
            args["away_team"] = away_team
        
        # Call the tool
        if tool_name in TOOL_REGISTRY:
            result = TOOL_REGISTRY[tool_name](**args)
            return {"success": True, "data": result}
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _aggregate_step_results(
    task: str,
    home_team: str,
    away_team: str,
    plan: AnalysisPlan,
    working_memory: WorkingMemory,
):
    """Aggregate results from completed steps."""
    # Gather results from working memory
    team_home_analysis = None
    team_away_analysis = None
    h2h_analysis = None
    prediction = None
    
    for key, stored in working_memory.tool_results.items():
        tool_name = stored["tool"]
        result = stored["result"]
        
        if tool_name == "get_team_stats":
            if stored["args"].get("team") == home_team:
                team_home_analysis = result
            else:
                team_away_analysis = result
        
        elif tool_name == "get_head_to_head":
            h2h_analysis = result
        
        elif tool_name == "run_prediction_model":
            prediction = result
    
    # Default prediction if none found
    if prediction is None:
        prediction = {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}
    
    return aggregate_results(
        task=task,
        home_team=home_team,
        away_team=away_team,
        prediction=prediction,
        team_analysis_home=team_home_analysis,
        team_analysis_away=team_away_analysis,
        historical_analysis=h2h_analysis,
    )


def _save_execution_log(state: AgentState):
    """Save execution log to disk."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"execution_{timestamp}.json"
    
    log_data = state.to_dict()
    log_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    log_path.write_text(json.dumps(log_data, indent=2, default=str))
