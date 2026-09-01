"""Context management: summarize/externalize large tool outputs so the
agent receives compact, relevant context instead of raw dumps.

Phase 4 (Weeks 7-8):
- Rule-based summarization for structured data
- LLM-based summarization for complex outputs
- Context window management
"""
from __future__ import annotations

import json
from typing import Any


MAX_INLINE_CHARS = 2000
MAX_CONTEXT_TOKENS = 4000  # Approximate token limit for context window


def compress(tool_name: str, raw_output: Any) -> str:
    """Return a compact summary if output is large; otherwise pass through.
    
    Args:
        tool_name: Name of the tool that produced the output
        raw_output: The raw output to compress
        
    Returns:
        Compressed string representation
    """
    # Convert to string if not already
    if isinstance(raw_output, dict):
        # Try to find a summary field first
        if "summary" in raw_output:
            return raw_output["summary"]
        
        # Use rule-based compression for known tool outputs
        compressed = _compress_structured_output(tool_name, raw_output)
        if compressed:
            return compressed
        
        # Fall back to JSON with truncation
        output_str = json.dumps(raw_output, indent=2, default=str)
    elif isinstance(raw_output, list):
        output_str = json.dumps(raw_output, indent=2, default=str)
    else:
        output_str = str(raw_output)
    
    if len(output_str) <= MAX_INLINE_CHARS:
        return output_str
    
    return output_str[:MAX_INLINE_CHARS] + f"... [truncated, full output stored under '{tool_name}']"


def _compress_structured_output(tool_name: str, data: dict) -> str | None:
    """Apply tool-specific compression rules."""
    
    if tool_name == "get_team_stats":
        return _compress_team_stats(data)
    elif tool_name == "get_recent_form":
        return _compress_form(data)
    elif tool_name == "get_head_to_head":
        return _compress_h2h(data)
    elif tool_name == "get_league_position":
        return _compress_league_position(data)
    elif tool_name == "team_analysis":
        return _compress_team_analysis(data)
    elif tool_name == "historical_analysis":
        return _compress_historical_analysis(data)
    elif tool_name == "scenario":
        return _compress_scenario(data)
    elif tool_name == "run_prediction_model":
        return _compress_prediction(data)
    
    return None


def _compress_team_stats(data: dict) -> str:
    """Compress team stats output."""
    return (
        f"{data.get('team', 'Unknown')}: "
        f"Strength {data.get('strength', 0):.2f}, "
        f"Goals/Game {data.get('goals_avg', 0):.1f}, "
        f"Conceded/Game {data.get('conceded_avg', 0):.1f}"
    )


def _compress_form(data: dict) -> str:
    """Compress form output."""
    return (
        f"{data.get('team', 'Unknown')} form: "
        f"{data.get('form_points_per_game', 0):.1f} PPG"
    )


def _compress_h2h(data: list) -> str:
    """Compress head-to-head output."""
    if not data:
        return "No H2H data available"
    
    total = len(data)
    recent = data[-3:] if len(data) > 3 else data
    dates = [m.get("date", "?") for m in recent]
    return f"H2H: {total} matches found, last on {', '.join(dates)}"


def _compress_league_position(data: dict) -> str:
    """Compress league position output."""
    return (
        f"{data.get('team', 'Unknown')}: "
        f"Position {data.get('position', '?')} "
        f"(as of {data.get('before_date', '?')})"
    )


def _compress_team_analysis(data: dict) -> str:
    """Compress team analysis sub-agent output."""
    return data.get("summary", "Team analysis completed")


def _compress_historical_analysis(data: dict) -> str:
    """Compress historical analysis sub-agent output."""
    return data.get("summary", "Historical analysis completed")


def _compress_scenario(data: dict) -> str:
    """Compress scenario analysis sub-agent output."""
    return data.get("summary", "Scenario analysis completed")


def _compress_prediction(data: dict) -> str:
    """Compress prediction output."""
    home = data.get("home_win", 0)
    draw = data.get("draw", 0)
    away = data.get("away_win", 0)
    return f"Prediction: H:{home:.1%} D:{draw:.1%} A:{away:.1%}"


def build_context_window(
    tool_results: list[dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """Build a context window from multiple tool results.
    
    Args:
        tool_results: List of {tool_name, result} dicts
        max_tokens: Approximate token limit
        
    Returns:
        Formatted context string
    """
    context_parts = []
    current_length = 0
    
    for entry in tool_results:
        tool_name = entry.get("tool_name", "unknown")
        result = entry.get("result", {})
        
        compressed = compress(tool_name, result)
        
        # Estimate token count (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(compressed) // 4
        
        if current_length + estimated_tokens > max_tokens:
            # Truncate to fit
            remaining_chars = (max_tokens - current_length) * 4
            if remaining_chars > 100:
                context_parts.append(compressed[:remaining_chars] + "...")
            break
        
        context_parts.append(f"[{tool_name}]\n{compressed}")
        current_length += estimated_tokens
    
    return "\n\n".join(context_parts)


def summarize_for_agent(
    full_context: str,
    task: str,
    max_length: int = 1000,
) -> str:
    """Summarize context specifically for agent consumption.
    
    This is a rule-based summarization. Could be enhanced with LLM calls.
    """
    if len(full_context) <= max_length:
        return full_context
    
    # Extract key sections
    lines = full_context.split("\n")
    important_lines = []
    
    for line in lines:
        # Keep lines with key metrics or summaries
        if any(keyword in line.lower() for keyword in [
            "prediction", "form", "strength", "position", "won", "lost",
            "draw", "goals", "summary", "analysis"
        ]):
            important_lines.append(line)
    
    summarized = "\n".join(important_lines)
    
    if len(summarized) > max_length:
        return summarized[:max_length] + "..."
    
    return summarized
