# Agent Documentation

## Overview

The Football Deep Agent system uses a multi-agent architecture with specialized sub-agents for different analysis tasks. The system follows a perceive-plan-act-observe loop with re-planning capabilities.

## Architecture

### Manager Agent (agent/manager.py)
The top-level orchestrator that:
- Receives user requests
- Creates analysis plans
- Delegates to sub-agents
- Aggregates results
- Handles re-planning on failures

### Planner (agent/planner.py)
Task decomposition module that:
- Converts user requests into ordered step lists
- Supports hardcoded templates for common tasks
- Enables re-planning when steps fail
- Extracts team names from natural language

### Memory System (agent/memory.py)
Two-tier memory architecture:

**Working Memory**: In-session storage for current analysis
- Tool results
- Step outputs
- Context variables

**Persistent Memory**: Long-term storage on disk
- Previous analyses
- Team information
- Failed operations

### Context Manager (agent/context.py)
Manages context window for LLM calls:
- Compresses large tool outputs
- Summarizes structured data
- Prevents context overflow

### Critic Agent (agent/critic.py)
Reviews results before final output:
- Validates data completeness
- Checks for contradictions
- Verifies model reliability
- Returns pass/fail verdict

### Result Aggregator (agent/aggregator.py)
Combines outputs from multiple sub-agents:
- Team Analysis results
- Historical Analysis results
- Prediction results
- Scenario Analysis results

## Sub-Agents

### Team Analysis Agent (agents/team_analysis.py)
Analyzes a single team's performance:
- Recent form (points per game)
- Goals scored/conceded averages
- Strength rating
- League position
- Home/away performance splits

**Input**: `{team: str, before_date: str, n_matches: int}`
**Output**: `{team, form_points_per_game, goals_scored_avg, ...}`

### Historical Analysis Agent (agents/historical_analysis.py)
Analyzes head-to-head records:
- Total matches played
- Win/draw/loss records
- Average goals per team
- Recent trends
- Notable results

**Input**: `{team_a: str, team_b: str, before_date: str, n_matches: int}`
**Output**: `{team_a, team_b, total_matches, ...}`

### Scenario Agent (agents/scenario.py)
Evaluates hypothetical changes:
- Player absence impact
- Formation changes
- Home crowd boost
- Fatigue effects

**Input**: `{home_team, away_team, match_date, scenario_type, custom_impact}`
**Output**: `{base_prediction, adjusted_prediction, impact_analysis, ...}`

## Tool Schemas

### get_team_stats
```json
{
  "name": "get_team_stats",
  "description": "Get summary statistics for a team as of a given date",
  "parameters": {
    "team": {"type": "string", "description": "Team name"},
    "before_date": {"type": "string", "description": "ISO date string"}
  },
  "returns": {
    "team": "string",
    "strength": "float (0-1)",
    "goals_avg": "float",
    "conceded_avg": "float"
  }
}
```

### get_recent_form
```json
{
  "name": "get_recent_form",
  "description": "Get recent form (points per game) for a team",
  "parameters": {
    "team": {"type": "string", "description": "Team name"},
    "before_date": {"type": "string", "description": "ISO date string"},
    "n": {"type": "integer", "description": "Number of matches (default: 5)"}
  },
  "returns": {
    "team": "string",
    "form_points_per_game": "float"
  }
}
```

### get_head_to_head
```json
{
  "name": "get_head_to_head",
  "description": "Get head-to-head match history between two teams",
  "parameters": {
    "team_a": {"type": "string", "description": "First team name"},
    "team_b": {"type": "string", "description": "Second team name"},
    "before_date": {"type": "string", "description": "ISO date string"},
    "n": {"type": "integer", "description": "Number of matches (default: 10)"}
  },
  "returns": "list of match records"
}
```

### get_league_position
```json
{
  "name": "get_league_position",
  "description": "Get team's league position as of a given date",
  "parameters": {
    "team": {"type": "string", "description": "Team name"},
    "before_date": {"type": "string", "description": "ISO date string"}
  },
  "returns": {
    "team": "string",
    "position": "integer (1 = top)"
  }
}
```

### run_prediction_model
```json
{
  "name": "run_prediction_model",
  "description": "Run ML model to predict match outcome",
  "parameters": {
    "home_team": {"type": "string", "description": "Home team name"},
    "away_team": {"type": "string", "description": "Away team name"},
    "match_date": {"type": "string", "description": "ISO date string (optional)"}
  },
  "returns": {
    "home_win": "float (probability)",
    "draw": "float (probability)",
    "away_win": "float (probability)"
  }
}
```

## Re-Planning Triggers

The system triggers re-planning when:

1. **Tool Failure**: Data unavailable, computation error
2. **Missing Information**: Required data not found
3. **Quality Threshold**: Confidence below minimum
4. **Contradiction**: Model predictions disagree significantly

## Memory Design

### Working Memory
- Clears at start of each analysis
- Stores tool results for current session
- Enables context building for LLM calls

### Persistent Memory
- JSON file-backed storage
- Records all completed analyses
- Stores team information for quick lookup
- Tracks failed operations for learning

## Sub-Agent Boundaries

Each sub-agent owns a specific analysis domain:
- **Team Analysis**: Single team performance metrics
- **Historical Analysis**: H2H records and trends
- **Scenario**: Hypothetical change impacts

Sub-agents receive compact context, not raw data.
Results are aggregated before final review.

## Context Compression

Rule-based compression for structured data:
- Team stats → brief summary
- Form → points per game
- H2H → match count and trend
- Predictions → probability breakdown

## Evaluation Metrics

Agent performance measured by:
- Task completion rate
- Tool success rate
- Recovery rate
- Planning success
- Average steps
- Execution time
- Token usage
