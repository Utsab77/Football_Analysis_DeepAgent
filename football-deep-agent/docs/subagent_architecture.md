# Sub-Agent Architecture - API Key Mapping

## Overview

The Football Deep Agent system uses **3 specialized sub-agents**, each with its own
API key for isolation, rate limiting, and cost tracking.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MANAGER AGENT (agent/manager.py)                    │
│                              API Key: Not needed (orchestration only)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │  TEAM ANALYSIS    │ │  HISTORICAL       │ │  SCENARIO         │
        │  SUB-AGENT        │ │  ANALYSIS         │ │  SUB-AGENT        │
        │                   │ │  SUB-AGENT        │ │                   │
        │  API Key #1       │ │  API Key #2       │ │  API Key #3       │
        │  (in .env file)   │ │  (in .env file)   │ │  (in .env file)   │
        │                   │ │                   │ │                   │
        │  Model:           │ │  Model:           │ │  Model:           │
        │  claude-3-haiku   │ │  claude-3-haiku   │ │  claude-3-sonnet  │
        │                   │ │                   │ │                   │
        │  Purpose:         │ │  Purpose:         │ │  Purpose:         │
        │  • Team form      │ │  • H2H records    │ │  • What-if        │
        │  • Strength       │ │  • Trends         │ │  • Player absence │
        │  • Home/away      │ │  • Notable games  │ │  • Formation      │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                        ┌─────────────────────────┐
                        │   RESULT AGGREGATOR     │
                        │   (agent/aggregator.py) │
                        │   No API key needed     │
                        └─────────────────────────┘
```

---

## Sub-Agent 1: Team Analysis Agent

**File**: `agents/team_analysis.py`
**API Key**: Set in `.env` as `SUB_AGENT_1_API_KEY`

### Purpose
Analyzes a single team's performance metrics:
- Recent form (points per game)
- Goals scored/conceded averages
- Strength rating
- League position
- Home/away performance splits

### Why This Key?
- **Fast responses**: Uses claude-3-haiku for quick analysis
- **Lower cost**: Team analysis is straightforward, doesn't need powerful model
- **High volume**: This agent runs most frequently (called for both teams)

### When It's Called
```
User: "Predict Arsenal vs Chelsea"
   ↓
Manager calls Team Analysis Agent TWICE:
   1. Analyze Arsenal (home team)
   2. Analyze Chelsea (away team)
   ↓
Each call uses Sub-Agent 1's API key
```

---

## Sub-Agent 2: Historical Analysis Agent

**File**: `agents/historical_analysis.py`
**API Key**: Set in `.env` as `SUB_AGENT_2_API_KEY`

### Purpose
Analyzes head-to-head records between two teams:
- Total matches played
- Win/draw/loss records
- Average goals per team
- Recent trends (who's dominant)
- Notable results (high-scoring games)

### Why This Key?
- **Separate rate limit**: H2H queries can be complex
- **Data isolation**: Historical data processing doesn't affect team analysis
- **Parallel execution**: Can run alongside team analysis

### When It's Called
```
User: "Predict Arsenal vs Chelsea"
   ↓
Manager calls Historical Analysis Agent ONCE:
   1. Get Arsenal vs Chelsea H2H record
   ↓
Uses Sub-Agent 2's API key
```

---

## Sub-Agent 3: Scenario Agent

**File**: `agents/scenario.py`
**API Key**: Set in `.env` as `SUB_AGENT_3_API_KEY`

### Purpose
Evaluates hypothetical changes and their impact:
- Key player absence (e.g., star striker injured)
- Goalkeeper absence (higher impact)
- Formation changes (defensive/attacking)
- Home crowd boost
- Fatigue from recent fixtures

### Why This Key?
- **More capable model**: Uses claude-3-sonnet for complex reasoning
- **Higher temperature**: Needs creative scenario generation
- **Separate tracking**: Scenario analysis is optional/premium feature

### When It's Called
```
User: "What if Haaland is injured for Man City vs Arsenal?"
   ↓
Manager calls Scenario Agent ONCE:
   1. Evaluate Haaland absence impact
   ↓
Uses Sub-Agent 3's API key
```

---

## Why 3 API Keys Are Enough

### 1. Rate Limiting Protection
```
Without separate keys:
- 100 requests/hour shared across all agents
- One agent could block others

With 3 separate keys:
- 100 requests/hour PER agent
- Total capacity: 300 requests/hour
- No cross-agent blocking
```

### 2. Cost Isolation
```
Track spending per agent type:
- Team Analysis: $X.XX (most frequent)
- Historical Analysis: $Y.YY (moderate)
- Scenario Analysis: $Z.ZZ (least frequent, highest quality)
```

### 3. Failure Isolation
```
If Sub-Agent 1 hits rate limit:
- Sub-Agent 2 and 3 still work
- System degrades gracefully
- Can retry with different agent
```

### 4. Parallel Execution
```
All 3 agents can run simultaneously:
- Team Analysis (both teams) → 2 parallel calls
- Historical Analysis → 1 call
- Scenario Analysis → 1 call (if requested)

Total: 4 parallel LLM calls possible
```

---

## Configuration

### .env File (NOT committed to git - see .gitignore)
```bash
# Sub-Agent 1: Team Analysis
SUB_AGENT_1_API_KEY=your-api-key-here

# Sub-Agent 2: Historical Analysis
SUB_AGENT_2_API_KEY=your-api-key-here

# Sub-Agent 3: Scenario Analysis
SUB_AGENT_3_API_KEY=your-api-key-here
```

### Usage in Code
```python
from config.subagent_config import get_subagent_api_key

# Get API key for specific agent
team_key = get_subagent_api_key("team_analysis")
historical_key = get_subagent_api_key("historical_analysis")
scenario_key = get_subagent_api_key("scenario")
```

---

## Model Selection Rationale

| Sub-Agent | Model | Why? |
|-----------|-------|------|
| Team Analysis | claude-3-haiku | Fast, cheap, straightforward analysis |
| Historical Analysis | claude-3-haiku | Fast, data lookup focused |
| Scenario Analysis | claude-3-sonnet | Needs reasoning for hypotheticals |

---

## Scaling Considerations

If you need more capacity later:

1. **Add more API keys**: Create additional OpenRouter accounts
2. **Load balance**: Distribute calls across multiple keys per agent
3. **Specialized models**: Use different models per agent type
4. **Caching**: Cache frequent queries to reduce API calls

---

## Security Notes

- API keys are stored in `.env` file (not committed to git)
- `.gitignore` includes `.env` to prevent accidental commits
- Never share API keys in documentation or code
- Use environment variables or secrets management in production

---

## Summary

**3 API keys are sufficient because:**
- Each key maps to one specialized sub-agent
- Sub-agents have distinct responsibilities
- Keys provide isolation and parallelism
- System can scale by adding more keys later

**File Locations:**
- Configuration: `config/subagent_config.py`
- Environment: `.env` (gitignored)
- Documentation: `docs/subagent_architecture.md`
