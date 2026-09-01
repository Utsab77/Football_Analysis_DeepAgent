# Phase 0 - Foundations Study Notes

## Person B: Data & Cognition Lead

### 1. Feature Engineering for Football Prediction

**Key Concepts:**
- **Feature Engineering**: The process of using domain knowledge to create input variables (features) from raw data that improve machine learning model performance.
- **Pre-match Features**: All features must be computable BEFORE kickoff. This is the anti-leakage rule.
- **Time-based Features**: Recent form, rolling averages, and momentum indicators.

**Essential Football Features:**
1. **Team Strength**: Win rate over last N matches (placeholder for Elo/xG ratings)
2. **Recent Form**: Points-per-game over last 5 matches
3. **Goals Scored Average**: Average goals scored per match over last 10 games
4. **Goals Conceded Average**: Average goals conceded per match over last 10 games
5. **Home Advantage**: Binary indicator (1.0 for home team, 0.0 for away)
6. **League Position Difference**: Difference in league standings between teams

### 2. Time-Based Validation

**Why Random Splits Are Wrong:**
- Football matches are time-ordered events
- Random train/test splits leak future information into training
- The model would "see" future results when predicting past matches

**Correct Approach:**
- Train on earlier matches, test on later matches
- Use chronological splits (e.g., train on 2020-2023, test on 2024)
- Never shuffle matches across train/test boundaries

### 3. Data Leakage in Football Analytics

**Types of Leakage:**
1. **Temporal Leakage**: Using future match data to predict past matches
2. **Target Leakage**: Features that directly encode the match result
3. **Train-Test Contamination**: Same match appearing in both train and test sets

**Detection Methods:**
- Verify feature calculation dates are strictly before match dates
- Check that no future matches influence current feature values
- Test with synthetic data where future matches have different results

### 4. Football Analytics Fundamentals

**Home/Away Advantage:**
- Home teams win approximately 46% of matches in major leagues
- Factors: crowd support, familiarity with pitch, travel fatigue for away team
- Feature implementation: binary indicator (1.0 for home team)

**Recent Form:**
- Points per game over last 5 matches
- Captures current team momentum and confidence
- More responsive to recent changes than season-long statistics

**Head-to-Head Statistics:**
- Historical results between two specific teams
- Some teams have psychological advantages over others
- Useful for derbies and rivalries

**Goals Scored/Conceded:**
- Average goals scored and conceded over last 10 matches
- Offensive and defensive strength indicators
- More recent matches weighted higher for accuracy

**League Position:**
- Current standings reflect overall season performance
- Position difference indicates relative team strength
- Updated after each match week

### 5. Agent Context Engineering

**Context Management:**
- Summarize large tool outputs into compact representations
- Externalize detailed data, keep only relevant summaries in context
- Prevent context window overflow in LLM-based agents

**Memory Design:**
- **Working Memory**: Current analysis session data
- **Persistent Memory**: Previous analyses, tool results, completed tasks
- **Episodic Memory**: Successful/failed operation patterns

### 6. Planning and Task Decomposition

**Task Decomposition:**
- Break complex requests into ordered sub-tasks
- Each sub-task should be independently executable
- Dependencies between tasks must be explicit

**Planning Approaches:**
1. **Hardcoded Plans**: Pre-defined task sequences for known request types
2. **LLM-Generated Plans**: Dynamic decomposition based on request analysis
3. **Hybrid**: Template-based with LLM adaptation

**Re-planning Triggers:**
- Tool failure (data unavailable, computation error)
- Contradictory results (model predictions disagree significantly)
- Missing information (required data not found)
- Quality thresholds (confidence below minimum)

### 7. Sub-Agent Architecture

**Delegation Principles:**
- Each sub-agent owns a specific analysis domain
- Sub-agents receive compact context, not raw data
- Results are aggregated before final review

**Team Analysis Agent:**
- Focuses on single team's recent performance
- Returns: form, goals, strength, home/away splits

**Historical Analysis Agent:**
- Examines head-to-head records and trends
- Returns: historical patterns, notable results

**Scenario Agent:**
- Evaluates hypothetical changes (player absence, formation change)
- Returns: impact assessment, adjusted predictions

### 8. Repository Structure Understanding

**Project Layout:**
```
football-deep-agent/
├── agent/            # Orchestration: manager, planner, memory, context, critic
├── agents/           # Specialized sub-agents
├── tools/            # Tool functions the agent calls
├── ml/               # Prediction engine
├── data/             # Raw/processed data
├── evaluation/       # Benchmark tasks and evaluator
├── tests/            # Unit and integration tests
└── docs/             # Documentation
```

**Key Files for Person B:**
- `ml/preprocessing.py` - Data loading and cleaning
- `ml/features.py` - Feature engineering
- `tools/form.py` - Form and H2H tools
- `agent/planner.py` - Task planning
- `agent/memory.py` - Persistent memory
- `agents/*.py` - Sub-agent implementations
- `agent/context.py` - Context management

### Learning Objectives

By the end of Phase 0, I should be able to:
1. Explain why time-based validation is essential for football prediction
2. Identify and prevent data leakage in feature engineering
3. Describe the role of each football analytics feature
4. Explain how agent context management prevents LLM overflow
5. Design task decomposition for complex analysis requests
6. Articulate the difference between working and persistent memory
7. Justify sub-agent delegation as a context management strategy
