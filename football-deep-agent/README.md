# Football Deep Agent

Deep Agent-based football match analysis and prediction system.
A learning project: understand every core AI/ML and agent concept before
hiding it behind a framework.

## Team

| Person | Role | Owns |
|---|---|---|
| A | Agent & ML Engineering Lead | agent loop, tool calling, ML pipeline, model serving, error handling |
| B | Data & Cognition Lead | data collection, feature engineering, planning, memory, sub-agent delegation |

Roles rotate at the project midpoint. Architecture, reviews, testing,
evaluation, docs, and the final report are shared.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # fill in your LLM API key + data source
```

## Current phase

Start here: `docs/architecture.md` for the system diagram, then
`ml/README.md` for Phase 1 (build the prediction engine first, before
any agent code).

## Phase roadmap (10 weeks)

| Week | Focus | Deliverable |
|---|---|---|
| 0 | Foundations | Teach-back session (each person explains ML + agent concepts, no notes) |
| 1-2 | Football ML foundation | Working ML predictor (Home/Draw/Away probabilities) |
| 3-4 | Minimal Deep Agent (no framework) | CLI agent completing a 3-5 step analysis with 1 real tool |
| 5-6 | Planning, memory, self-correction | Agent completes 5-15 step task with a re-planning event |
| 7-8 | Tools, sub-agents, context management | Multi-agent analysis, robust long-running agent |
| 9 | Critic, scenarios, evaluation | Advanced system with guardrails |
| 10 | Framework comparison (LangGraph/LangChain) + final report | Final demo |

See `docs/architecture.md` for the full plan this scaffold was generated from.

## Project structure

```
football-deep-agent/
├── agent/            # from-scratch orchestration: manager, planner, memory, context, critic, state
├── agents/            # specialized sub-agents (team analysis, historical, prediction, scenario)
├── tools/             # tool functions the agent calls (match data, stats, form, prediction, simulation)
├── ml/                # prediction engine: preprocessing, features, models, ensemble, evaluation
├── data/              # raw/ processed/ schemas/
├── evaluation/        # benchmark tasks + evaluator for both agent and ML performance
├── framework_version/ # Week 10 LangGraph/LangChain comparison implementation
├── api/               # FastAPI app
├── frontend/          # React/Next.js UI (added after the agent works)
├── tests/
└── docs/
```

## Ground rule

This is an AI/ML learning project, not an app-dev project. Don't reach
for a library until you can explain what it would replace by hand.
