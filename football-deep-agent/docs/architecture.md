# Architecture

```
USER
  │
  ▼
Manager Agent
  │
  ▼
PLANNER -- creates analysis plan
  │
  ┌───────────────┼───────────────┐
  ▼               ▼               ▼
Football Data   Team Analysis   Prediction
Tools           Agent           Agent
  │               │               │
  ▼               ▼               ▼
Historical Data  Form/H2H/       ML Models
Current Data     Home-Away
Player Data      Analysis
  │               │               │
  └───────────────┼───────────────┘
                  ▼
          RESULT AGGREGATOR
                  │
                  ▼
            CRITIC AGENT
             ┌────┴────┐
             ▼         ▼
          Failure     Pass
             │          │
             ▼          ▼
          RE-PLAN   FINAL RESULT
             │
             └────────► Agent

Supporting services: Memory | Context Manager | Persistent State | Guardrails | Evaluation
```

## Principle

The agent is the reasoning/orchestration layer. It does not guess match
outcomes itself -- statistical/ML models do the actual prediction. The
agent decides what information is needed, retrieves it, runs the models,
critiques the result, and re-plans on failure.

## Simple vs Deep Agent (Week 10 experiment)

- **Simple**: User → LLM → Tools → ML Model → Answer
- **Deep**: User → Planner → Memory → Multiple Tools → Sub-Agents → ML Models → Critic → Re-planning → Final Answer

Run the same benchmark against both and compare task completion, recovery,
long-task performance, cost, execution time, and prediction quality.
