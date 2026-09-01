# Evaluation Documentation

## Overview

The evaluation system measures both agent behavior and ML prediction quality. These are kept separate because a good prediction model does not imply a good agent, and vice versa.

## Benchmark Tasks

### Task Categories

1. **Specified Match Prediction** (spec_*): Direct match prediction requests
2. **Recent Form Analysis** (form_*): Team form analysis
3. **Model Comparison** (model_*): Comparing different ML models
4. **Recovery from Failed Retrieval** (recovery_*): Handling data unavailability
5. **Scenario Analysis** (scenario_*): What-if analysis
6. **Missing Data Handling** (missing_*): Proceeding with incomplete data
7. **Long Multi-Step Analysis** (long_*): Complex, multi-step tasks
8. **Sub-Agent Delegation** (delegate_*): Using specialized sub-agents
9. **Context Compression** (context_*): Summarizing large outputs
10. **Resuming Interrupted Task** (resume_*): Continuing previous work

### Task Difficulty Levels

- **Easy**: 1-2 tools, straightforward request
- **Medium**: 3-4 tools, may require aggregation
- **Hard**: 5+ tools, edge cases, or error recovery

## Agent Metrics

### Task Completion Rate
**Definition**: Percentage of tasks that produce a valid result
**Calculation**: `completed_tasks / total_tasks`
**Target**: > 90%

### Tool Success Rate
**Definition**: Percentage of tool calls that return valid results
**Calculation**: `successful_tool_calls / total_tool_calls`
**Target**: > 85%

### Recovery Rate
**Definition**: Percentage of failed tasks that succeed after retry/replanning
**Calculation**: `recovered_tasks / initially_failed_tasks`
**Target**: > 70%

### Planning Success Rate
**Definition**: Percentage of tasks where the plan correctly identifies required tools
**Calculation**: `tasks_with_correct_tool_sequence / total_tasks`
**Target**: > 80%

### Average Steps
**Definition**: Average number of steps per task
**Target**: Minimize while maintaining completion rate

### Average Execution Time
**Definition**: Mean time to complete a task
**Target**: < 30 seconds for simple tasks

## ML Metrics

### Classification Metrics

**Accuracy**: Overall correct predictions
```python
accuracy = correct_predictions / total_predictions
```

**Precision (macro)**: Average precision across H/D/A
```python
precision = avg(precision_per_class)
```

**Recall (macro)**: Average recall across H/D/A
```python
recall = avg(recall_per_class)
```

**F1 Score (macro)**: Harmonic mean of precision/recall
```python
f1 = 2 * (precision * recall) / (precision + recall)
```

### Probabilistic Metrics

**Log Loss**: Measures prediction confidence
```python
log_loss = -1/N * sum(y_true * log(y_pred))
```

**Brier Score**: Measures probability calibration
```python
brier = 1/N * sum((y_pred - y_true)^2)
```

## Evaluation Process

### Step 1: Run Benchmark
```python
from evaluation.evaluator import run_benchmark
results = run_benchmark(agent_run_fn)
```

### Step 2: Analyze Results
```python
metrics = results["metrics"]
print(f"Completion Rate: {metrics['task_completion_rate']:.1%}")
print(f"Average Duration: {metrics['average_duration']:.2f}s")
```

### Step 3: Compare Implementations
```python
from evaluation.evaluator import compare_agents
comparison = compare_agents(from_scratch_results, framework_results)
```

## Validation Rules

### Anti-Leakage Validation
- All features must be computable before match date
- Test with synthetic data to verify no future data leakage
- Time-based train/test splits (never random)

### Prediction Validation
- Probabilities must sum to ~1.0
- No negative probabilities
- No probabilities > 1.0

### Agent Validation
- All planned steps must complete or fail gracefully
- Tool calls must use correct schemas
- Results must include required output fields

## Framework Comparison

### From-Scratch Implementation
- Full control over agent loop
- Custom planning and memory
- Manual context management
- Direct LLM API integration

### Framework Implementation (LangGraph/LangChain)
- Pre-built agent primitives
- Built-in memory management
- Standardized tool schemas
- Community support

### Comparison Dimensions
1. **Development Time**: Hours to implement
2. **Flexibility**: Ability to customize behavior
3. **Memory Handling**: Persistence and recall capabilities
4. **Debugging Experience**: Ease of troubleshooting
5. **Performance**: Task completion and execution time
6. **Token Usage**: LLM API costs

## Reporting

### Evaluation Report Structure
1. Executive Summary
2. Task Completion Results
3. Tool Performance Analysis
4. Recovery and Error Handling
5. ML Model Comparison
6. Framework Comparison (Week 10)
7. Recommendations

### Success Criteria
- Task completion rate > 90%
- Tool success rate > 85%
- Recovery rate > 70%
- No data leakage in features
- Ensemble outperforms single models
