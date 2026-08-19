"""Evaluation harness for AGENT performance (not ML performance --
see ml/evaluation.py for that, kept separate on purpose).

Metrics tracked (see docs/evaluation.md):
    task completion rate, tool success rate, recovery rate,
    planning success, average steps, average execution time, token usage
"""
from __future__ import annotations

import json
import time
from pathlib import Path

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark_tasks.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_benchmark_tasks() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        return []
    return json.loads(BENCHMARK_PATH.read_text())


def run_benchmark(agent_run_fn) -> dict:
    """`agent_run_fn`: callable(task: str) -> AgentState-like object with .to_dict()."""
    tasks = load_benchmark_tasks()
    results = []
    for task in tasks:
        start = time.monotonic()
        state = agent_run_fn(task["prompt"])
        duration = time.monotonic() - start
        results.append({
            "task": task["prompt"],
            "duration_seconds": duration,
            "steps": len(state.completed_steps),
            "tool_calls": len(state.tool_calls),
            "result": state.result,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"run_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    return {"results_path": str(out_path), "n_tasks": len(tasks)}
