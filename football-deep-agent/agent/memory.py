"""Persistent memory: previous analyses, tool results, completed/failed tasks.

Phase 3 (Weeks 5-6): 
- JSON-file-backed store for persistence
- Working memory for current session
- Memory recall for previous analyses
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "agent_memory.json"


class WorkingMemory:
    """In-session memory for current analysis."""
    
    def __init__(self):
        self.tool_results: dict = {}
        self.context: dict = {}
        self.step_outputs: list[dict] = []
    
    def store_tool_result(self, tool_name: str, args: dict, result: dict):
        """Store result from a tool call."""
        key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        self.tool_results[key] = {
            "tool": tool_name,
            "args": args,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def get_tool_result(self, tool_name: str, args: dict) -> dict | None:
        """Retrieve a previously stored tool result."""
        key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        return self.tool_results.get(key)
    
    def store_step_output(self, step: str, output: dict):
        """Store output from a completed step."""
        self.step_outputs.append({
            "step": step,
            "output": output,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def get_context(self, key: str) -> dict | None:
        """Get context value by key."""
        return self.context.get(key)
    
    def set_context(self, key: str, value: dict):
        """Set context value."""
        self.context[key] = value
    
    def clear(self):
        """Clear working memory."""
        self.tool_results.clear()
        self.context.clear()
        self.step_outputs.clear()


class PersistentMemory:
    """Long-term memory stored on disk."""
    
    def __init__(self):
        self._data = self._load()
    
    def _load(self) -> dict:
        """Load memory from disk."""
        if not MEMORY_PATH.exists():
            return {
                "analyses": [],
                "tool_results": [],
                "failed_operations": [],
                "team_info": {},
                "match_history": {},
            }
        return json.loads(MEMORY_PATH.read_text())
    
    def save(self):
        """Save memory to disk."""
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(json.dumps(self._data, indent=2, default=str))
    
    def record_analysis(self, task: str, result: dict, teams: list[str] = None):
        """Record a completed analysis."""
        entry = {
            "task": task,
            "result": result,
            "teams": teams or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._data["analyses"].append(entry)
        
        # Also store team info for quick lookup
        if teams:
            for team in teams:
                if team not in self._data["team_info"]:
                    self._data["team_info"][team] = []
                self._data["team_info"][team].append({
                    "task": task,
                    "timestamp": entry["timestamp"],
                })
        
        self.save()
    
    def record_tool_result(self, tool_name: str, args: dict, result: dict, success: bool = True):
        """Record a tool call result."""
        entry = {
            "tool": tool_name,
            "args": args,
            "result": result,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if success:
            self._data["tool_results"].append(entry)
        else:
            self._data["failed_operations"].append(entry)
        
        self.save()
    
    def find_previous_analysis(self, teams: list[str]) -> dict | None:
        """Find a previous analysis involving the same teams."""
        teams_set = set(teams)
        for analysis in reversed(self._data["analyses"]):
            if set(analysis.get("teams", [])) == teams_set:
                return analysis
        return None
    
    def get_team_history(self, team: str) -> list[dict]:
        """Get analysis history for a specific team."""
        return self._data["team_info"].get(team, [])
    
    def get_recent_analyses(self, n: int = 5) -> list[dict]:
        """Get the n most recent analyses."""
        return self._data["analyses"][-n:]
    
    def get_failed_operations(self) -> list[dict]:
        """Get all failed operations."""
        return self._data["failed_operations"]
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than specified days."""
        cutoff = datetime.now(timezone.utc).isoformat()
        # Simple implementation - could be enhanced with proper date parsing
        pass


# Module-level instances for convenience
_working_memory = WorkingMemory()
_persistent_memory = PersistentMemory()


def get_working_memory() -> WorkingMemory:
    """Get the working memory instance."""
    return _working_memory


def get_persistent_memory() -> PersistentMemory:
    """Get the persistent memory instance."""
    return _persistent_memory


def load_memory() -> dict:
    """Load persistent memory (legacy interface)."""
    return _persistent_memory._data


def save_memory(memory: dict) -> None:
    """Save persistent memory (legacy interface)."""
    _persistent_memory._data = memory
    _persistent_memory.save()


def record_analysis(memory: dict, task: str, result: dict) -> dict:
    """Record analysis (legacy interface)."""
    _persistent_memory.record_analysis(task, result)
    return _persistent_memory._data
