"""Sub-Agent API Configuration.

Maps 3 API keys to 3 specialized sub-agents for the Football Deep Agent system.

WHY 3 SUB-AGENTS:
1. Team Analysis Agent - Analyzes individual team performance
2. Historical Analysis Agent - Analyzes head-to-head records
3. Scenario Agent - Evaluates hypothetical changes

WHY 3 API KEYS:
- Rate limiting: Each agent has its own quota
- Isolation: One agent's failure doesn't affect others
- Parallelism: All 3 can run simultaneously
- Cost tracking: Monitor usage per agent type
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)


@dataclass
class SubAgentConfig:
    """Configuration for a single sub-agent."""
    name: str
    api_key: str
    provider: str
    model: str
    max_tokens: int
    temperature: float


class SubAgentRegistry:
    """Registry that maps API keys to sub-agents."""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openrouter")
        self.agents = self._initialize_agents()
    
    def _initialize_agents(self) -> dict[str, SubAgentConfig]:
        """Initialize all 3 sub-agents with their API keys."""
        
        # Sub-Agent 1: Team Analysis
        team_agent = SubAgentConfig(
            name="team_analysis",
            api_key=os.getenv("SUB_AGENT_1_API_KEY", ""),
            provider=self.provider,
            model="anthropic/claude-3-haiku",  # Fast, cost-effective for analysis
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for consistent analysis
        )
        
        # Sub-Agent 2: Historical Analysis
        historical_agent = SubAgentConfig(
            name="historical_analysis",
            api_key=os.getenv("SUB_AGENT_2_API_KEY", ""),
            provider=self.provider,
            model="anthropic/claude-3-haiku",
            max_tokens=2000,
            temperature=0.3,
        )
        
        # Sub-Agent 3: Scenario Analysis
        scenario_agent = SubAgentConfig(
            name="scenario",
            api_key=os.getenv("SUB_AGENT_3_API_KEY", ""),
            provider=self.provider,
            model="anthropic/claude-3-sonnet",  # More capable for complex scenarios
            max_tokens=3000,
            temperature=0.5,  # Slightly higher for creative scenario generation
        )
        
        return {
            "team_analysis": team_agent,
            "historical_analysis": historical_agent,
            "scenario": scenario_agent,
        }
    
    def get_agent(self, agent_name: str) -> SubAgentConfig:
        """Get configuration for a specific sub-agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}")
        return self.agents[agent_name]
    
    def get_api_key(self, agent_name: str) -> str:
        """Get API key for a specific sub-agent."""
        return self.get_agent(agent_name).api_key
    
    def get_all_agents(self) -> dict[str, SubAgentConfig]:
        """Get all sub-agent configurations."""
        return self.agents.copy()
    
    def validate_keys(self) -> dict[str, bool]:
        """Validate that all API keys are set."""
        return {
            name: bool(agent.api_key)
            for name, agent in self.agents.items()
        }


# Module-level instance for easy access
registry = SubAgentRegistry()


def get_subagent_api_key(agent_name: str) -> str:
    """Convenience function to get API key for a sub-agent."""
    return registry.get_api_key(agent_name)


def get_all_subagent_keys() -> dict[str, str]:
    """Get all sub-agent API keys."""
    return {name: agent.api_key for name, agent in registry.get_all_agents().items()}
