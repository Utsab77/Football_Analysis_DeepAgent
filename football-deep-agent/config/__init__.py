"""Configuration module for Football Deep Agent.

This module handles API key management and sub-agent configuration.
"""
from config.subagent_config import (
    SubAgentConfig,
    SubAgentRegistry,
    registry,
    get_subagent_api_key,
    get_all_subagent_keys,
)

__all__ = [
    "SubAgentConfig",
    "SubAgentRegistry",
    "registry",
    "get_subagent_api_key",
    "get_all_subagent_keys",
]
