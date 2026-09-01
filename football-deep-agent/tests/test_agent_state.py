from agent.state import AgentState


def test_state_logs_tool_calls():
    state = AgentState(task="Analyze Arsenal vs Chelsea")
    state.log_tool_call("get_team_stats", {"team": "Arsenal"}, {"strength": 0.7})
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0]["tool"] == "get_team_stats"


def test_state_serializes():
    state = AgentState(task="test")
    d = state.to_dict()
    assert d["task"] == "test"
    assert "started_at" in d
