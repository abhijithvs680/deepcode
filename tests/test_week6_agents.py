import pytest
from app.agents.planner_agent import PlannerAgent
from app.core.debug_trace import DebugTrace

@pytest.mark.asyncio
async def test_planner_agent_heuristics():
    """Verify PlannerAgent breaks a query into subtasks correctly."""
    class MockDB: pass
    class MockProvider:
        async def embed(self, texts):
            return {"embeddings": [[0.1] * 1536 for _ in texts]}
    
    db = MockDB()
    # Null out db to skip database checks for isolated unit test
    # We just want to check expanded heuristics logic execution.
    planner = PlannerAgent(db=None, provider=MockProvider())
    
    state = {
        "query": "Explain ContextBuilder and rewrite the formatting limits",
        "repo_id": "MOCK-REPO",
        "trace": DebugTrace(),
        "embed_cache": {}
    }
    
    out_state = await planner.run(state)
    
    # Subtasks should be populated via query expansion heuristics
    assert "sub_tasks" in out_state
    sub_tasks = out_state["sub_tasks"]
    assert len(sub_tasks) > 1
    assert sub_tasks[0] == "Explain ContextBuilder and rewrite the formatting limits"
    # Should flag symbols extracted
    assert any("ContextBuilder" in t for t in sub_tasks)
