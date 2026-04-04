import pytest
from datetime import datetime, timedelta, timezone

from app.memory.scoring import memory_score


def test_memory_scoring_decay():
    """Test memory scoring logic decays older similarities linearly."""
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    
    # Perfect recent memory
    sc1 = memory_score(1.0, now, now)
    assert sc1 == 1.0
    
    # 90 days ago decays weight heavily
    old = now - timedelta(days=90)
    sc2 = memory_score(1.0, old, now)
    assert sc2 == 0.5  # linear floor hit securely bounds memory limits
    
    # 45 days ago should be 0.75
    mid = now - timedelta(days=45)
    sc3 = memory_score(1.0, mid, now)
    assert sc3 == 0.75
    
    # Semantic scaling integrates with recency
    sc4 = memory_score(0.8, mid, now)
    assert sc4 == pytest.approx(0.6)  # 0.8 * 0.75


@pytest.mark.asyncio
async def test_memory_builder_truncation():
    """Ensure context builder correctly limits memory sizes and extracts strings."""
    from app.agents.context_builder import assemble_memory_context
    memories = [
        {"type": "decision", "symbol": "ContextBuilder", "context": "Refactored token limits.", "timestamp": "2026-04-01"},
        {"type": "bug_fix", "symbol": "scoring", "context": "Null ptr on asyncpg query.", "timestamp": "2026-04-02"},
    ]
    formatted = assemble_memory_context(memories, max_tokens=1500)
    assert "[PAST DECISION | ContextBuilder]" in formatted
    assert "[PAST BUG FIX | scoring]" in formatted
    
    # Test super restrictive token max (will only fit first entry if tokens=25)
    formatted_restricted = assemble_memory_context(memories, max_tokens=25)
    assert "[PAST DECISION | ContextBuilder]" in formatted_restricted
    assert "[PAST BUG FIX | scoring]" not in formatted_restricted
