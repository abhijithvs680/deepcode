"""
Reranker — Uses an LLM to evaluate the relevance of retrieved code chunks 
and re-orders them to remove semantic noise.
"""
import re
from app.providers.base import LLMProvider
from app.core.logger import get_logger

logger = get_logger(__name__)

async def rerank(
    query: str,
    chunks: list[dict],
    provider: LLMProvider,
    top_k: int = 5
) -> tuple[list[dict], list[dict]]:
    """
    Rerank code chunks by asking the LLM to score their relevance to the query.
    
    Returns:
        (best_chunks, trace_entries)
    """
    if not chunks:
        return [], []
        
    scored_chunks = []
    trace_entries = []
    
    for chunk in chunks:
        br = chunk.get("score_breakdown") or {}
        breakdown_hint = (
            f"composite_hints semantic={br.get('semantic')} keyword={br.get('keyword')} "
            f"symbol={br.get('symbol')} path={br.get('path')} git={br.get('git')}"
        )
        prompt = f"""
        Query: {query}

        Prior retrieval scores (hints only, not ground truth): {breakdown_hint}

        Code Snippet ({chunk['file_path']}):
        ```
        {chunk['content'][:6000]}
        ```

        Task: Score the relevance of this code snippet to answering the query on a scale of 1 to 10.
        Reply with ONLY the integer score. No explanation.
        """
        
        try:
            # We use generation, not embedding
            response = await provider.generate(prompt)
            # Find the first number in the response string
            matches = re.findall(r'-?\d+', response.strip())
            
            if matches:
                # LLM should return a simple number, but sometimes adds chatter
                score = float(matches[0])
            else:
                score = 5.0
                
        except Exception as exc:
            logger.warning("Reranker failed for a chunk, defaulting to 5: %s", exc)
            score = 5.0
            
        scored_chunks.append((score, chunk))
        trace_entries.append({
            "file_path": chunk["file_path"],
            "chunk_index": chunk["chunk_index"],
            "rerank_score": score
        })
        
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Take top_k
    best_chunks = [c for _, c in scored_chunks[:top_k]]
    
    logger.info("Reranked %d chunks down to top %d", len(chunks), len(best_chunks))
    return best_chunks, trace_entries
