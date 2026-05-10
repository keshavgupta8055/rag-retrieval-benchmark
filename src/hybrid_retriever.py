"""
Hybrid Retriever — Reciprocal Rank Fusion (RRF) of dense + sparse results.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods", SIGIR 2009.
"""
from typing import Dict, List, Tuple

from src.preprocessing import ContextChunk


class HybridRetriever:
    """
    Combines a dense retriever (FAISS + SentenceTransformers) and a sparse
    retriever (BM25) using Reciprocal Rank Fusion.

    RRF formula:
        score(d) = Σ_r  1 / (k + rank_r(d))

    where rank_r(d) is document d's 1-indexed rank in retriever r, and k is a
    smoothing constant (default 60, from the original paper).

    Why RRF over score fusion?
    ─────────────────────────
    Dense retriever returns cosine similarities in [−1, 1].
    BM25 returns term-frequency-based scores in [0, ∞).
    Even after normalising BM25 to [0, 1] per query, the *distributions*
    differ: cosine similarities cluster around 0.3–0.8 while BM25 scores are
    heavily skewed toward 0. Fusing these directly amplifies whichever
    retriever happens to produce larger absolute values.

    RRF sidesteps this entirely — it only uses rank positions, which are
    already on the same ordinal scale for both retrievers.

    Over-fetching strategy:
    ───────────────────────
    We retrieve `min(top_k * 3, 50)` candidates from each retriever before
    fusion. This ensures RRF has enough candidates to re-rank meaningfully,
    especially at small top_k where both retrievers may agree on the top doc
    but disagree on positions 2–5.
    """

    def __init__(self, dense_retriever, sparse_retriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[Tuple[ContextChunk, float]]:
        candidate_k = min(top_k * 3, 50)

        dense_results = self.dense.search(query, top_k=candidate_k)
        sparse_results = self.sparse.search(query, top_k=candidate_k)

        chunk_lookup: Dict[int, ContextChunk] = {}
        rrf_scores: Dict[int, float] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (chunk, _) in enumerate(sparse_results):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        return [(chunk_lookup[cid], rrf_scores[cid]) for cid in sorted_ids[:top_k]]
