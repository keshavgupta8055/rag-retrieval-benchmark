import re
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.preprocessing import ContextChunk


class BM25Retriever:
    """
    BM25 Okapi retriever over ContextChunks.

    Tokenisation: simple lowercase word regex — fast and sufficient for
    English SQuAD text. For multilingual use, swap in a proper tokeniser.

    Score normalisation: raw BM25 scores are divided by the per-query max
    so they land in [0, 1]. This matches the dense retriever's cosine
    similarity range and allows RRF score fusion to treat both equally.

    BM25 parameters (rank-bm25 defaults):
      k1=1.5, b=0.75 — standard Okapi values; tune for domain-specific corpora.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks: List[ContextChunk] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: List[ContextChunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"  BM25 index: {len(chunks)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[ContextChunk, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built yet — call build_index() first.")

        raw_scores = self.bm25.get_scores(self._tokenize(query))

        # Normalise to [0, 1] so scores are comparable with dense cosine sims.
        max_score = raw_scores.max()
        normalized_scores = raw_scores / max_score if max_score > 0 else raw_scores

        top_indices = np.argsort(normalized_scores)[::-1][:top_k]
        return [(self.chunks[i], float(normalized_scores[i])) for i in top_indices]
