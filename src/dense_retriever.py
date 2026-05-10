from typing import List, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is required for dense retrieval.\n"
        "Install via: pip install faiss-cpu"
    ) from exc

from sentence_transformers import SentenceTransformer

from src.preprocessing import ContextChunk


class DenseRetriever:
    """
    Retriever using a SentenceTransformer bi-encoder + FAISS inner-product index.

    Embeddings are L2-normalised before indexing so inner product equals
    cosine similarity — scores are in [0, 1] and directly comparable to
    normalised BM25 scores used in HybridRetriever.

    Index type: IndexFlatIP (exact brute-force).
    For >100k chunks, swap to IndexIVFFlat or HNSW for faster search.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[ContextChunk] = []

    def build_index(self, chunks: List[ContextChunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"  Dense index: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[ContextChunk, float]]:
        if self.index is None:
            raise RuntimeError("Dense index has not been built yet — call build_index() first.")

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, top_k)
        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
