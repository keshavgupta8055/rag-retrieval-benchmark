"""
Ablation study runners for the dense vs sparse RAG comparison.

Two ablation modes:
  1. top-k sweep   — vary the number of retrieved chunks (indices built once)
  2. chunk-size sweep — vary the chunking window size (indices rebuilt each time)
"""
import time
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from src.config import RAGConfig
from src.dense_retriever import DenseRetriever
from src.evaluation import exact_match, f1_score, retrieval_hit
from src.generator import HFAnswerGenerator
from src.hybrid_retriever import HybridRetriever
from src.preprocessing import split_contexts_into_chunks
from src.sparse_retriever import BM25Retriever


def _score_pipeline(
    retriever,
    query_examples: List[Dict],
    generator: HFAnswerGenerator,
    top_k: int,
    desc: str = "",
) -> Dict:
    """Run one retriever over all queries and return aggregate metrics."""
    em_list, f1_list, rhr_list, lat_list = [], [], [], []
    for ex in tqdm(query_examples, desc=desc, leave=False):
        t0 = time.perf_counter()
        retrieved = retriever.search(ex["question"], top_k=top_k)
        texts = [c.text for c, _ in retrieved]
        pred = generator.generate_answer(ex["question"], texts)
        lat_list.append(time.perf_counter() - t0)

        em_list.append(exact_match(pred, ex["answers"]))
        f1_list.append(f1_score(pred, ex["answers"]))
        rhr_list.append(retrieval_hit(ex["answers"], texts))

    n = len(query_examples)
    return {
        "em": sum(em_list) / n,
        "f1": sum(f1_list) / n,
        "retrieval_hit_rate": sum(rhr_list) / n,
        "avg_total_latency_sec": sum(lat_list) / n,
    }


def run_topk_ablation(
    query_examples: List[Dict],
    contexts: List[str],
    generator: HFAnswerGenerator,
    config: RAGConfig,
) -> pd.DataFrame:
    """
    Sweep top_k ∈ config.ablation_top_k_values for dense, sparse, and hybrid.

    Indices are built ONCE at max(top_k_values) and reused across all k values.
    Only the number of results returned changes per iteration, making this
    ablation relatively cheap (~4× the cost of a single run for 4 k values).

    Returns long-format DataFrame:
    [pipeline, top_k, em, f1, retrieval_hit_rate, avg_total_latency_sec]
    """
    print("\n[Ablation] Building shared indices for top-k sweep...")
    tokenizer_name = config.chunker_tokenizer_name or config.dense_model_name
    chunks = split_contexts_into_chunks(
        contexts=contexts,
        tokenizer_name=tokenizer_name,
        max_tokens=config.chunk_max_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    print(f"[Ablation] {len(chunks)} chunks")

    dense = DenseRetriever(config.dense_model_name)
    dense.build_index(chunks)

    sparse = BM25Retriever()
    sparse.build_index(chunks)

    hybrid = HybridRetriever(dense, sparse, rrf_k=config.rrf_k)
    retrievers = {"dense": dense, "sparse": sparse, "hybrid": hybrid}

    records: List[Dict] = []
    for k in sorted(config.ablation_top_k_values):
        print(f"[Ablation] top_k={k}")
        for name, retriever in retrievers.items():
            scores = _score_pipeline(
                retriever, query_examples, generator, top_k=k,
                desc=f"  {name} k={k}",
            )
            records.append({"pipeline": name, "top_k": k, **scores})

    return pd.DataFrame(records)


def run_chunk_ablation(
    query_examples: List[Dict],
    contexts: List[str],
    generator: HFAnswerGenerator,
    config: RAGConfig,
) -> pd.DataFrame:
    """
    Sweep chunk_max_tokens ∈ config.ablation_chunk_sizes.

    Each chunk size requires rebuilding both the FAISS index and the BM25
    corpus — this is the expensive ablation. To keep runtime tractable, the
    evaluation is capped at 40 queries regardless of config.max_query_examples.

    Returns long-format DataFrame:
    [pipeline, chunk_size, n_chunks, em, f1, retrieval_hit_rate]
    """
    eval_examples = query_examples[:40] if len(query_examples) > 40 else query_examples
    tokenizer_name = config.chunker_tokenizer_name or config.dense_model_name
    records: List[Dict] = []

    for chunk_size in sorted(config.ablation_chunk_sizes):
        # Scale overlap proportionally so smaller chunks still have overlap.
        overlap = min(config.chunk_overlap_tokens, chunk_size // 5)
        print(f"\n[Ablation] chunk_size={chunk_size} tokens (overlap={overlap})")

        chunks = split_contexts_into_chunks(
            contexts=contexts,
            tokenizer_name=tokenizer_name,
            max_tokens=chunk_size,
            overlap_tokens=overlap,
        )
        print(f"[Ablation] {len(chunks)} chunks")

        dense = DenseRetriever(config.dense_model_name)
        dense.build_index(chunks)

        sparse = BM25Retriever()
        sparse.build_index(chunks)

        hybrid = HybridRetriever(dense, sparse, rrf_k=config.rrf_k)
        retrievers = {"dense": dense, "sparse": sparse, "hybrid": hybrid}

        for name, retriever in retrievers.items():
            scores = _score_pipeline(
                retriever, eval_examples, generator, top_k=config.top_k,
                desc=f"  {name} chunk={chunk_size}",
            )
            records.append({
                "pipeline": name,
                "chunk_size": chunk_size,
                "n_chunks": len(chunks),
                **scores,
            })

    return pd.DataFrame(records)
