"""
Experiment runner — supports multiple datasets.

Each dataset gets its own retrieval index (chunks differ per dataset), but
all datasets share the same generator model (loaded once). Results are tagged
with a 'dataset' column so downstream analysis and plots can group by both
pipeline and dataset.
"""

import json
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from src.config import RAGConfig
from src.data_loader import load_data
from src.dense_retriever import DenseRetriever
from src.evaluation import bootstrap_ci, exact_match, f1_score, retrieval_hit
from src.generator import HFAnswerGenerator
from src.hybrid_retriever import HybridRetriever
from src.preprocessing import split_contexts_into_chunks
from src.sparse_retriever import BM25Retriever


def _run_single_pipeline(
    pipeline_name: str,
    retriever,
    question: str,
    gold_answers: List[str],
    generator: HFAnswerGenerator,
    top_k: int,
) -> Dict:
    t0 = time.perf_counter()
    retrieved = retriever.search(question, top_k=top_k)
    retrieval_latency = time.perf_counter() - t0

    contexts = [chunk.text for chunk, _ in retrieved]

    t1 = time.perf_counter()
    prediction = generator.generate_answer(question, contexts)
    generation_latency = time.perf_counter() - t1

    return {
        "pipeline": pipeline_name,
        "question": question,
        "gold_answers": gold_answers,
        "prediction": prediction,
        "em":               exact_match(prediction, gold_answers),
        "f1":               f1_score(prediction, gold_answers),
        "retrieval_hit":    retrieval_hit(gold_answers, contexts),
        "retrieval_latency_sec":  retrieval_latency,
        "generation_latency_sec": generation_latency,
        "total_latency_sec":      retrieval_latency + generation_latency,
        "retrieved_chunk_ids":    [chunk.chunk_id for chunk, _ in retrieved],
        "retrieved_scores":       [score for _, score in retrieved],
        "retrieved_texts":        contexts,   # JSON only
    }


def _build_summary(results_df: pd.DataFrame, n_bootstrap: int, ci_alpha: float) -> pd.DataFrame:
    records = []
    for (dataset, pipeline), grp in results_df.groupby(["dataset", "pipeline"]):
        em_s  = grp["em"].tolist()
        f1_s  = grp["f1"].tolist()
        rhr_s = grp["retrieval_hit"].tolist()

        em_lo,  em_hi  = bootstrap_ci(em_s,  n_bootstrap=n_bootstrap, alpha=ci_alpha)
        f1_lo,  f1_hi  = bootstrap_ci(f1_s,  n_bootstrap=n_bootstrap, alpha=ci_alpha)
        rhr_lo, rhr_hi = bootstrap_ci(rhr_s, n_bootstrap=n_bootstrap, alpha=ci_alpha)

        records.append({
            "dataset":  dataset,
            "pipeline": pipeline,
            "n_queries": len(grp),
            "em":   grp["em"].mean(),
            "f1":   grp["f1"].mean(),
            "retrieval_hit_rate": grp["retrieval_hit"].mean(),
            "median_f1":       grp["f1"].median(),
            "std_f1":          grp["f1"].std(),
            "fraction_zero_f1":(grp["f1"] == 0).mean(),
            "em_ci_low":  em_lo,   "em_ci_high":  em_hi,
            "f1_ci_low":  f1_lo,   "f1_ci_high":  f1_hi,
            "rhr_ci_low": rhr_lo,  "rhr_ci_high": rhr_hi,
            "avg_retrieval_latency_sec":  grp["retrieval_latency_sec"].mean(),
            "avg_generation_latency_sec": grp["generation_latency_sec"].mean(),
            "avg_total_latency_sec":      grp["total_latency_sec"].mean(),
        })

    return (pd.DataFrame(records)
              .sort_values(["dataset", "pipeline"])
              .reset_index(drop=True))


def _run_one_dataset(
    dataset_name: str,
    config: RAGConfig,
    generator: HFAnswerGenerator,
) -> List[Dict]:
    """Build index and run all pipelines for one dataset. Returns raw result dicts."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    print("Loading data...")
    _, query_examples, contexts = load_data(
        dataset_name=dataset_name,
        max_corpus_examples=config.max_corpus_examples,
        max_query_examples=config.max_query_examples,
        seed=config.random_seed,
    )

    tokenizer_name = config.chunker_tokenizer_name or config.dense_model_name
    print(f"Chunking contexts (tokenizer: {tokenizer_name})...")
    chunks = split_contexts_into_chunks(
        contexts=contexts,
        tokenizer_name=tokenizer_name,
        max_tokens=config.chunk_max_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    print(f"Total chunks: {len(chunks)}")

    print("Building dense index (FAISS)...")
    dense = DenseRetriever(config.dense_model_name)
    dense.build_index(chunks)

    print("Building sparse index (BM25)...")
    sparse = BM25Retriever()
    sparse.build_index(chunks)

    retrievers: Dict[str, object] = {"dense": dense, "sparse": sparse}
    if config.enable_hybrid:
        print("Building hybrid retriever (RRF)...")
        retrievers["hybrid"] = HybridRetriever(dense, sparse, rrf_k=config.rrf_k)

    results: List[Dict] = []
    for ex in tqdm(query_examples, desc=f"  [{dataset_name}] Queries"):
        for name, retriever in retrievers.items():
            row = _run_single_pipeline(
                pipeline_name=name,
                retriever=retriever,
                question=ex["question"],
                gold_answers=ex["answers"],
                generator=generator,
                top_k=config.top_k,
            )
            row["dataset"] = dataset_name
            results.append(row)

    return results


def run_experiment(config: RAGConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(config.output_dir, exist_ok=True)

    # Load generator once — shared across all datasets
    print("Loading generator model (shared across datasets)...")
    generator = HFAnswerGenerator(
        model_name=config.generator_model_name,
        max_new_tokens=config.max_new_tokens,
        max_context_tokens=config.max_context_tokens,
    )

    all_results: List[Dict] = []
    for dataset_name in config.dataset_names:
        all_results.extend(_run_one_dataset(dataset_name, config, generator))

    results_df = pd.DataFrame(all_results)
    summary_df = _build_summary(results_df, config.n_bootstrap, config.ci_alpha)

    # ── Save ──────────────────────────────────────────────────────────────
    results_csv = os.path.join(config.output_dir, "query_level_results.csv")
    summary_csv = os.path.join(config.output_dir, "summary_results.csv")
    results_json = os.path.join(config.output_dir, "query_level_results.json")

    results_df.drop(columns=["retrieved_texts"]).to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────
    ci_pct = int(config.ci_alpha * 100)
    print("\n" + "=" * 72)
    print("FINAL COMPARISON REPORT")
    print("=" * 72)
    for dataset in config.dataset_names:
        print(f"\n  ── {dataset.upper()} ──")
        sub = summary_df[summary_df["dataset"] == dataset]
        for _, row in sub.iterrows():
            print(f"  Pipeline : {row['pipeline']}")
            print(f"  EM       : {row['em']:.3f}  [{row['em_ci_low']:.3f}, {row['em_ci_high']:.3f}] {ci_pct}% CI")
            print(f"  F1       : {row['f1']:.3f}  [{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] {ci_pct}% CI")
            print(f"  RHR      : {row['retrieval_hit_rate']:.3f}  [{row['rhr_ci_low']:.3f}, {row['rhr_ci_high']:.3f}] {ci_pct}% CI")
            print(f"  Median F1: {row['median_f1']:.3f}  |  Zero-F1: {row['fraction_zero_f1']:.1%}")
            print(f"  Latency  : {row['avg_total_latency_sec']:.3f}s  "
                  f"(ret {row['avg_retrieval_latency_sec']:.3f}s + "
                  f"gen {row['avg_generation_latency_sec']:.3f}s)\n")
    print("=" * 72)
    print(f"\nSaved: {results_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {results_json}")

    return results_df, summary_df
