import argparse
import os

from src.config import RAGConfig
from src.experiment import run_experiment
from src.visualization import plot_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense vs Sparse vs Hybrid RAG — multi-dataset comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    parser.add_argument("--datasets", nargs="+",
                        default=["squad", "trivia_qa"],
                        choices=["squad", "trivia_qa"],
                        help="Datasets to evaluate on. Both by default.")

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--max_corpus_examples", type=int, default=600)
    parser.add_argument("--max_query_examples",  type=int, default=300)

    # ── Chunking ──────────────────────────────────────────────────────────
    parser.add_argument("--chunk_max_tokens",     type=int, default=250)
    parser.add_argument("--chunk_overlap_tokens", type=int, default=50)
    parser.add_argument("--chunker_tokenizer_name", type=str, default="")

    # ── Retrieval ─────────────────────────────────────────────────────────
    parser.add_argument("--top_k",            type=int, default=3)
    parser.add_argument("--dense_model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no_hybrid",        action="store_true")
    parser.add_argument("--rrf_k",            type=int, default=60)

    # ── Generator ─────────────────────────────────────────────────────────
    parser.add_argument("--generator_model_name", type=str,
                        default="deepset/roberta-base-squad2")
    parser.add_argument("--max_new_tokens",    type=int, default=96)
    parser.add_argument("--max_context_tokens",type=int, default=420)

    # ── Ablations ─────────────────────────────────────────────────────────
    parser.add_argument("--run_topk_ablation",  action="store_true")
    parser.add_argument("--run_chunk_ablation", action="store_true")

    # ── Stats ─────────────────────────────────────────────────────────────
    parser.add_argument("--n_bootstrap", type=int, default=1000)

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="outputs")

    return parser.parse_args()


def main():
    args = parse_args()

    config = RAGConfig(
        dataset_names=args.datasets,
        max_corpus_examples=args.max_corpus_examples,
        max_query_examples=args.max_query_examples,
        chunk_max_tokens=args.chunk_max_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        chunker_tokenizer_name=args.chunker_tokenizer_name,
        top_k=args.top_k,
        dense_model_name=args.dense_model_name,
        enable_hybrid=not args.no_hybrid,
        rrf_k=args.rrf_k,
        generator_model_name=args.generator_model_name,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        run_topk_ablation=args.run_topk_ablation,
        run_chunk_ablation=args.run_chunk_ablation,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
    )

    results_df, summary_df = run_experiment(config)

    # ── Ablations ─────────────────────────────────────────────────────────
    ablation_topk_df  = None
    ablation_chunk_df = None

    if config.run_topk_ablation or config.run_chunk_ablation:
        from src.ablation import run_chunk_ablation, run_topk_ablation
        from src.data_loader import load_data
        from src.generator import HFAnswerGenerator

        # Use first dataset for ablations (SQuAD by default)
        abl_dataset = config.dataset_names[0]
        print(f"\nReloading data for ablations (dataset: {abl_dataset})...")
        _, query_examples, contexts = load_data(
            dataset_name=abl_dataset,
            max_corpus_examples=config.max_corpus_examples,
            max_query_examples=config.max_query_examples,
            seed=config.random_seed,
        )
        generator = HFAnswerGenerator(
            model_name=config.generator_model_name,
            max_new_tokens=config.max_new_tokens,
            max_context_tokens=config.max_context_tokens,
        )

        if config.run_topk_ablation:
            print("\nRunning top-k ablation...")
            ablation_topk_df = run_topk_ablation(query_examples, contexts, generator, config)
            ablation_topk_df.to_csv(os.path.join(config.output_dir,"ablation_topk.csv"), index=False)
            print(ablation_topk_df.to_string(index=False))

        if config.run_chunk_ablation:
            print("\nRunning chunk-size ablation...")
            ablation_chunk_df = run_chunk_ablation(query_examples, contexts, generator, config)
            ablation_chunk_df.to_csv(os.path.join(config.output_dir,"ablation_chunk.csv"), index=False)
            print(ablation_chunk_df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_results(summary_df, results_df, config.output_dir,
                 ablation_topk_df=ablation_topk_df,
                 ablation_chunk_df=ablation_chunk_df)

    # ── List generated files ──────────────────────────────────────────────
    print("\nGenerated files in", config.output_dir + ":")
    for f in sorted(os.listdir(config.output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
