from dataclasses import dataclass, field
from typing import List


@dataclass
class RAGConfig:
    # ── Datasets ───────────────────────────────────────────────────────────
    # Run the full comparison on every dataset in this list.
    # Supported: "squad", "trivia_qa"
    # The two datasets test complementary retrieval regimes:
    #   SQuAD    — passage-based QA, high lexical overlap → BM25 favoured
    #   TriviaQA — trivia-style QA, paraphrased answers  → dense favoured
    dataset_names: List[str] = field(default_factory=lambda: ["squad", "trivia_qa"])

    max_corpus_examples: int = 600
    max_query_examples: int = 300   # raised from 60 → tighter CIs
    random_seed: int = 42

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_max_tokens: int = 250
    chunk_overlap_tokens: int = 50
    # Separate tokenizer for chunking removes the bias where windows are sized
    # to the dense encoder's context length, disadvantaging BM25.
    # Leave empty to default to dense_model_name.
    chunker_tokenizer_name: str = ""

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k: int = 3
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Hybrid (RRF) ───────────────────────────────────────────────────────
    enable_hybrid: bool = True
    rrf_k: int = 60

    # ── Generator ──────────────────────────────────────────────────────────
    generator_model_name: str = "deepset/roberta-base-squad2"
    max_new_tokens: int = 96
    max_context_tokens: int = 420

    # ── Ablations ──────────────────────────────────────────────────────────
    run_topk_ablation: bool = False
    ablation_top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    run_chunk_ablation: bool = False
    ablation_chunk_sizes: List[int] = field(default_factory=lambda: [100, 200, 350])

    # ── Statistics ─────────────────────────────────────────────────────────
    n_bootstrap: int = 1000
    ci_alpha: float = 0.95

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "outputs"
