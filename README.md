# Dense vs Sparse vs Hybrid RAG — Multi-Dataset Comparison

A rigorous benchmarking framework comparing three retrieval strategies for
Retrieval-Augmented QA across **two complementary datasets**:

| Pipeline | Retriever | Index |
|---|---|---|
| **Dense** | SentenceTransformers bi-encoder | FAISS `IndexFlatIP` |
| **Sparse** | BM25 Okapi | rank-bm25 |
| **Hybrid** | Reciprocal Rank Fusion (RRF) | — |

---

## Why two datasets?

| Dataset | Question style | Expected winner | Why |
|---|---|---|---|
| **SQuAD** | Written while reading the passage | Sparse (BM25) | High lexical overlap between question and passage |
| **TriviaQA** | Trivia-style, paraphrased | Dense | Semantic matching needed; exact keywords rarely shared |

Running both datasets in one experiment directly tests whether your findings
**generalise**.

---

## Project structure

```
.
├── main.py                     # Entry point
├── requirements.txt
├── src/
│   ├── config.py               # RAGConfig — all hyperparameters
│   ├── data_loader.py          # Unified loader for SQuAD + TriviaQA
│   ├── preprocessing.py        # Overlapping token-window chunker
│   ├── dense_retriever.py      # SentenceTransformer + FAISS
│   ├── sparse_retriever.py     # BM25 Okapi (normalised scores)
│   ├── hybrid_retriever.py     # Reciprocal Rank Fusion
│   ├── generator.py            # Extractive QA (deepset/roberta-base-squad2)
│   ├── evaluation.py           # EM, F1, RHR, bootstrap CI
│   ├── experiment.py           # Multi-dataset experiment loop
│   ├── ablation.py             # Top-k and chunk-size ablation runners
│   └── visualization.py        # All plots (per-dataset + cross-dataset)
└── outputs/                    # Generated CSV, JSON, PNG files
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the experiment

### Full run — both datasets, 300 queries each (recommended)
```powershell
python main.py
```
Expected runtime: ~60–90 min on CPU. Models and datasets are cached after
the first run.

### Fast smoke-test (both datasets, 20 queries each, ~5 min)
```powershell
python main.py --max_query_examples 20 --max_corpus_examples 150
```

### SQuAD only
```powershell
python main.py --datasets squad
```

### TriviaQA only
```powershell
python main.py --datasets trivia_qa
```

### With top-k ablation (sweeps k=1,3,5,10 — builds index once per dataset)
```powershell
python main.py --run_topk_ablation
```

### With chunk-size ablation (slow — rebuilds index 3× per dataset)
```powershell
python main.py --run_chunk_ablation
```

### All ablations together
```powershell
python main.py --run_topk_ablation --run_chunk_ablation
```

### Without hybrid retriever
```powershell
python main.py --no_hybrid
```

---

## Output files

### Per-dataset plots (generated for each dataset separately)
| File | Contents |
|---|---|
| `accuracy_comparison_{dataset}.png` | EM + F1 + RHR bars with 95% CI error bars |
| `f1_distribution_{dataset}.png` | Violin + strip plot of per-query F1 |
| `failure_mode_breakdown_{dataset}.png` | Stacked bars: where failures happen |
| `pipeline_disagreement_{dataset}.png` | Dense F1 vs Sparse F1 scatter |

### Cross-dataset plots (generated when both datasets are run)
| File | Contents |
|---|---|
| `cross_dataset_f1.png` | Grouped bars: F1 per pipeline × dataset |
| `cross_dataset_rhr.png` | Grouped bars: RHR per pipeline × dataset |
| `dataset_delta.png` | Bar: TriviaQA F1 − SQuAD F1 per pipeline |

### Shared plots
| File | Contents |
|---|---|
| `latency_comparison.png` | Stacked retrieval + generation latency |
| `accuracy_latency_tradeoff.png` | F1 vs total latency scatter |
| `topk_ablation.png` | F1 and RHR vs top_k *(if --run_topk_ablation)* |
| `chunk_ablation.png` | F1 and RHR vs chunk size *(if --run_chunk_ablation)* |

### Data files
| File | Contents |
|---|---|
| `query_level_results.csv` | Per-query EM, F1, RHR, latencies, dataset column |
| `summary_results.csv` | Per-pipeline aggregates with bootstrap CIs |
| `query_level_results.json` | Full results including retrieved chunk texts |

---

## Key design decisions

### Two-dataset design
SQuAD and TriviaQA test complementary retrieval regimes. SQuAD questions are
written while looking at the passage, creating high lexical overlap that favours
BM25. TriviaQA questions are trivia-style with paraphrasing, favouring semantic
(dense) retrieval. Running both datasets together produces a generalisation
claim: "sparse wins on lexical-overlap QA; dense wins on paraphrased QA".

### Corrected Retrieval Hit Rate (RHR)
The original `retrieval_hit` checked chunk vs gold *paragraph* overlap. This
version checks whether any gold *answer string* is a substring of any retrieved
chunk — the correct definition. This lets RHR diagnose whether failure is a
retrieval problem or a generator problem.

### Bootstrap confidence intervals
With n=300 queries per dataset, 95% CIs are approximately ±6pp — tight enough
to make reliable claims. The original n=60 gave ±13pp CIs where most
differences were within noise.

### Hybrid via Reciprocal Rank Fusion
RRF uses rank positions rather than raw scores, avoiding the scale mismatch
between cosine similarity (dense) and BM25 scores (sparse). Each retriever
fetches min(3k, 50) candidates before fusion.

### Shared generator
All pipelines use `deepset/roberta-base-squad2` (extractive span selection).
Since it cannot hallucinate tokens outside the context, answer quality
directly measures retrieval quality.

---

## Notes

- First run downloads models (~90 MB MiniLM, ~500 MB RoBERTa) and both
  datasets. Subsequent runs use the Hugging Face cache (`~/.cache/huggingface`).
- Results are fully reproducible given the same `--random_seed` (default: 42).
- For memory-constrained machines, try `--generator_model_name deepset/minilm-uncased-squad2`
  (~120 MB, ~3× faster).
- TriviaQA uses the `rc` (reading comprehension) split, which includes
  Wikipedia context passages. The `unfiltered` split does not.
