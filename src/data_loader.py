"""
Unified data loader for SQuAD and TriviaQA.

Both loaders return the same dict schema so the rest of the pipeline is
completely dataset-agnostic:
  {
    "id":       str,
    "question": str,
    "context":  str,          # source paragraph (used for RHR tracking)
    "answers":  List[str],    # deduplicated gold answer strings
  }

SQuAD (squad):
  - Context is the passage the question was written from.
  - Answers are verbatim spans extracted from that passage.
  - Questions have high lexical overlap with the passage → BM25-friendly.

TriviaQA rc.wikipedia (trivia_qa / rc):
  - Context comes from entity_pages.wiki_context (Wikipedia paragraphs).
  - Answers are aliased strings (e.g. "Sinclair Lewis", "Harry Sinclair Lewis").
  - Questions are trivia-style with paraphrase → semantic retrieval more useful.
  - We use the first non-empty wiki_context paragraph as the "gold context"
    so RHR can check whether the answer is present in retrieved chunks.
"""

import random
from typing import Dict, List, Tuple

from datasets import load_dataset


# ── SQuAD loader ──────────────────────────────────────────────────────────

def load_squad_data(
    max_corpus_examples: int,
    max_query_examples: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    ds = load_dataset("squad", split="validation")
    return _load_squad_like(ds, max_corpus_examples, max_query_examples, seed,
                            context_key="context",
                            answers_extractor=lambda row: list(dict.fromkeys(row["answers"]["text"])))


# ── TriviaQA loader ───────────────────────────────────────────────────────

def load_trivia_qa_data(
    max_corpus_examples: int,
    max_query_examples: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Load TriviaQA (rc.wikipedia split).

    Context strategy:
      Each TriviaQA example has 1-N Wikipedia entity pages. We use the first
      non-empty wiki_context string as the retrieval source paragraph. Only
      examples with at least one non-empty wiki_context are kept.

    Answer strategy:
      TriviaQA provides a canonical `value` plus a list of `aliases`
      (e.g. "Sinclair Lewis", "Harry Sinclair Lewis", "H. S. Lewis").
      We use all deduplicated aliases as gold answers so that any valid
      surface form counts as correct for EM/F1 evaluation.
    """
    ds = load_dataset("trivia_qa", "rc", split="validation")

    context_set = set()
    unique_contexts: List[str] = []
    corpus_examples: List[Dict] = []

    for row in ds:
        wiki_contexts = row.get("entity_pages", {}).get("wiki_context", [])
        # Use the first non-empty Wikipedia context paragraph.
        context = next((c for c in wiki_contexts if c and len(c.strip()) > 20), None)
        if context is None:
            continue

        answers = list(dict.fromkeys(
            [row["answer"]["value"]] + row["answer"].get("aliases", [])
        ))
        if not answers:
            continue

        if context not in context_set:
            context_set.add(context)
            unique_contexts.append(context)

        corpus_examples.append({
            "id":       row.get("question_id", ""),
            "question": row["question"],
            "context":  context,
            "answers":  answers,
        })

        if len(unique_contexts) >= max_corpus_examples:
            break

    # Queries: all examples whose context is in our corpus.
    filtered_queries = [ex for ex in corpus_examples if ex["context"] in context_set]
    rng = random.Random(seed)
    rng.shuffle(filtered_queries)
    query_examples = filtered_queries[:max_query_examples]

    print(
        f"  Corpus: {len(unique_contexts)} unique contexts "
        f"({len(corpus_examples)} total rows)\n"
        f"  Queries: {len(query_examples)} evaluation examples "
        f"(shuffled, seed={seed})"
    )
    return corpus_examples, query_examples, unique_contexts


# ── Unified dispatcher ────────────────────────────────────────────────────

def load_data(
    dataset_name: str,
    max_corpus_examples: int,
    max_query_examples: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Route to the correct loader based on dataset_name.
    Supported: "squad", "trivia_qa".
    """
    if dataset_name == "squad":
        return load_squad_data(max_corpus_examples, max_query_examples, seed)
    elif dataset_name == "trivia_qa":
        return load_trivia_qa_data(max_corpus_examples, max_query_examples, seed)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported values: 'squad', 'trivia_qa'."
        )


# ── Internal helper (SQuAD-format datasets) ───────────────────────────────

def _load_squad_like(ds, max_corpus_examples, max_query_examples, seed,
                     context_key, answers_extractor):
    context_set = set()
    unique_contexts: List[str] = []
    corpus_examples: List[Dict] = []

    for row in ds:
        context = row[context_key]
        if context not in context_set:
            context_set.add(context)
            unique_contexts.append(context)

        corpus_examples.append({
            "id":       row.get("id", ""),
            "question": row["question"],
            "context":  context,
            "answers":  answers_extractor(row),
        })

        if len(unique_contexts) >= max_corpus_examples:
            break

    filtered_queries = [ex for ex in corpus_examples if ex["context"] in context_set]
    rng = random.Random(seed)
    rng.shuffle(filtered_queries)
    query_examples = filtered_queries[:max_query_examples]

    print(
        f"  Corpus: {len(unique_contexts)} unique contexts "
        f"({len(corpus_examples)} total rows)\n"
        f"  Queries: {len(query_examples)} evaluation examples "
        f"(shuffled, seed={seed})"
    )
    return corpus_examples, query_examples, unique_contexts
