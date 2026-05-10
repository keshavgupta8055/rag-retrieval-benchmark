from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

# Minimum chunk size: discard trailing fragments shorter than this.
# Micro-chunks (1–9 tokens) are noise — they waste a retrieval slot and
# confuse both the dense encoder (too short for meaningful embeddings) and
# BM25 (too few terms for reliable scoring).
_MIN_CHUNK_TOKENS = 10


@dataclass
class ContextChunk:
    chunk_id: int
    source_context_id: int
    text: str


def split_contexts_into_chunks(
    contexts: List[str],
    tokenizer_name: str,
    max_tokens: int = 250,
    overlap_tokens: int = 50,
) -> List[ContextChunk]:
    """
    Split each context string into overlapping fixed-width token windows.

    tokenizer_name should be set via RAGConfig.chunker_tokenizer_name
    (defaulting to dense_model_name when empty). Using a neutral tokenizer
    like bert-base-uncased avoids implicitly sizing chunks to the dense
    encoder's context window, which would disadvantage BM25.

    Chunks shorter than _MIN_CHUNK_TOKENS tokens are discarded.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chunks: List[ContextChunk] = []
    next_chunk_id = 0

    step = max_tokens - overlap_tokens
    if step <= 0:
        raise ValueError(
            f"max_tokens ({max_tokens}) must be larger than overlap_tokens ({overlap_tokens})"
        )

    for context_id, context in enumerate(contexts):
        token_ids = tokenizer.encode(context, add_special_tokens=False)

        if not token_ids:
            continue

        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + max_tokens]

            if len(window) < _MIN_CHUNK_TOKENS:
                continue

            chunk_text = tokenizer.decode(window, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(
                    ContextChunk(
                        chunk_id=next_chunk_id,
                        source_context_id=context_id,
                        text=chunk_text,
                    )
                )
                next_chunk_id += 1

    return chunks
