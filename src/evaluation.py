import random
import re
import string
from collections import Counter
from typing import Iterable, List, Tuple


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold_answers: Iterable[str]) -> float:
    pred_norm = normalize_text(prediction)
    return float(any(pred_norm == normalize_text(gold) for gold in gold_answers))


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        best_f1 = max(best_f1, score)
    return best_f1


def retrieval_hit(
    gold_answers: List[str],
    retrieved_chunk_texts: List[str],
) -> int:
    """
    Returns 1 if any retrieved chunk contains any gold answer string,
    0 otherwise.

    FIX (v2): The original implementation compared retrieved chunks against
    the gold *context paragraph*, not the gold *answer strings*. A chunk from
    the same paragraph that does NOT contain the answer would incorrectly score
    as a retrieval hit, hiding generator failures.

    Correct interpretation: retrieval succeeds only when the answer text is
    present in the retrieved context — i.e. the model had the information it
    needed. Checking for the answer substring directly tests this.

    This change makes the Retrieval Hit Rate (RHR) a true diagnostic: if
    RHR is high but EM/F1 is low, the generator is the bottleneck; if RHR
    is low, retrieval needs improvement first.
    """
    for answer in gold_answers:
        answer_norm = normalize_text(answer)
        if not answer_norm:
            continue
        for chunk_text in retrieved_chunk_texts:
            if answer_norm in normalize_text(chunk_text):
                return 1
    return 0


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Non-parametric percentile bootstrap confidence interval for the mean.

    Returns (lower_bound, upper_bound) for a two-sided interval at `alpha`.

    With n=60 queries, a 5pp performance difference can easily fall within the
    95% CI. Reporting CIs instead of bare point estimates prevents over-
    interpreting small differences that are within sampling noise.

    Method: percentile bootstrap. Bias-corrected accelerated (BCa) would be
    more accurate at very small n, but percentile bootstrap is sufficient and
    much simpler to audit.
    """
    n = len(scores)
    if n == 0:
        return 0.0, 0.0

    rng = random.Random(seed)
    means = sorted(
        sum(scores[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )

    lower_idx = int((1 - alpha) / 2 * n_bootstrap)
    upper_idx = min(int((1 + alpha) / 2 * n_bootstrap), n_bootstrap - 1)
    return means[lower_idx], means[upper_idx]
