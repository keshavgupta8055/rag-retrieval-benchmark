from typing import List

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class HFAnswerGenerator:
    """
    Extractive QA generator using a SQuAD-finetuned span-selection model.

    Model: deepset/roberta-base-squad2 (default)
    ─────────────────────────────────────────────
    Unlike generative models (flan-t5, GPT), this model predicts start/end
    token positions within the context — it physically cannot hallucinate
    words that are not in the retrieved text, making it ideal for evaluating
    retrieval quality in isolation.

    Other swappable options (change generator_model_name in config):
      deepset/roberta-large-squad2    -- highest EM/F1, ~1.4 GB
      deepset/deberta-v3-base-squad2  -- strong, ~700 MB
      deepset/minilm-uncased-squad2   -- fastest, ~120 MB

    Per-chunk strategy:
    ───────────────────
    We run the model independently on each retrieved chunk and take the
    answer span with the highest start+end logit sum. This is more accurate
    than concatenating all chunks into one long string, which can cause the
    model to cross chunk boundaries and produce incoherent spans.
    """

    def __init__(
        self,
        model_name: str = "deepset/roberta-base-squad2",
        max_new_tokens: int = 96,       # API compat only — unused by extractive model
        max_context_tokens: int = 420,  # API compat only — unused by extractive model
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        model_max_length = self.tokenizer.model_max_length
        if model_max_length is None or model_max_length > 4096:
            model_max_length = 512
        self.max_length = min(model_max_length, 512)
        self.max_answer_len = 64

        print(f"  Generator: {model_name} on {self.device} (max_length={self.max_length})")

    def _predict_best_span(self, question: str, context: str):
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        sequence_ids = inputs.sequence_ids(0)

        context_token_indices = [
            idx for idx, sid in enumerate(sequence_ids) if sid == 1
        ]
        if not context_token_indices:
            return "", float("-inf")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0].detach().cpu()
        end_logits = outputs.end_logits[0].detach().cpu()

        best_score = float("-inf")
        best_span = ""

        for start_index in context_token_indices:
            for end_index in context_token_indices:
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > self.max_answer_len:
                    continue
                start_char, _ = offset_mapping[start_index]
                _, end_char = offset_mapping[end_index]
                candidate = context[start_char:end_char].strip()
                if not candidate:
                    continue
                score = start_logits[start_index].item() + end_logits[end_index].item()
                if score > best_score:
                    best_score = score
                    best_span = candidate

        return best_span, best_score

    def generate_answer(self, question: str, retrieved_contexts: List[str]) -> str:
        """
        Extract the best answer span across all retrieved chunks.

        Each chunk is evaluated independently; the chunk that produces the
        highest logit-sum score wins. This avoids the cross-chunk boundary
        issue that arises when joining all contexts into one long string.
        """
        best_answer = ""
        best_score = float("-inf")

        for context in retrieved_contexts:
            candidate, score = self._predict_best_span(question, context)
            if score > best_score:
                best_answer = candidate
                best_score = score

        return best_answer.strip()
