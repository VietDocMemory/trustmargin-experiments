"""Answer-only, length-normalized teacher-forced likelihood."""
from __future__ import annotations

import math
from collections.abc import Sequence


def mean_answer_log_probs(token_log_probs: Sequence[float], answer_mask: Sequence[bool]) -> float:
    if len(token_log_probs) != len(answer_mask):
        raise ValueError("Log-probability and mask lengths differ")
    answer_values = [float(value) for value, included in zip(token_log_probs, answer_mask) if included]
    if not answer_values:
        raise ValueError("Empty answer token span")
    result = sum(answer_values) / len(answer_values)
    if not math.isfinite(result):
        raise ValueError("Non-finite likelihood")
    return result


def answer_mask(prompt_length: int, answer_length: int) -> list[bool]:
    if prompt_length < 1 or answer_length < 1:
        raise ValueError("Prompt and answer must each contain at least one token")
    # Shifted logits position i predicts token i+1. The first answer token is
    # therefore predicted at position prompt_length - 1.
    return [False] * (prompt_length - 1) + [True] * answer_length


def teacher_forced_log_likelihood(model, tokenizer, prompt_ids: list[int], answer: str, device: str) -> float:
    import torch

    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("A non-empty tokenized chat prompt is required")
    if not answer.strip():
        raise ValueError("Cannot score an empty answer")
    # The chat template already supplies BOS and the assistant turn prefix.
    # Tokenize the continuation alone to avoid inserting a second BOS/EOS.
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("Answer has no tokens")
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        try:
            # Gemma can project logits only for the answer boundary/span. This
            # avoids allocating [prompt_length, vocabulary_size] on a 12 GB GPU.
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                logits_to_keep=len(answer_ids) + 1,
            ).logits
            if logits.shape[1] != len(answer_ids) + 1:
                raise RuntimeError("Unexpected logits_to_keep shape")
            answer_logits = logits[0, :-1, :].float()
            targets = ids[0, -len(answer_ids):]
            selected = torch.log_softmax(answer_logits, dim=-1).gather(
                1, targets.unsqueeze(1)
            ).squeeze(1)
        except (TypeError, RuntimeError):
            # Compatibility path for test doubles and older model classes.
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            shifted = logits[0, :-1, :].float()
            targets = ids[0, 1:]
            token_log_probs = torch.log_softmax(shifted, dim=-1).gather(
                1, targets.unsqueeze(1)
            ).squeeze(1)
            mask = torch.tensor(
                answer_mask(len(prompt_ids), len(answer_ids)),
                dtype=torch.bool,
                device=device,
            )
            selected = token_log_probs[mask]
        result = selected.mean().item()
    if selected.numel() != len(answer_ids) or not math.isfinite(result):
        raise ValueError("Invalid answer-only likelihood")
    return result
