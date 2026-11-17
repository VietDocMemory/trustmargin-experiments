"""Frozen linear answerability probe for retrieved context plus question."""
from __future__ import annotations

import json
import math
from pathlib import Path


def bare_prompt_ids(tokenizer, passage: str, question: str, max_len: int) -> tuple[list[int], bool]:
    """Render the exact bare prompt used to train the exported probe."""

    def render(candidate_passage: str) -> list[int]:
        content = f"{candidate_passage}\n\n{question}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    ids = render(passage)
    if len(ids) <= max_len:
        return ids, False
    low, high = 0, len(passage)
    while low < high:
        middle = (low + high + 1) // 2
        if len(render(passage[:middle])) <= max_len:
            low = middle
        else:
            high = middle - 1
    return render(passage[:low]), True


class AnswerabilityProbe:
    """Load and apply a frozen standardized linear probe."""

    def __init__(self, directory: str | Path, base_model: str, torch, device: str):
        import numpy as np

        self.directory = Path(directory)
        config_path = self.directory / "config.json"
        weights_path = self.directory / "probe.npz"
        if not config_path.is_file():
            raise FileNotFoundError(f"Answerability probe config not found: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Answerability probe weights not found: {weights_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if self.config["model"] != base_model:
            raise ValueError(
                f"Answerability probe model {self.config['model']!r} differs from {base_model!r}"
            )
        if self.config.get("prompt_style") != "bare" or self.config.get("pooling") != "last":
            raise ValueError("Answerability probe requires bare prompting and last-token pooling")

        with np.load(weights_path) as archive:
            arrays = {
                name: np.asarray(archive[name], dtype=np.float32).reshape(-1)
                for name in ("weight", "mean", "scale")
            }
            bias = float(np.asarray(archive["bias"]).reshape(-1)[0])
        arrays["scale"] = np.where(arrays["scale"] == 0, 1.0, arrays["scale"])

        self.hidden_state_index = int(self.config["hidden_state_index"])
        self.hidden_size = int(self.config["hidden_size"])
        self.max_len = int(self.config["max_len"])
        self.threshold = float(self.config["threshold"])
        self.weight = torch.tensor(arrays["weight"], dtype=torch.float32, device=device)
        self.mean = torch.tensor(arrays["mean"], dtype=torch.float32, device=device)
        self.scale = torch.tensor(arrays["scale"], dtype=torch.float32, device=device)
        self.bias = torch.tensor(bias, dtype=torch.float32, device=device)
        if any(value.numel() != self.hidden_size for value in (self.weight, self.mean, self.scale)):
            raise ValueError("Answerability probe array size does not match hidden_size")

    def score(self, model, tokenizer, question: str, contexts: list[str], torch) -> dict:
        """Score each retrieved chunk plus the top-2 pair and max-pool the probabilities."""
        if not contexts:
            raise ValueError("Answerability probe requires at least one retrieved context")

        candidates = [
            {
                "kind": "single",
                "chunk_indices": [index],
                "passage": context,
            }
            for index, context in enumerate(contexts[:8], start=1)
        ]
        if len(contexts) >= 2:
            candidates.append(
                {
                    "kind": "top2_combined",
                    "chunk_indices": [1, 2],
                    "passage": f"{contexts[0]}\n\n{contexts[1]}",
                }
            )

        encoded = [
            bare_prompt_ids(tokenizer, candidate["passage"], question, self.max_len)
            for candidate in candidates
        ]
        max_prompt_tokens = max(len(prompt_ids) for prompt_ids, _ in encoded)
        input_ids = torch.zeros(
            (len(encoded), max_prompt_tokens),
            dtype=torch.long,
            device=self.weight.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        final_token_indices = []
        for row_index, (prompt_ids, _) in enumerate(encoded):
            prompt_length = len(prompt_ids)
            input_ids[row_index, :prompt_length] = torch.tensor(
                prompt_ids,
                dtype=torch.long,
                device=self.weight.device,
            )
            attention_mask[row_index, :prompt_length] = 1
            final_token_indices.append(prompt_length - 1)

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            states = output.hidden_states
            if self.hidden_state_index >= len(states):
                raise ValueError(
                    f"Probe requests hidden_states[{self.hidden_state_index}], got {len(states)} states"
                )
            hidden = states[self.hidden_state_index]
            row_indices = torch.arange(len(encoded), device=self.weight.device)
            token_indices = torch.tensor(
                final_token_indices,
                dtype=torch.long,
                device=self.weight.device,
            )
            features = hidden[row_indices, token_indices].float()
            if features.shape != (len(encoded), self.hidden_size):
                raise ValueError(
                    "Probe expects feature shape "
                    f"{(len(encoded), self.hidden_size)}, got {tuple(features.shape)}"
                )
            logits = ((features - self.mean) / self.scale) @ self.weight + self.bias
            probabilities = torch.sigmoid(logits).tolist()

        if not all(math.isfinite(float(probability)) for probability in probabilities):
            raise ValueError("Answerability probe returned a non-finite probability")

        component_scores = []
        for candidate, (prompt_ids, truncated), probability in zip(
            candidates,
            encoded,
            probabilities,
        ):
            component_scores.append(
                {
                    "kind": candidate["kind"],
                    "chunk_indices": candidate["chunk_indices"],
                    "prob_answerable": float(probability),
                    "prompt_tokens": len(prompt_ids),
                    "context_truncated": truncated,
                }
            )

        winning_index = max(
            range(len(component_scores)),
            key=lambda index: component_scores[index]["prob_answerable"],
        )
        winner = component_scores[winning_index]
        probability = winner["prob_answerable"]
        return {
            "prob_answerable": probability,
            "threshold": self.threshold,
            "pred_answerable": probability >= self.threshold,
            "hidden_state_index": self.hidden_state_index,
            "prompt_style": "bare",
            "pooling": "last",
            "context_source": "retrieved",
            "aggregation": "max_top8_single_plus_top2_combined",
            "component_scores": component_scores,
            "winning_component_index": winning_index,
            "winning_kind": winner["kind"],
            "winning_chunk_indices": winner["chunk_indices"],
            "prompt_tokens": winner["prompt_tokens"],
            "context_truncated": winner["context_truncated"],
            "any_context_truncated": any(
                component["context_truncated"] for component in component_scores
            ),
        }
