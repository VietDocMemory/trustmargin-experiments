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
        passage = "\n\n".join(contexts)
        prompt_ids, truncated = bare_prompt_ids(tokenizer, passage, question, self.max_len)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.weight.device)
        attention_mask = torch.ones_like(input_ids)
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
            feature = states[self.hidden_state_index][0, -1].float()
            if feature.numel() != self.hidden_size:
                raise ValueError(
                    f"Probe expects hidden size {self.hidden_size}, got {feature.numel()}"
                )
            logit = ((feature - self.mean) / self.scale) @ self.weight + self.bias
            probability = float(torch.sigmoid(logit).item())

        if not math.isfinite(probability):
            raise ValueError("Answerability probe returned a non-finite probability")
        return {
            "prob_answerable": probability,
            "threshold": self.threshold,
            "pred_answerable": probability >= self.threshold,
            "hidden_state_index": self.hidden_state_index,
            "prompt_style": "bare",
            "pooling": "last",
            "context_source": "retrieved",
            "prompt_tokens": len(prompt_ids),
            "context_truncated": truncated,
        }
