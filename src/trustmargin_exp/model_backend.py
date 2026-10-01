"""One checkpoint-loaded Gemma base model for both candidates and scoring."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .config import Config
from .likelihood import teacher_forced_log_likelihood


@contextmanager
def _in_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class GemmaBackend:
    def __init__(self, config: Config):
        import sys
        import torch

        self.config = config
        self.torch = torch
        if not config.d2l_checkpoint.is_file():
            raise FileNotFoundError(f"D2L checkpoint not found: {config.d2l_checkpoint}")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = config.device
        if config.device == "cuda":
            self.dtype = torch.bfloat16 if config.dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        sys.path.insert(0, str(config.d2l_repo / "src"))
        from ctx_to_lora.model_loading import get_tokenizer
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

        with _in_directory(config.d2l_repo):
            # D2L checkpoints contain Python configuration objects and must be trusted.
            state = torch.load(config.d2l_checkpoint, map_location="cpu", weights_only=False)
            actual = state.get("base_model_name_or_path")
            if actual != config.base_model:
                raise ValueError(f"Checkpoint base model {actual!r} differs from {config.base_model!r}")
            self.model = ModulatedPretrainedModel.from_state_dict(
                state,
                train=False,
                use_sequence_packing=False,
                use_flash_attn=False,
                base_model_kwargs={
                    "device_map": config.device,
                    "torch_dtype": self.dtype,
                    "attn_implementation": "eager",
                },
            ).eval()
            self.tokenizer = get_tokenizer(config.base_model)
        self.model.reset()

    def prompt(self, question: str | None = None, contexts: list[str] | None = None) -> list[int]:
        if not question and not contexts:
            raise ValueError("Question or context is required")
        parts = []
        if contexts is not None:
            parts.append("<context>\n" + "\n\n".join(contexts) + "\n</context>")
        if question:
            parts.append("Question:\n" + question)
        parts.append("Answer:")
        messages = [
            {"role": "system", "content": "Answer concisely. Use the supplied context when present."},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            add_special_tokens=False, return_attention_mask=False,
        )

    def _generate(self, prompt_ids: list[int], adapter=None) -> str:
        torch = self.torch
        self.model.reset()
        input_ids = torch.tensor([prompt_ids], device=self.device)
        try:
            with torch.inference_mode():
                if adapter is None:
                    output = self.model.base_model.generate(
                        input_ids=input_ids, max_new_tokens=self.config.max_new_tokens,
                        do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    # D2L's generate() expects its LoRA forwards to have been
                    # patched by internalize(); a cached adapter skips that call.
                    self.model.patch_lora_forward()
                    self.model.generated_loras = adapter
                    output = self.model.generate(
                        input_ids=input_ids, max_new_tokens=self.config.max_new_tokens,
                        do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                    )
            return self.tokenizer.decode(output[0, len(prompt_ids):], skip_special_tokens=True).strip()
        finally:
            self.model.reset()

    def generate_base(self, question: str, contexts: list[str] | None = None) -> str:
        return self._generate(self.prompt(question, contexts))

    def generate_with_adapter(self, question: str, adapter) -> str:
        return self._generate(self.prompt(question), adapter)

    def score_teacher_forced(self, prompt: list[int], answer: str, adapter=None) -> float:
        self.model.reset()
        if adapter is not None:
            from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers

            self.model.patch_lora_forward()
            merged = self.model.combine_lora(
                adapter,
                self.torch.tensor([1], device=self.device),
                lora_bias=self.model.hypernet.get_head_bias()
                if self.model.hypernet.config.use_bias else None,
            )
            apply_lora_to_layers(
                self.model.base_model, self.model.hypernet.layer_indices, merged,
                self.torch.tensor([1], device=self.device),
            )
        try:
            return teacher_forced_log_likelihood(
                self.model.base_model, self.tokenizer, prompt, answer, self.device
            )
        finally:
            self.model.reset()

    def memory(self) -> dict:
        if self.config.log_cuda_memory and self.device == "cuda":
            return {
                "peak_allocated_mb": self.torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mb": self.torch.cuda.max_memory_reserved() / 2**20,
                "allocated_after_mb": self.torch.cuda.memory_allocated() / 2**20,
            }
        return {"peak_allocated_mb": None, "peak_reserved_mb": None, "allocated_after_mb": None}
