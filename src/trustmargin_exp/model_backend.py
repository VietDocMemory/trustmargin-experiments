"""One checkpoint-loaded Gemma base model for both candidates and scoring."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from .config import Config
from .likelihood import teacher_forced_log_likelihood
from .text import ABSTENTION, looks_vietnamese, normalize_abstention


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
        self.last_generation: dict = {}

    def prompt(
        self,
        question: str | None = None,
        contexts: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> list[int]:
        if not question and not contexts:
            raise ValueError("Question or context is required")
        context_text = "\n\n".join(contexts or [])

        def render(text: str) -> list[int]:
            parts = []
            if contexts is not None:
                parts.append("<context>\n" + text + "\n</context>")
            if question:
                parts.append("Câu hỏi:\n" + question)
            parts.append("Câu trả lời:")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Trả lời hoàn toàn bằng tiếng Việt, ngắn gọn và trực tiếp. "
                        "Chỉ dùng thông tin trong tài liệu hoặc ngữ cảnh được cung cấp. "
                        f"Nếu không đủ thông tin, chỉ trả lời chính xác: \"{ABSTENTION}\" "
                        "Không thêm giải thích cho câu từ chối."
                    ),
                },
                {"role": "user", "content": "\n\n".join(parts)},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_attention_mask=False,
            )

        prompt_ids = render(context_text)
        if max_tokens and len(prompt_ids) > max_tokens and contexts is not None:
            empty_length = len(render(""))
            budget = max(1, max_tokens - empty_length - 8)
            context_ids = self.tokenizer.encode(context_text, add_special_tokens=False)[:budget]
            prompt_ids = render(self.tokenizer.decode(context_ids, skip_special_tokens=True))
            while len(prompt_ids) > max_tokens and context_ids:
                context_ids = context_ids[: max(1, len(context_ids) - (len(prompt_ids) - max_tokens) - 4)]
                prompt_ids = render(self.tokenizer.decode(context_ids, skip_special_tokens=True))
        if max_tokens and len(prompt_ids) > max_tokens:
            raise ValueError("Prompt overhead exceeds configured token budget")
        return prompt_ids

    def _generate_once(self, prompt_ids: list[int], max_new_tokens: int, adapter=None):
        torch = self.torch
        self.model.reset()
        input_ids = torch.tensor([prompt_ids], device=self.device)
        try:
            with torch.inference_mode():
                if adapter is None:
                    output = self.model.base_model.generate(
                        input_ids=input_ids, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=self.config.repetition_penalty,
                        no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    )
                else:
                    # D2L's generate() expects its LoRA forwards to have been
                    # patched by internalize(); a cached adapter skips that call.
                    self.model.patch_lora_forward()
                    self.model.generated_loras = adapter
                    output = self.model.generate(
                        input_ids=input_ids, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=self.config.repetition_penalty,
                        no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    )
            generated = output[0, len(prompt_ids):]
            eos_ids = self.tokenizer.eos_token_id
            eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
            ended_with_eos = bool(len(generated) and int(generated[-1]) in eos_ids)
            return generated, ended_with_eos
        finally:
            self.model.reset()

    def _generate(self, prompt_ids: list[int], adapter=None) -> str:
        generated, ended_with_eos = self._generate_once(
            prompt_ids, self.config.max_new_tokens, adapter
        )
        initially_truncated = len(generated) >= self.config.max_new_tokens and not ended_with_eos
        retried = False
        if initially_truncated and self.config.max_retry_new_tokens > self.config.max_new_tokens:
            retried = True
            generated, ended_with_eos = self._generate_once(
                prompt_ids, self.config.max_retry_new_tokens, adapter
            )
        raw_answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer, abstention_normalized = normalize_abstention(raw_answer)
        final_truncated = (
            len(generated) >= self.config.max_retry_new_tokens and not ended_with_eos
            if retried
            else initially_truncated
        )
        self.last_generation = {
            "raw_answer": raw_answer,
            "answer_tokens": int(len(generated)),
            "initially_truncated": initially_truncated,
            "retried": retried,
            "final_truncated": final_truncated,
            "ended_with_eos": ended_with_eos,
            "abstention_normalized": abstention_normalized,
            "looks_vietnamese": looks_vietnamese(answer),
            "prompt_tokens": len(prompt_ids),
        }
        return answer

    def generate_base(self, question: str, contexts: list[str] | None = None) -> str:
        return self._generate(
            self.prompt(question, contexts, self.config.max_generation_prompt_tokens)
        )

    def generate_with_adapter(self, question: str, adapter) -> str:
        return self._generate(
            self.prompt(question, max_tokens=self.config.max_generation_prompt_tokens), adapter
        )

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
