"""Document-hash cache around D2L's existing internalize/generate API."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .model_backend import GemmaBackend, _in_directory


class D2LSource:
    def __init__(self, backend: GemmaBackend, cache_dir: str | Path):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_generation_latency = 0.0
        self.adapter_load_latency = 0.0
        self.d2l_generation_latency = 0.0
        self.adapter_cache_hit = False
        self.generation_diagnostics: dict = {}

    def build_adapter(self, document_or_context: str):
        if not document_or_context.strip():
            raise ValueError("D2L document is empty")
        checkpoint = self.backend.config.d2l_checkpoint
        digest = hashlib.sha256(
            checkpoint.resolve().as_posix().encode()
            + str(checkpoint.stat().st_mtime_ns).encode()
            + document_or_context.encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{digest}.pt"
        torch = self.backend.torch
        started = time.perf_counter()
        if path.is_file():
            # Only read files created by this experiment; weights_only excludes objects.
            adapter = torch.load(path, map_location=self.backend.device, weights_only=True)
            self.adapter_cache_hit = True
            self.adapter_load_latency = time.perf_counter() - started
            self.adapter_generation_latency = 0.0
            return adapter
        self.backend.model.reset()
        with torch.inference_mode(), _in_directory(self.backend.config.d2l_repo):
            self.backend.model.internalize(document_or_context)
        adapter = self.backend.model.generated_loras
        if not adapter:
            raise RuntimeError("D2L internalize produced no adapter")
        cpu_adapter = {
            module: {part: tensor.detach().cpu() for part, tensor in matrices.items()}
            for module, matrices in adapter.items()
        }
        temporary = path.with_suffix(".tmp")
        try:
            torch.save(cpu_adapter, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.backend.model.reset()
        self.adapter_cache_hit = False
        self.adapter_generation_latency = time.perf_counter() - started
        self.adapter_load_latency = 0.0
        return adapter

    def generate_d2l(self, question: str, adapter) -> str:
        started = time.perf_counter()
        answer = self.backend.generate_with_adapter(question, adapter)
        self.generation_diagnostics = dict(self.backend.last_generation)
        self.d2l_generation_latency = time.perf_counter() - started
        return answer
