"""Persistent subprocess wrapper for the original async RAG retriever."""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from .config import Config


class RAGSource:
    def __init__(self, config: Config):
        self.config = config
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "RAGSource":
        if not self.config.rag_python.is_file():
            raise FileNotFoundError(f"RAG Python not found: {self.config.rag_python}")
        worker = Path(__file__).with_name("rag_worker.py")
        self.process = subprocess.Popen(
            [str(self.config.rag_python), "-u", str(worker), str(self.config.rag_repo)],
            cwd=self.config.rag_repo,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)

    def _retrieve_sync(self, question: str, collection_name: str, top_k: int) -> list[dict]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("RAGSource must be used as a context manager")
        process.stdin.write(json.dumps({"question": question, "collection_name": collection_name, "top_k": top_k}, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"RAG worker exited with code {process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return [
            {"content": str(p["content"]), "score": p.get("score"), "page": p.get("page"), "chunk_index": p.get("chunk_index")}
            for p in result["passages"][:top_k]
        ]

    async def retrieve(self, question: str, collection_name: str, top_k: int) -> list[dict]:
        return await asyncio.to_thread(self._retrieve_sync, question, collection_name, top_k)
