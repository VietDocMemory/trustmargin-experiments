"""JSONL bridge launched by the RAG environment; never imports its generator."""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path


async def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(repo))
    # Reserve stdout for the JSONL protocol; third-party model loaders can print.
    with contextlib.redirect_stdout(sys.stderr):
        from src.retrieval.search_engine import RAGRetriever

        retriever = RAGRetriever()
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    passages = await retriever.search(
                        request["question"], request["collection_name"], request["top_k"]
                    )
                result = {"passages": passages[: request["top_k"]]}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        await retriever.client.close()


if __name__ == "__main__":
    asyncio.run(main())
