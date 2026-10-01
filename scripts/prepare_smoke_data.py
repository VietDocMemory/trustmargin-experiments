"""Prepare a short legal-document smoke corpus with the RAG environment."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_REPO = ROOT.parent / "vietnamese-rag-system"
COLLECTION = "setup_smoke_legal"


async def main() -> None:
    import fitz
    from langchain_core.documents import Document

    sys.path.insert(0, str(RAG_REPO))
    os.chdir(RAG_REPO)  # RAG's .env is resolved relative to its repo.
    from src.ingestion.vector_store import VectorStoreManager
    from src.utils.nlp_utils import segment_vietnamese

    pdf = fitz.open(RAG_REPO / "data/72_2020_QH14_431147.pdf")
    page_text = pdf[1].get_text()
    end = page_text.find("12.")
    if end <= 0:
        raise ValueError("Expected definition boundary not found on page 2")
    document = page_text[:end].strip()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "setup_smoke_document.txt").write_text(document, encoding="utf-8")

    dataset = RAG_REPO / "data/rag_evaluation_dataset.jsonl"
    with dataset.open(encoding="utf-8") as source, (output_dir / "setup_smoke_input.jsonl").open("w", encoding="utf-8") as target:
        for index in range(3):
            row = json.loads(next(source))
            row.update(id=f"setup-smoke-{index + 1}", dataset="setup-smoke", split="smoke")
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    store = VectorStoreManager()
    try:
        await store.create_collection(COLLECTION)
        chunk = Document(
            page_content=segment_vietnamese(document),
            metadata={
                "original_text": document,
                "page": 2,
                "chunk_index": 0,
                "chunk_length": len(document),
                "is_list": False,
            },
        )
        await store.upsert_documents([chunk], collection_name=COLLECTION, batch_size=1)
    finally:
        await store.client.close()
    print(f"Prepared 3 smoke samples and Qdrant collection {COLLECTION}")


if __name__ == "__main__":
    asyncio.run(main())
