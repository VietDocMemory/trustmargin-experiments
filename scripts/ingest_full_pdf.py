"""Ingest all pages of the legal PDF with the RAG repository's own pipeline."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
RAG_REPO = ROOT.parent / "vietnamese-rag-system"
sys.path.insert(0, str(RAG_REPO))

from src.ingestion.pdf_loader import PDFIngestionPipeline
from src.ingestion.vector_store import VectorStoreManager

PDF = RAG_REPO / "data/72_2020_QH14_431147.pdf"
COLLECTION = "full_legal_109_pages"


async def main() -> None:
    documents = PDFIngestionPipeline().process_pdf(str(PDF))
    pages = sorted({int(doc.metadata["page"]) for doc in documents})
    if pages != list(range(109)):
        raise ValueError(f"PDF coverage incomplete: {pages}")
    store = VectorStoreManager()
    try:
        await store.create_collection(COLLECTION)
        await store.upsert_documents(documents, COLLECTION, batch_size=16)
    finally:
        await store.client.close()
    client = QdrantClient(host="127.0.0.1", port=6333)
    count = client.count(COLLECTION, exact=True).count
    if count != len(documents):
        raise ValueError(f"Qdrant stored {count} of {len(documents)} chunks")
    report = {"pdf": str(PDF), "collection": COLLECTION, "pages": len(pages),
              "chunks": len(documents), "qdrant_points": count}
    output = ROOT / "outputs/full_pdf_rag_ingestion.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
