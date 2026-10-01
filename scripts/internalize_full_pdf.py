"""Internalize every PDF page independently with D2L and audit page coverage.

One page corresponds to one adapter. D2L cannot fit the full document in one
context window, so these adapters must be selected or combined at query time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trustmargin_exp.config import load_config
from trustmargin_exp.model_backend import GemmaBackend, _in_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=ROOT.parent / "vietnamese-rag-system/data/72_2020_QH14_431147.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full_pdf_d2l")
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
    if any(not page.strip() for page in pages):
        raise ValueError("A PDF page has no extractable text; OCR is required")

    config = load_config()
    backend = GemmaBackend(config)
    ctx_tokenizer = backend.tokenizer
    template = config.d2l_repo / "chat_templates/google/gemma-2-2b-it.jinja"
    ctx_tokenizer.chat_template = template.read_text(encoding="utf-8")
    max_context = backend.model.ctx_encoder.base_model.config.max_position_embeddings
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    manifest = {"pdf": str(pdf), "sha256": digest, "total_pages": len(pages),
                "max_context_tokens": max_context, "pages": []}
    for index, page in enumerate(pages, 1):
        input_ids = ctx_tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": page.strip()}],
            tokenize=True, add_generation_prompt=True, add_special_tokens=False,
        )
        if len(input_ids) > max_context:
            raise ValueError(f"Page {index} has {len(input_ids)} tokens, exceeding {max_context}; split it before internalization")
        page_hash = hashlib.sha256(page.encode("utf-8")).hexdigest()
        target = args.output / f"page_{index:03d}.pt"
        started = time.perf_counter()
        if not target.exists():
            with backend.torch.inference_mode(), _in_directory(config.d2l_repo):
                backend.model.internalize(page)
            adapter = {name: {key: value.detach().cpu() for key, value in matrices.items()}
                       for name, matrices in backend.model.generated_loras.items()}
            backend.torch.save({"page": index, "text_sha256": page_hash,
                                "pdf_sha256": digest, "adapter": adapter}, target)
            backend.model.reset()
        else:
            cached = backend.torch.load(target, map_location="cpu", weights_only=True)
            if cached["page"] != index or cached["text_sha256"] != page_hash or cached["pdf_sha256"] != digest:
                raise ValueError(f"Stale adapter cache: {target}")
        manifest["pages"].append({"page": index, "tokens": len(input_ids),
                                   "text_sha256": page_hash, "adapter": str(target),
                                   "seconds": round(time.perf_counter() - started, 3)})
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{index}/{len(pages)}: {len(input_ids)} tokens", flush=True)
    print(json.dumps({"processed_pages": len(manifest["pages"]),
                      "total_tokens": sum(p["tokens"] for p in manifest["pages"]),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
