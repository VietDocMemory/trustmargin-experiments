"""Three-stage experiment runner. Re-arbitration never loads Gemma."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from .config import Config
from .logging_utils import read_jsonl, write_jsonl
from .metrics import summarize
from .trustmargin import arbitrate


def _stable_id(row: dict, index: int) -> str:
    return str(row.get("id") or hashlib.sha256(f'{index}\0{row["question"]}'.encode()).hexdigest()[:16])


def normalize_samples(
    rows: list[dict],
    dataset: str,
    collection: str,
    document: str | None,
    require_document: bool = True,
) -> list[dict]:
    normalized = []
    ids = set()
    for index, raw in enumerate(rows):
        question = raw["question"].strip()
        golds = raw.get("gold_answers") or raw.get("ground_truth_answer") or raw.get("answer")
        if isinstance(golds, str):
            golds = [golds]
        if not question or not isinstance(golds, list) or not golds or any(not isinstance(g, str) for g in golds):
            raise ValueError(f"Sample {index} lacks a question or gold answer")
        sample_id = _stable_id(raw, index)
        if sample_id in ids:
            raise ValueError(f"Duplicate sample id {sample_id}")
        ids.add(sample_id)
        document_path = raw.get("document_path") or document
        document_text = raw.get("document_text")
        if require_document and not document_path and not document_text:
            raise ValueError(f"Sample {sample_id} needs document_path, document_text, or --document")
        normalized.append({
            "id": sample_id, "dataset": raw.get("dataset", dataset), "question": question,
            "gold_answers": golds, "question_type": raw.get("question_type"),
            "split": raw.get("split"), "collection_name": raw.get("collection_name", collection),
            "document_path": document_path, "document_text": document_text,
        })
    return normalized


def validate_alignment(rows: list[dict]) -> None:
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate candidate id {row['id']}")
        ids.add(row["id"])
        if not row.get("question") or not row.get("gold_answers"):
            raise ValueError(f"Unaligned sample {row['id']}: missing question/gold")
        if not isinstance(row.get("rag_answer"), str) or not isinstance(row.get("d2l_answer"), str):
            raise ValueError(f"Unaligned sample {row['id']}: missing candidate")
        if not isinstance(row.get("retrieved_contexts"), list):
            raise ValueError(f"Unaligned sample {row['id']}: missing contexts")
        if row.get("candidate_ids") != {"d2l": row["id"], "rag": row["id"]}:
            raise ValueError(f"Unaligned candidate ids for {row['id']}")


def load_document(sample: dict) -> str:
    if sample.get("document_text"):
        return sample["document_text"]
    path = Path(sample["document_path"]).expanduser().resolve()
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Document has no extractable text: {path}")
    return text


async def _retrieve_all(samples: list[dict], config: Config) -> list[dict]:
    from .rag_source import RAGSource

    with RAGSource(config) as rag:
        for sample in samples:
            started = time.perf_counter()
            passages = await rag.retrieve(sample["question"], sample["collection_name"], config.top_k)
            if not passages:
                raise ValueError(f"No retrieved contexts for sample {sample['id']}")
            sample["passages"] = passages
            sample["retrieval_latency"] = time.perf_counter() - started
    return samples


def generate_candidates(
    input_path: str | Path,
    output_path: str | Path,
    config: Config,
    collection: str,
    document: str | None,
    limit: int | None = None,
    overwrite: bool = False,
    document_source: str = "provided",
) -> list[dict]:
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; cached candidates are preserved")
    if not config.d2l_checkpoint.is_file():
        raise FileNotFoundError(f"D2L checkpoint not found: {config.d2l_checkpoint}")
    if not config.rag_python.is_file():
        raise FileNotFoundError(f"RAG Python not found: {config.rag_python}")
    if document_source not in {"provided", "retrieved"}:
        raise ValueError("document_source must be 'provided' or 'retrieved'")
    raw = read_jsonl(input_path)
    samples = normalize_samples(
        raw[:limit] if limit else raw,
        Path(input_path).stem,
        collection,
        document,
        require_document=document_source == "provided",
    )
    for sample in samples:
        if not sample["collection_name"]:
            raise ValueError(f"Missing collection for {sample['id']}")
        if sample["document_path"] and not Path(sample["document_path"]).expanduser().is_file():
            raise FileNotFoundError(f"Document not found: {sample['document_path']}")
    samples = asyncio.run(_retrieve_all(samples, config))
    # The RAG worker has exited; BGE and reranker no longer occupy GPU memory.
    from .model_backend import GemmaBackend
    from .d2l_source import D2LSource

    backend = GemmaBackend(config)
    d2l = D2LSource(backend, Path(output_path).parent / "adapter_cache")
    outputs = []
    for sample in samples:
        started = time.perf_counter()
        if backend.device == "cuda" and config.log_cuda_memory:
            backend.torch.cuda.reset_peak_memory_stats()
        contexts = [passage["content"] for passage in sample["passages"]]
        document_text = (
            "\n\n".join(contexts)
            if document_source == "retrieved"
            else load_document(sample)
        )
        document_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
        adapter = d2l.build_adapter(document_text)
        rag_started = time.perf_counter()
        rag_answer = backend.generate_base(sample["question"], contexts)
        rag_diagnostics = dict(backend.last_generation)
        rag_latency = time.perf_counter() - rag_started
        d2l_answer = d2l.generate_d2l(sample["question"], adapter)
        d2l_diagnostics = dict(d2l.generation_diagnostics)
        if not rag_answer or not d2l_answer:
            raise ValueError(f"Empty candidate for {sample['id']}")
        record = {
            "id": sample["id"], "candidate_ids": {"d2l": sample["id"], "rag": sample["id"]},
            "dataset": sample["dataset"], "split": sample["split"],
            "question": sample["question"], "gold_answers": sample["gold_answers"],
            "question_type": sample["question_type"],
            "document_id": Path(sample["document_path"]).stem if sample["document_path"] else document_hash[:16],
            "document_hash": document_hash,
            "document_source": document_source,
            "document_chars": len(document_text),
            "collection_name": sample["collection_name"],
            "retrieved_contexts": sample["passages"],
            "retrieval_scores": [p["score"] for p in sample["passages"]],
            "d2l_answer": d2l_answer, "rag_answer": rag_answer,
            "raw_d2l_answer": d2l_diagnostics["raw_answer"],
            "raw_rag_answer": rag_diagnostics["raw_answer"],
            "candidate_diagnostics": {
                "d2l": d2l_diagnostics,
                "rag": rag_diagnostics,
            },
            "base_model": config.base_model, "tokenizer": config.base_model,
            "adapter_cache_hit": d2l.adapter_cache_hit,
            "latency": {
                "retrieval": sample["retrieval_latency"],
                "adapter_build": d2l.adapter_generation_latency + d2l.adapter_load_latency,
                "adapter_generation": d2l.adapter_generation_latency,
                "adapter_load": d2l.adapter_load_latency,
                "rag_generation": rag_latency,
                "d2l_generation": d2l.d2l_generation_latency,
                "likelihood_scoring": None,
                "total": sample["retrieval_latency"] + time.perf_counter() - started,
            },
            "memory": backend.memory(),
        }
        outputs.append(record)
    validate_alignment(outputs)
    write_jsonl(output_path, outputs, overwrite=overwrite)
    return outputs


def score_candidates(input_path: str | Path, output_path: str | Path, config: Config, overwrite: bool = False) -> list[dict]:
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; cached scores are preserved")
    rows = read_jsonl(input_path)
    validate_alignment(rows)
    from .model_backend import GemmaBackend

    backend = GemmaBackend(config)
    for row in rows:
        if row.get("base_model") != config.base_model or row.get("tokenizer") != config.base_model:
            raise ValueError(f"Model/tokenizer mismatch for {row['id']}")
        started = time.perf_counter()
        if backend.device == "cuda" and config.log_cuda_memory:
            backend.torch.cuda.reset_peak_memory_stats()
        contexts = [passage["content"] for passage in row["retrieved_contexts"]]
        views = {
            "q_only": backend.prompt(
                row["question"], max_tokens=config.max_score_prompt_tokens
            ),
            "q_context": backend.prompt(
                row["question"], contexts, config.max_score_prompt_tokens
            ),
            "context_only": backend.prompt(
                contexts=contexts, max_tokens=config.max_score_prompt_tokens
            ),
        }
        row["score_prompt_tokens"] = {name: len(prompt) for name, prompt in views.items()}
        row["scores"] = {
            name: {
                "d2l": backend.score_teacher_forced(prompt, row["d2l_answer"]),
                "rag": backend.score_teacher_forced(prompt, row["rag_answer"]),
            }
            for name, prompt in views.items()
        }
        elapsed = time.perf_counter() - started
        row["latency"]["likelihood_scoring"] = elapsed
        row["latency"]["total"] += elapsed
        scoring_memory = backend.memory()
        generation_memory = row.get("memory", {})
        row["memory"] = {
            "peak_allocated_mb": max(
                (value for value in (generation_memory.get("peak_allocated_mb"), scoring_memory.get("peak_allocated_mb")) if value is not None),
                default=None,
            ),
            "peak_reserved_mb": max(
                (value for value in (generation_memory.get("peak_reserved_mb"), scoring_memory.get("peak_reserved_mb")) if value is not None),
                default=None,
            ),
            "allocated_after_mb": scoring_memory.get("allocated_after_mb"),
            "generation": generation_memory,
            "scoring": scoring_memory,
        }
    write_jsonl(output_path, rows, overwrite=overwrite)
    return rows


def run_eval(input_path: str | Path, output_path: str | Path, report_path: str | Path, config: Config, lambda_bind: float | None = None, tau: float | None = None, sweep: bool = False, overwrite: bool = False) -> dict:
    for artifact in (Path(output_path), Path(report_path)):
        if artifact.exists() and not overwrite:
            raise FileExistsError(f"{artifact} exists; pass --overwrite")
    rows = read_jsonl(input_path)
    validate_alignment(rows)
    lam = config.lambda_bind if lambda_bind is None else lambda_bind
    threshold = config.tau if tau is None else tau
    # Oracle analysis is calculated before TrustMargin decisions.
    both_wrong_threshold = getattr(config, "both_wrong_f1_threshold", 0.2)
    oracle_report = summarize(rows, lam, threshold, both_wrong_threshold)
    if sweep:
        if any(row.get("split") != "validation" for row in rows):
            raise ValueError("Threshold sweep requires every row to have split='validation'")
        trials = []
        for candidate_lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            scores = sorted(
                arbitrate(row["scores"], candidate_lam, 0.0)["trust_score"]
                for row in rows
            )
            quantile_thresholds = {
                scores[round(index * (len(scores) - 1) / 20)] for index in range(21)
            }
            thresholds = quantile_thresholds | {-3.0, -2.0, -1.5, -1.0, -0.5, 0.0}
            for candidate_tau in sorted(thresholds):
                report = summarize(
                    rows,
                    candidate_lam,
                    candidate_tau,
                    both_wrong_threshold,
                )
                metrics = report["metrics"]["tm"]
                trials.append({
                    "lambda_bind": candidate_lam,
                    "tau": candidate_tau,
                    "token_f1": metrics["token_f1"],
                    "rouge_l_f1": metrics.get("rouge_l_f1", 0.0),
                    "exact_match": metrics["exact_match"],
                    "selected_rag": report["selection_counts"]["rag"],
                    "selected_d2l": report["selection_counts"]["d2l"],
                })
        best = max(
            trials,
            key=lambda row: (
                row["token_f1"], row["rouge_l_f1"], row["exact_match"],
                -abs(row["lambda_bind"] - config.lambda_bind),
                -abs(row["tau"] - config.tau),
            ),
        )
        lam, threshold = best["lambda_bind"], best["tau"]
        oracle_report["sweep"] = {"label": "validation-tuned", "trials": trials, "best": best}
    output_rows = []
    for row in rows:
        decision = arbitrate(row["scores"], lam, threshold)
        row.update(decision)
        row["final_answer"] = row[f'{decision["selected_source"]}_answer']
        output_rows.append(row)
    report = summarize(rows, lam, threshold, both_wrong_threshold)
    if sweep:
        report["sweep"] = oracle_report["sweep"]
    report["lambda_bind"] = lam
    report["tau"] = threshold
    write_jsonl(output_path, output_rows, overwrite=overwrite)
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
