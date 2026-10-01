"""Optional W&B tracking for completed, cached experiment runs."""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT, Config
from .logging_utils import read_jsonl


def log_experiment(
    results_path: str | Path,
    report_path: str | Path,
    config: Config,
    *,
    project: str = "trustmargin-d2l-rag",
    entity: str | None = None,
    mode: str = "online",
    group: str | None = None,
    name: str | None = None,
    extra_files: list[str | Path] | None = None,
) -> str | None:
    """Upload metrics, per-question decisions, and JSONL artifacts.

    Authentication is handled by the W&B SDK; this function never accepts a key.
    """
    import wandb

    results_path = Path(results_path).resolve()
    report_path = Path(report_path).resolve()
    rows = read_jsonl(results_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not rows or report["n"] != len(rows):
        raise ValueError("W&B inputs have inconsistent or empty sample counts")
    if mode not in ("online", "offline"):
        raise ValueError("W&B mode must be online or offline")

    values = {
        "base_model": config.base_model,
        "d2l_checkpoint": config.d2l_checkpoint.name,
        "device": config.device,
        "dtype": config.dtype,
        "max_new_tokens": config.max_new_tokens,
        "top_k": config.top_k,
        "lambda_bind": report["lambda_bind"],
        "tau": report["tau"],
        "samples": len(rows),
        "dataset": sorted({str(row.get("dataset")) for row in rows}),
        "collection": sorted({str(row.get("collection_name")) for row in rows}),
        "document_hashes": sorted({str(row.get("document_hash")) for row in rows}),
    }
    log_dir = ROOT / "outputs/wandb"
    log_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=project, entity=entity, mode=mode, group=group, name=name,
        job_type="evaluation", config=values, dir=str(log_dir),
    ) as run:
        scalars = {"samples": report["n"], "oracle/gain_over_best_single_f1": report["oracle_gain_over_best_single_source_f1"]}
        for source, metrics in report["metrics"].items():
            scalars.update({f"metrics/{source}/{metric}": value for metric, value in metrics.items()})
        scalars.update({f"oracle/count_{key}": value for key, value in report["oracle_counts"].items()})
        for key in ("retrieval", "adapter_build", "rag_generation", "d2l_generation", "likelihood_scoring", "total"):
            measurements = [row.get("latency", {}).get(key) for row in rows]
            measurements = [value for value in measurements if isinstance(value, (int, float))]
            if measurements:
                scalars[f"latency/mean_{key}_seconds"] = sum(measurements) / len(measurements)
        gpu_peaks = [row.get("memory", {}).get("peak_allocated_mb") for row in rows]
        gpu_peaks = [value for value in gpu_peaks if isinstance(value, (int, float))]
        if gpu_peaks:
            scalars["gpu/max_peak_allocated_mb"] = max(gpu_peaks)
        scalars["selection/rag_count"] = sum(row.get("selected_source") == "rag" for row in rows)
        scalars["selection/d2l_count"] = sum(row.get("selected_source") == "d2l" for row in rows)
        scalars["adapter/cache_hit_count"] = sum(bool(row.get("adapter_cache_hit")) for row in rows)
        if report.get("sweep"):
            best = report["sweep"]["best"]
            scalars.update({f"sweep/best_{key}": value for key, value in best.items()})
        run.log(scalars)

        columns = ["id", "question", "d2l_answer", "rag_answer", "selected_source", "final_answer", "gold_answers", "m_prior", "m_bind", "trust_score"]
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*(json.dumps(row[key], ensure_ascii=False) if key == "gold_answers" else row.get(key) for key in columns))
        run.log({"decisions": table})

        artifact = wandb.Artifact(f"trustmargin-results-{run.id}", type="evaluation")
        files = [results_path, report_path, *(Path(p).resolve() for p in extra_files or [])]
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        return run.url
