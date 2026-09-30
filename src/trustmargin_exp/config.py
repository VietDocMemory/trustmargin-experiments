"""Configuration with all paths resolved independently of the current directory."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, fallback: Path) -> Path:
    raw = Path(os.environ.get(name, str(fallback))).expanduser()
    return (raw if raw.is_absolute() else ROOT / raw).resolve()


@dataclass(frozen=True)
class Config:
    base_model: str
    max_new_tokens: int
    top_k: int
    lambda_bind: float
    tau: float
    dtype: str
    device: str
    seed: int
    log_cuda_memory: bool
    rag_repo: Path
    d2l_repo: Path
    d2l_checkpoint: Path
    rag_python: Path


def load_config(path: str | Path | None = None) -> Config:
    import yaml

    config_path = Path(path) if path else ROOT / "configs/default.yaml"
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["base_model"] != "google/gemma-2-2b-it":
        raise ValueError("This experiment requires google/gemma-2-2b-it")
    if cfg["generation"]["do_sample"] or cfg["generation"]["temperature"] != 0:
        raise ValueError("Generation must be deterministic")
    rag_repo = _path("RAG_REPO_PATH", ROOT.parent / "vietnamese-rag-system")
    d2l_repo = _path("D2L_REPO_PATH", ROOT.parent / "doc-to-lora")
    python_name = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return Config(
        base_model=cfg["base_model"],
        max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
        top_k=int(cfg["rag"]["top_k"]),
        lambda_bind=float(cfg["trustmargin"]["lambda_bind"]),
        tau=float(cfg["trustmargin"]["tau"]),
        dtype=cfg["runtime"]["dtype"],
        device=cfg["runtime"]["device"],
        seed=int(cfg["runtime"]["seed"]),
        log_cuda_memory=bool(cfg["logging"]["log_cuda_memory"]),
        rag_repo=rag_repo,
        d2l_repo=d2l_repo,
        d2l_checkpoint=_path("D2L_CHECKPOINT", d2l_repo / "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"),
        rag_python=_path("RAG_PYTHON", rag_repo / ".venv" / python_name),
    )
