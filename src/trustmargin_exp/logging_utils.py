"""Small JSONL helpers; files are written atomically after each stage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict], overwrite: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to regenerate")
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
