"""Create a deterministic stratified validation split without copying label contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "vietnamese-rag-system/data/rag_evaluation_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--size", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation_input.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/split_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 50 <= args.size <= 100:
        raise ValueError("Validation size must be between 50 and 100")
    if not args.overwrite and (args.output.exists() or args.manifest.exists()):
        raise FileExistsError("Validation artifacts already exist; pass --overwrite deliberately")

    source_bytes = args.source.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["question_type"])].append((index, row))

    exact = {key: args.size * len(group) / len(rows) for key, group in groups.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    for key in sorted(groups, key=lambda item: exact[item] - allocation[item], reverse=True):
        if sum(allocation.values()) >= args.size:
            break
        allocation[key] += 1

    rng = random.Random(args.seed)
    selected: list[tuple[int, dict]] = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], allocation[key]))
    selected.sort(key=lambda item: item[0])

    output_rows = []
    validation_ids = []
    for index, row in selected:
        sample_id = f"legal-{index + 1:04d}"
        validation_ids.append(sample_id)
        output_rows.append({
            "id": sample_id,
            "dataset": "vietnamese-legal-qa",
            "split": "validation",
            "source_index": index,
            "question": row["question"],
            "ground_truth_answer": row["ground_truth_answer"],
            "question_type": row["question_type"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": args.seed,
        "validation_size": len(output_rows),
        "heldout_size": len(rows) - len(output_rows),
        "validation_question_types": dict(Counter(row["question_type"] for row in output_rows)),
        "validation_ids": validation_ids,
        "policy": "Only validation rows may be used for lambda/tau selection; held-out rows were not materialized or run.",
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
