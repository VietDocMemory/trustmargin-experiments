"""Log cached evaluation outputs without rerunning retrieval or GPU inference."""
import argparse
from pathlib import Path

from trustmargin_exp.config import ROOT, load_config
from trustmargin_exp.tracking import log_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/results.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/report.json")
    parser.add_argument("--artifact-file", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project", default="trustmargin-d2l-rag")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--name")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    url = log_experiment(
        args.results, args.report, load_config(args.config), project=args.project,
        entity=args.entity, group=args.group, name=args.name, mode=args.mode,
        extra_files=args.artifact_file,
    )
    print(url or "W&B run saved locally in offline mode under outputs/wandb")


if __name__ == "__main__":
    main()
