import argparse
import json
from pathlib import Path

from trustmargin_exp.config import ROOT, load_config
from trustmargin_exp.runner import run_eval


def main():
    parser = argparse.ArgumentParser(description="Oracle, TrustMargin, metrics, optional validation sweep")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/scored_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/results.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/report.json")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lambda-bind", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--sweep", action="store_true", help="Requires split=validation on every sample")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log the completed evaluation to W&B")
    parser.add_argument("--wandb-project", default="trustmargin-d2l-rag")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-artifact-file", type=Path, action="append", default=[])
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_eval(args.input, args.output, args.report, config, args.lambda_bind, args.tau, args.sweep, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.wandb:
        from trustmargin_exp.tracking import log_experiment

        url = log_experiment(args.output, args.report, config, project=args.wandb_project,
                             entity=args.wandb_entity, group=args.wandb_group,
                             mode=args.wandb_mode, extra_files=[args.input, *args.wandb_artifact_file])
        print(url or "W&B run saved locally in offline mode under outputs/wandb")


if __name__ == "__main__":
    main()
