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
    args = parser.parse_args()
    report = run_eval(args.input, args.output, args.report, load_config(args.config), args.lambda_bind, args.tau, args.sweep, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
