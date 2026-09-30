import argparse
from pathlib import Path

from trustmargin_exp.config import ROOT, load_config
from trustmargin_exp.runner import score_candidates


def main():
    parser = argparse.ArgumentParser(description="Cache six base-Gemma likelihood views per sample")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/candidates.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scored_candidates.jsonl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = score_candidates(args.input, args.output, load_config(args.config), args.overwrite)
    print(f"Wrote {len(rows)} scored candidates to {args.output}")


if __name__ == "__main__":
    main()
