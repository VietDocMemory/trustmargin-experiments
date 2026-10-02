import argparse
from pathlib import Path

from trustmargin_exp.config import ROOT, load_config
from trustmargin_exp.runner import generate_candidates


def main():
    parser = argparse.ArgumentParser(description="Retrieve, then generate and cache both candidates")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--document", type=str, help="Shared .txt or .pdf document; rows may provide document_path instead")
    parser.add_argument("--collection", required=True, help="Existing Qdrant collection/session ID")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/candidates.jsonl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--document-source",
        choices=("provided", "retrieved"),
        default="provided",
        help="Use a supplied document or the label-free retrieved passages as bounded D2L input",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = generate_candidates(
        args.input, args.output, load_config(args.config), args.collection,
        args.document, args.limit, args.overwrite, args.document_source,
    )
    print(f"Wrote {len(rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
