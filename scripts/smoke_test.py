"""Run three to five real samples and check all experiment invariants."""
import argparse
import math
from pathlib import Path

from trustmargin_exp.config import ROOT, load_config
from trustmargin_exp.runner import generate_candidates, run_eval, score_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--limit", type=int, default=3, choices=(3, 4, 5))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    candidates = generate_candidates(args.input, ROOT / "outputs/smoke_candidates.jsonl", config, args.collection, args.document, args.limit, True)
    scored = score_candidates(ROOT / "outputs/smoke_candidates.jsonl", ROOT / "outputs/smoke_scored.jsonl", config, True)
    report = run_eval(ROOT / "outputs/smoke_scored.jsonl", ROOT / "outputs/smoke_results.jsonl", ROOT / "outputs/smoke_report.json", config, overwrite=True)
    assert len(candidates) == len(scored) == args.limit
    for row in scored:
        assert row["retrieved_contexts"]
        assert row["rag_answer"] and row["d2l_answer"]
        assert all(math.isfinite(value) for view in row["scores"].values() for value in view.values())
    from trustmargin_exp.logging_utils import read_jsonl
    outputs = read_jsonl(ROOT / "outputs/smoke_results.jsonl")
    for row in outputs:
        assert all(math.isfinite(row[key]) for key in ("m_prior", "m_bind", "trust_score"))
        assert row["final_answer"] == row[f'{row["selected_source"]}_answer']
    first = outputs[0]
    print(f'QUESTION:\n{first["question"]}\n\nD2L:\n{first["d2l_answer"]}\n\nRAG:\n{first["rag_answer"]}')
    print(f'\nM_prior: {first["m_prior"]:.4f}\nM_bind: {first["m_bind"]:.4f}\nM: {first["trust_score"]:.4f}\nSELECTED: {first["selected_source"]}')
    print(f'Oracle F1: {report["metrics"]["oracle"]["token_f1"]:.4f}; wrote {len(outputs)} JSONL records')
    peaks = [row["memory"].get("peak_allocated_mb") for row in outputs]
    print(f"GPU peak allocated MB per sample: {peaks}; inspect for growth")
    for phase in ("generation", "scoring"):
        allocated = [row["memory"].get(phase, {}).get("allocated_after_mb") for row in outputs]
        if all(value is not None for value in allocated):
            growth = max(allocated) - min(allocated)
            print(f"GPU {phase} allocated-after range: {growth:.1f} MB")
            assert growth < max(512, 0.1 * allocated[0]), f"Possible GPU memory growth during {phase}"


if __name__ == "__main__":
    main()
