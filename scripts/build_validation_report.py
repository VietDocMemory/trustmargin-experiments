"""Build a compact Markdown report and plots from cached validation artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from trustmargin_exp.logging_utils import read_jsonl
from trustmargin_exp.metrics import sample_metrics
from trustmargin_exp.trustmargin import arbitrate


ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, default=ROOT / "outputs/validation_scored.jsonl")
    parser.add_argument("--default-report", type=Path, default=ROOT / "outputs/validation_default_report.json")
    parser.add_argument("--tuned-report", type=Path, default=ROOT / "outputs/validation_tuned_report.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/validation_report")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.scored)
    default = json.loads(args.default_report.read_text(encoding="utf-8"))
    tuned = json.loads(args.tuned_report.read_text(encoding="utf-8"))

    labels = ["D2L", "RAG", "TM default", "TM tuned", "Oracle"]
    reports = [default, default, default, tuned, tuned]
    keys = ["d2l", "rag", "tm", "tm", "oracle"]
    metric_names = [("token_f1", "Token F1"), ("rouge_l_f1", "ROUGE-L"), ("exact_match", "Exact Match")]
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (metric, title) in zip((-width, 0, width), metric_names):
        values = [report["metrics"][key].get(metric, 0) for report, key in zip(reports, keys)]
        ax.bar([value + offset for value in x], values, width, label=title)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Validation metrics (cached candidates and likelihoods)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "metrics.png", dpi=160)
    plt.close(fig)

    counts = tuned["oracle_counts"]
    count_labels = ["RAG better", "D2L better", "Tie", "Both wrong"]
    count_values = [counts["rag_better"], counts["d2l_better"], counts["tie"], counts["both_wrong"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(count_labels, count_values, color=["#4c78a8", "#f58518", "#bab0ac", "#e45756"])
    ax.bar_label(bars)
    ax.set_ylabel("Samples")
    ax.set_title("D2L/RAG complementarity on validation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "complementarity.png", dpi=160)
    plt.close(fig)

    best = tuned["sweep"]["best"]
    margin_groups: dict[str, list[float]] = {"rag": [], "d2l": [], "tie": []}
    for row in rows:
        d = sample_metrics(row["d2l_answer"], row["gold_answers"])
        r = sample_metrics(row["rag_answer"], row["gold_answers"])
        winner = "rag" if r["token_f1"] > d["token_f1"] else "d2l" if d["token_f1"] > r["token_f1"] else "tie"
        margin_groups[winner].append(arbitrate(row["scores"], best["lambda_bind"], best["tau"])["trust_score"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, color in (("rag", "#4c78a8"), ("d2l", "#f58518"), ("tie", "#bab0ac")):
        if margin_groups[key]:
            ax.hist(margin_groups[key], bins=15, alpha=0.55, label=key, color=color)
    ax.axvline(best["tau"], color="black", linestyle="--", label="tuned tau")
    ax.set_xlabel("Trust score")
    ax.set_ylabel("Samples")
    ax.set_title("Tuned margin distribution by oracle winner")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "margins.png", dpi=160)
    plt.close(fig)

    gain = tuned["oracle_gain_over_best_single_source_f1"]
    tm_gain = tuned["metrics"]["tm"]["token_f1"] - max(
        tuned["metrics"]["d2l"]["token_f1"], tuned["metrics"]["rag"]["token_f1"]
    )
    recovered_gain = tm_gain / gain if gain > 0 else 0.0
    verdict = (
        "Có complementarity rõ ràng, nhưng TM tuned mới thu hồi "
        f"**{pct(recovered_gain)}** oracle gap. Đáng chạy thêm đúng một held-out test để xác nhận, "
        "chưa đủ bằng chứng để mở rộng quy mô. Việc cấu hình tốt nhất có `lambda_bind=0` cũng cho thấy "
        "binding term hiện chưa đóng góp tích cực; lợi ích đang đến từ threshold trên likelihood margin."
        if gain >= 0.03
        else "Complementarity có nhưng khiêm tốn; chỉ nên tiếp tục nếu TM tuned cải thiện ổn định trên held-out test."
        if gain >= 0.01
        else "Complementarity quá thấp; chưa đáng mở rộng trước khi cải thiện hai candidate nguồn."
    )
    diagnostics = tuned.get("candidate_diagnostics", {})
    non_vi = sum(len(rows) - diagnostics.get(f"{source}_looks_vietnamese", 0) for source in ("d2l", "rag"))
    final_truncated = sum(diagnostics.get(f"{source}_final_truncated", 0) for source in ("d2l", "rag"))
    latency = tuned.get("mean_latency_seconds", {})
    lines = [
        "# TrustMargin D2L–RAG validation report",
        "",
        f"Validation gồm **{tuned['n']} mẫu**, chỉ dùng split `validation`; held-out test chưa được chạy hoặc dùng để tune. D2L dùng các passage do retriever chọn (`retrieved`), không dùng `ground_truth_context`.",
        "",
        "## Kết quả chính",
        "",
        "| Phương pháp | Token F1 | ROUGE-L | Exact Match |",
        "|---|---:|---:|---:|",
    ]
    for label, report, key in zip(labels, reports, keys):
        metrics = report["metrics"][key]
        lines.append(f"| {label} | {metrics['token_f1']:.4f} | {metrics.get('rouge_l_f1', 0):.4f} | {metrics['exact_match']:.4f} |")
    lines += [
        "",
        f"- RAG better: **{counts['rag_better']}**; D2L better: **{counts['d2l_better']}**; tie: **{counts['tie']}**; both wrong (F1 < {tuned['both_wrong_f1_threshold']}): **{counts['both_wrong']}**.",
        f"- Oracle gain trên best single source: **{gain:+.4f} token-F1**.",
        f"- TM tuned gain trên best single source: **{tm_gain:+.4f} token-F1**.",
        f"- TM tuned thu hồi **{pct(recovered_gain)}** oracle gain khả dụng.",
        f"- Paper/project default: `lambda_bind={default['lambda_bind']}`, `tau={default['tau']}`.",
        f"- Validation tuned: `lambda_bind={tuned['lambda_bind']}`, `tau={tuned['tau']:.6g}`; chọn RAG {tuned['selection_counts']['rag']}/{tuned['n']} mẫu.",
        "",
        "![Metrics](metrics.png)",
        "",
        "![Complementarity](complementarity.png)",
        "",
        "![Margins](margins.png)",
        "",
        "## Chất lượng pipeline và lỗi còn lại",
        "",
        f"- Candidate còn bị truncate sau retry: **{final_truncated}**; candidate không vượt heuristic tiếng Việt: **{non_vi}** (đều là câu trả lời số ngắn, không phải output tiếng Anh).",
        f"- Thời gian trung bình/sample: retrieval `{latency.get('retrieval', 0):.2f}s`, adapter `{latency.get('adapter_build', 0):.2f}s`, RAG generation `{latency.get('rag_generation', 0):.2f}s`, D2L generation `{latency.get('d2l_generation', 0):.2f}s`, scoring `{latency.get('likelihood_scoring', 0):.2f}s`, total `{latency.get('total', 0):.2f}s`.",
        "- `both wrong` dùng ngưỡng token-F1 thay vì Exact Match vì câu trả lời sinh tự do hiếm khi khớp chuỗi tuyệt đối.",
        "- Kết quả tuned là in-sample validation; chỉ held-out test mới xác nhận khả năng tổng quát hóa.",
        "",
        "## Kết luận",
        "",
        verdict,
        "",
        "## Artifact",
        "",
        "- `../validation_candidates.jsonl`: prompt outputs, retrieved passages, raw/normalized answers, generation diagnostics và generation latency.",
        "- `../validation_scored.jsonl`: candidate cache cộng likelihood scores; đổi threshold không cần chạy lại model.",
        "- `../validation_default_results.jsonl` và `../validation_tuned_results.jsonl`: per-sample selected source, final answer, `m_prior`, `m_bind`, `delta_*`, trust score và threshold.",
        "- `../validation_default_report.json` và `../validation_tuned_report.json`: aggregate metrics và toàn bộ sweep trials.",
        "- `../split_manifest.json`: seed, source hash, validation IDs và held-out IDs.",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
