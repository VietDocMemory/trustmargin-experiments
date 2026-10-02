"""Deterministic QA metrics and candidate complementarity analysis."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from statistics import mean

from .text import ABSTENTION


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    return float(any(normalize(prediction) == normalize(gold) for gold in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    pred = normalize(prediction).split()
    def score(gold: str) -> float:
        truth = normalize(gold).split()
        if not pred or not truth:
            return float(pred == truth)
        overlap = sum((Counter(pred) & Counter(truth)).values())
        return 2 * overlap / (len(pred) + len(truth))
    return max((score(gold) for gold in golds), default=0.0)


def rouge_l_f1(prediction: str, golds: list[str]) -> float:
    # Uses D2L's ROUGE-L metric dependency, without importing its training pipeline.
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return max((scorer.score(gold, prediction)["rougeL"].fmeasure for gold in golds), default=0.0)


def sample_metrics(prediction: str, golds: list[str], unanswerable: bool = False) -> dict:
    result = {"exact_match": exact_match(prediction, golds), "token_f1": token_f1(prediction, golds)}
    rouge = rouge_l_f1(prediction, golds)
    if rouge == rouge:  # NaN means optional package unavailable.
        result["rouge_l_f1"] = rouge
    if unanswerable:
        result["abstention_accuracy"] = float(normalize(prediction) == normalize(ABSTENTION))
    return result


def summarize(
    rows: list[dict],
    lambda_bind: float,
    tau: float,
    both_wrong_f1_threshold: float = 0.2,
) -> dict:
    from .trustmargin import arbitrate

    sums = {source: Counter() for source in ("d2l", "rag", "oracle", "tm")}
    counts = Counter()
    for row in rows:
        golds = row["gold_answers"]
        unanswerable = row.get("question_type") == "unanswerable"
        d = sample_metrics(row["d2l_answer"], golds, unanswerable)
        r = sample_metrics(row["rag_answer"], golds, unanswerable)
        # Oracle uses token F1, then EM; ties deterministically prefer D2L.
        d_key, r_key = (d["token_f1"], d["exact_match"]), (r["token_f1"], r["exact_match"])
        oracle = r if r_key > d_key else d
        selected = arbitrate(row["scores"], lambda_bind, tau)["selected_source"]
        tm = r if selected == "rag" else d
        for source, metrics in (("d2l", d), ("rag", r), ("oracle", oracle), ("tm", tm)):
            sums[source].update(metrics)
        counts["rag_better" if r_key > d_key else "d2l_better" if d_key > r_key else "tie"] += 1
        counts["both_wrong"] += int(
            d["token_f1"] < both_wrong_f1_threshold
            and r["token_f1"] < both_wrong_f1_threshold
        )
        counts["both_exact_match_wrong"] += int(
            d["exact_match"] == 0 and r["exact_match"] == 0
        )
        counts["unanswerable"] += int(unanswerable)
    n = len(rows)
    if n == 0:
        raise ValueError("No scored rows")
    averages = {source: {key: value / (counts["unanswerable"] if key == "abstention_accuracy" else n) for key, value in values.items()} for source, values in sums.items()}
    gain = averages["oracle"]["token_f1"] - max(averages["d2l"]["token_f1"], averages["rag"]["token_f1"])
    for key in (
        "rag_better", "d2l_better", "tie", "both_wrong",
        "both_exact_match_wrong", "unanswerable",
    ):
        counts[key] += 0
    selection_counts = Counter(
        arbitrate(row["scores"], lambda_bind, tau)["selected_source"] for row in rows
    )
    latency_keys = sorted({
        key
        for row in rows
        for key, value in row.get("latency", {}).items()
        if value is not None
    })
    latency = {
        key: mean(
            float(row["latency"][key])
            for row in rows
            if row.get("latency", {}).get(key) is not None
        )
        for key in latency_keys
    }
    diagnostic_summary = Counter()
    for row in rows:
        for source, values in row.get("candidate_diagnostics", {}).items():
            for key in (
                "initially_truncated", "retried", "final_truncated",
                "abstention_normalized", "looks_vietnamese",
            ):
                diagnostic_summary[f"{source}_{key}"] += int(bool(values.get(key)))
    return {
        "n": n,
        "metrics": averages,
        "oracle_counts": dict(counts),
        "selection_counts": {
            "d2l": selection_counts["d2l"],
            "rag": selection_counts["rag"],
        },
        "mean_latency_seconds": latency,
        "candidate_diagnostics": dict(diagnostic_summary),
        "both_wrong_f1_threshold": both_wrong_f1_threshold,
        "oracle_gain_over_best_single_source_f1": gain,
        "warning": "Low candidate complementarity; TrustMargin has little headroom" if gain < 0.01 else None,
    }
