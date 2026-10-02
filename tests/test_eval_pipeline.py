import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trustmargin_exp.runner import run_eval


def row(sample_id, split="validation"):
    return {
        "id": sample_id, "candidate_ids": {"d2l": sample_id, "rag": sample_id},
        "dataset": "synthetic", "split": split, "question": "What?", "gold_answers": ["correct"],
        "retrieved_contexts": [{"content": "evidence", "score": 0.8, "page": 1, "chunk_index": 0}],
        "d2l_answer": "incorrect", "rag_answer": "correct",
        "scores": {
            "q_only": {"d2l": -2.0, "rag": -1.0},
            "q_context": {"d2l": -2.0, "rag": -1.0},
            "context_only": {"d2l": -2.0, "rag": -2.0},
        },
        "latency": {"retrieval": 0.1, "adapter_build": 0.2, "rag_generation": 0.3, "d2l_generation": 0.4, "likelihood_scoring": 0.5, "total": 1.5},
        "memory": {"peak_allocated_mb": None, "peak_reserved_mb": None},
    }


class EvalPipelineTests(unittest.TestCase):
    def test_oracle_and_rearbitration_use_cached_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scored.jsonl"
            source.write_text(json.dumps(row("q1")) + "\n", encoding="utf-8")
            config = SimpleNamespace(lambda_bind=0.5, tau=-1.5)
            report = run_eval(source, root / "results.jsonl", root / "report.json", config)
            self.assertEqual(report["metrics"]["oracle"]["exact_match"], 1.0)
            self.assertEqual(report["oracle_counts"]["rag_better"], 1)
            result = json.loads((root / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["selected_source"], "rag")
            self.assertEqual(result["final_answer"], "correct")
            run_eval(source, root / "results.jsonl", root / "report.json", config, tau=2.0, overwrite=True)
            changed = json.loads((root / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(changed["selected_source"], "d2l")
            self.assertEqual(json.loads(source.read_text(encoding="utf-8"))["rag_answer"], "correct")

    def test_sweep_refuses_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scored.jsonl"
            source.write_text(json.dumps(row("q1", split="test")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation"):
                run_eval(source, root / "results.jsonl", root / "report.json", SimpleNamespace(lambda_bind=0.5, tau=-1.5), sweep=True)

    def test_validation_sweep_is_labeled_and_uses_cached_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scored.jsonl"
            source.write_text(json.dumps(row("q1")) + "\n", encoding="utf-8")
            report = run_eval(source, root / "results.jsonl", root / "report.json", SimpleNamespace(lambda_bind=0.5, tau=-1.5), sweep=True)
            self.assertEqual(report["sweep"]["label"], "validation-tuned")
            self.assertGreaterEqual(len(report["sweep"]["trials"]), 42)
            self.assertIn("selected_rag", report["sweep"]["best"])


if __name__ == "__main__":
    unittest.main()
