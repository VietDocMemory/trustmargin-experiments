import unittest

from trustmargin_exp.runner import normalize_samples, validate_alignment


class CandidateAlignmentTests(unittest.TestCase):
    def test_same_question_id_and_gold(self):
        samples = normalize_samples([
            {"id": "q1", "question": "What?", "ground_truth_answer": "A", "document_text": "document"}
        ], "dataset", "session", None)
        self.assertEqual(samples[0]["gold_answers"], ["A"])
        row = {
            "id": samples[0]["id"], "question": samples[0]["question"],
            "gold_answers": samples[0]["gold_answers"],
            "candidate_ids": {"d2l": "q1", "rag": "q1"},
            "d2l_answer": "A", "rag_answer": "B", "retrieved_contexts": [],
        }
        validate_alignment([row])
        row["candidate_ids"]["rag"] = "q2"
        with self.assertRaises(ValueError):
            validate_alignment([row])

    def test_duplicate_sample_rejected(self):
        rows = [{"id": "x", "question": "Q", "ground_truth_answer": "A", "document_text": "d"}] * 2
        with self.assertRaises(ValueError):
            normalize_samples(rows, "dataset", "session", None)


if __name__ == "__main__":
    unittest.main()
