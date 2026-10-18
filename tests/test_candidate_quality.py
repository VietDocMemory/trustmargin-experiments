import unittest

from trustmargin_exp.candidate_quality import has_repetition_loop


class CandidateQualityTests(unittest.TestCase):
    def test_detects_repeated_single_word_and_phrase(self):
        self.assertTrue(has_repetition_loop("dịch vụ, " * 10))
        self.assertTrue(has_repetition_loop("quy hoạch môi trường, " * 6))

    def test_preserves_normal_legal_repetition(self):
        answer = (
            "Quy chuẩn kỹ thuật môi trường là quy định bắt buộc áp dụng. "
            "Tiêu chuẩn môi trường là quy định tự nguyện áp dụng."
        )
        self.assertFalse(has_repetition_loop(answer))


if __name__ == "__main__":
    unittest.main()
