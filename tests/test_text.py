import unittest

from trustmargin_exp.text import ABSTENTION, looks_vietnamese, normalize_abstention


class TextNormalizationTests(unittest.TestCase):
    def test_vietnamese_refusal_is_canonicalized(self):
        answer, changed = normalize_abstention("Không có thông tin này trong tài liệu.")
        self.assertEqual(answer, ABSTENTION)
        self.assertTrue(changed)

    def test_english_refusal_is_canonicalized(self):
        answer, changed = normalize_abstention(
            "The provided context does not contain the answer."
        )
        self.assertEqual(answer, ABSTENTION)
        self.assertTrue(changed)

    def test_real_answer_is_preserved(self):
        answer, changed = normalize_abstention("Mức phạt là 10 triệu đồng.")
        self.assertEqual(answer, "Mức phạt là 10 triệu đồng.")
        self.assertFalse(changed)
        self.assertTrue(looks_vietnamese(answer))


if __name__ == "__main__":
    unittest.main()
