import math
import unittest

from trustmargin_exp.likelihood import answer_mask, mean_answer_log_probs


class LikelihoodTests(unittest.TestCase):
    def test_prompt_tokens_are_excluded_and_length_normalized(self):
        mask = answer_mask(3, 2)
        self.assertEqual(mask, [False, False, True, True])
        self.assertEqual(mean_answer_log_probs([-100, -100, -1, -3], mask), -2)

    def test_deterministic_and_finite(self):
        values = [-9, -0.25, -0.75]
        first = mean_answer_log_probs(values, answer_mask(2, 2))
        self.assertEqual(first, mean_answer_log_probs(values, answer_mask(2, 2)))
        self.assertTrue(math.isfinite(first))

    def test_empty_answer_rejected(self):
        with self.assertRaises(ValueError):
            answer_mask(2, 0)

    def test_torch_scorer_ignores_prompt_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is not installed in this lightweight test environment")
        from trustmargin_exp.likelihood import teacher_forced_log_likelihood

        class Tokenizer:
            def encode(self, answer, add_special_tokens=False):
                self_no_special = not add_special_tokens
                assert self_no_special
                return [2, 3]

        class Model:
            def __call__(self, input_ids, attention_mask):
                logits = torch.zeros((1, 4, 4))
                logits[0, 0, :] = torch.tensor([100.0, 0, 0, 0])
                logits[0, 1, :] = torch.tensor([0.0, 0, 3.0, 0])
                logits[0, 2, :] = torch.tensor([0.0, 0, 0, 1.0])
                return type("Output", (), {"logits": logits})()

        result = teacher_forced_log_likelihood(Model(), Tokenizer(), [0, 1], "answer", "cpu")
        expected = (
            torch.log_softmax(torch.tensor([0.0, 0, 3.0, 0]), 0)[2]
            + torch.log_softmax(torch.tensor([0.0, 0, 0, 1.0]), 0)[3]
        ) / 2
        self.assertAlmostEqual(result, expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()
