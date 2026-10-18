import unittest
from types import SimpleNamespace

from trustmargin_exp.model_backend import GemmaBackend


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token) for token in token_ids)


class GenerationRetryTests(unittest.TestCase):
    def make_backend(self, first_answer, retry_answer):
        backend = object.__new__(GemmaBackend)
        backend.config = SimpleNamespace(max_new_tokens=256, max_retry_new_tokens=384)
        backend.tokenizer = CharacterTokenizer()
        calls = []

        def fake_generate_once(prompt_ids, max_new_tokens, adapter=None, anti_repeat=False):
            calls.append((max_new_tokens, adapter, anti_repeat))
            answer = retry_answer if len(calls) > 1 else first_answer
            return [ord(character) for character in answer], True

        backend._generate_once = fake_generate_once
        return backend, calls

    def test_looping_d2l_is_regenerated_with_anti_repeat(self):
        backend, calls = self.make_backend("dịch vụ, " * 20, "Quy chuẩn bắt buộc; tiêu chuẩn tự nguyện.")
        answer = backend._generate([1, 2], adapter=object())
        self.assertEqual(answer, "Quy chuẩn bắt buộc; tiêu chuẩn tự nguyện.")
        self.assertEqual([(limit, anti_repeat) for limit, _, anti_repeat in calls], [(256, False), (384, True)])
        self.assertTrue(backend.last_generation["repetition_retry"])
        self.assertFalse(backend.last_generation["repetition_loop"])

    def test_normal_d2l_does_not_retry(self):
        backend, calls = self.make_backend("Quy chuẩn bắt buộc; tiêu chuẩn tự nguyện.", "unused")
        backend._generate([1, 2], adapter=object())
        self.assertEqual(len(calls), 1)
        self.assertFalse(backend.last_generation["repetition_retry"])

    def test_failed_retry_remains_flagged_for_arbitration(self):
        backend, calls = self.make_backend("dịch vụ, " * 20, "dịch vụ, " * 30)
        backend._generate([1, 2], adapter=object())
        self.assertEqual(len(calls), 2)
        self.assertTrue(backend.last_generation["repetition_loop"])


if __name__ == "__main__":
    unittest.main()
