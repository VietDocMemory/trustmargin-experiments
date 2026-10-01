from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from trustmargin_exp.model_backend import GemmaBackend


class AdapterLifecycleTests(unittest.TestCase):
    def test_cached_adapter_repatches_lora_before_generation(self):
        events = []

        class Model:
            def reset(self):
                events.append("reset")

            def patch_lora_forward(self):
                events.append("patch")

            def generate(self, **_kwargs):
                events.append("generate")
                raise RuntimeError("generation stopped for test")

        backend = GemmaBackend.__new__(GemmaBackend)
        backend.model = Model()
        backend.torch = SimpleNamespace(tensor=lambda value, device: value, inference_mode=nullcontext)
        backend.device = "cpu"
        backend.config = SimpleNamespace(max_new_tokens=1)
        backend.tokenizer = SimpleNamespace(pad_token_id=0)

        with self.assertRaisesRegex(RuntimeError, "generation stopped"):
            backend._generate([1], adapter={"cached": "weights"})

        self.assertEqual(events, ["reset", "patch", "generate", "reset"])


if __name__ == "__main__":
    unittest.main()
