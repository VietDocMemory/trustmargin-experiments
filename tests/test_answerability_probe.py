import json
from pathlib import Path

import numpy as np

from trustmargin_exp.answerability_probe import bare_prompt_ids
from trustmargin_exp.config import ROOT, load_config


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": False,
        }
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode("utf-8"))}


def test_bare_prompt_matches_training_layout_and_truncates_only_passage():
    tokenizer = TinyTokenizer()
    full, truncated = bare_prompt_ids(tokenizer, "context", "question", 100)
    assert truncated is False
    assert bytes(full).decode() == "<user>context\n\nquestion</user><assistant>"

    shortened, truncated = bare_prompt_ids(tokenizer, "x" * 100, "question", 45)
    rendered = bytes(shortened).decode()
    assert truncated is True
    assert len(shortened) <= 45
    assert "\n\nquestion</user><assistant>" in rendered


def test_default_config_enables_frozen_probe_in_shadow_mode():
    config = load_config()
    assert config.answerability_probe_enabled is True
    assert config.answerability_probe_mode == "shadow"
    assert config.answerability_probe_dir == ROOT / "artifacts/answerability_probe"
    exported = json.loads((config.answerability_probe_dir / "config.json").read_text())
    assert exported["model"] == config.base_model
    assert exported["hidden_state_index"] == 21
    assert exported["threshold"] == 0.45405974088596857
    with np.load(config.answerability_probe_dir / "probe.npz") as arrays:
        assert set(arrays.files) >= {"weight", "bias", "mean", "scale"}
        assert arrays["weight"].size == 2560
