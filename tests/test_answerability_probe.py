import json
from pathlib import Path

import numpy as np
import pytest

from trustmargin_exp.answerability_probe import AnswerabilityProbe, bare_prompt_ids
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


class FakeModel:
    def __call__(self, input_ids, **kwargs):
        torch = pytest.importorskip("torch")
        batch_size, sequence_length = input_ids.shape
        hidden = torch.zeros((batch_size, sequence_length, 2), dtype=torch.float32)
        # The last non-padding token in each prompt controls the first feature.
        for row, token_index in enumerate(kwargs["attention_mask"].sum(dim=1) - 1):
            hidden[row, token_index, 0] = float(row + 1)
        return type("Output", (), {"hidden_states": (hidden, hidden)})()


def test_probe_max_pools_top8_single_and_top2_combined(tmp_path):
    torch = pytest.importorskip("torch")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model": "fake-model",
                "prompt_style": "bare",
                "pooling": "last",
                "hidden_state_index": 1,
                "hidden_size": 2,
                "max_len": 100,
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        tmp_path / "probe.npz",
        weight=np.array([1.0, 0.0], dtype=np.float32),
        bias=np.array([0.0], dtype=np.float32),
        mean=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
    )
    probe = AnswerabilityProbe(tmp_path, "fake-model", torch, "cpu")

    result = probe.score(
        FakeModel(),
        TinyTokenizer(),
        "question",
        ["chunk one", "chunk two", "chunk three"],
        torch,
    )

    assert result["aggregation"] == "max_top8_single_plus_top2_combined"
    assert len(result["component_scores"]) == 4
    assert [item["kind"] for item in result["component_scores"]] == [
        "single",
        "single",
        "single",
        "top2_combined",
    ]
    assert result["winning_kind"] == "top2_combined"
    assert result["winning_chunk_indices"] == [1, 2]
    assert result["prob_answerable"] == result["component_scores"][-1]["prob_answerable"]
