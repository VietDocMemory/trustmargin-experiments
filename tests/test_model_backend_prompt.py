from trustmargin_exp.model_backend import GemmaBackend


class RecordingTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return list(range(len("\n".join(message["content"] for message in messages))))

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return bytes(token_ids).decode("utf-8", errors="ignore")


def backend_with_recording_tokenizer():
    backend = object.__new__(GemmaBackend)
    backend.tokenizer = RecordingTokenizer()
    return backend


def test_rag_prompt_uses_new_grounding_instruction_without_citations():
    backend = backend_with_recording_tokenizer()

    backend.prompt("Câu hỏi thử?", ["Bằng chứng một", "Bằng chứng hai"])

    messages = backend.tokenizer.messages
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Bạn là trợ lý trả lời câu hỏi bằng tiếng Việt." in messages[0]["content"]
    assert "không suy đoán" in messages[0]["content"]
    assert "chỉ nêu các thông tin cần" in messages[0]["content"]
    assert "không chép toàn bộ" in messages[0]["content"]
    assert "điều kiện, ngoại lệ, con số" in messages[0]["content"]
    assert "[Nguồn" not in messages[0]["content"]
    assert "NGỮ CẢNH:" in messages[1]["content"]
    assert "Bằng chứng một\n\nBằng chứng hai" in messages[1]["content"]
    assert "[Nguồn" not in messages[1]["content"]
    assert "CÂU HỎI:\nCâu hỏi thử?" in messages[1]["content"]


def test_d2l_prompt_remains_question_only():
    backend = backend_with_recording_tokenizer()

    backend.prompt("Câu hỏi D2L?")

    messages = backend.tokenizer.messages
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "NGỮ CẢNH:" not in messages[1]["content"]
    assert "tài liệu đã được ghi vào bộ nhớ" in messages[0]["content"]
    assert "Không lặp từ, lặp cụm từ" in messages[0]["content"]
    assert "CÂU HỎI:\nCâu hỏi D2L?" in messages[1]["content"]
