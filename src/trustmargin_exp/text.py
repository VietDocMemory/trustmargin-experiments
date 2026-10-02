"""Shared answer-language and abstention normalization helpers."""
from __future__ import annotations

import re
import unicodedata


ABSTENTION = "Tài liệu không đề cập."


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    return " ".join(
        "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        .replace("đ", "d")
        .split()
    )


_ABSTENTION_PATTERNS = (
    r"tai lieu (?:nay )?khong de cap",
    r"(?:tai lieu|van ban|ngu canh).*(?:khong co|khong cung cap|khong neu|khong tim thay).*(?:thong tin|cau tra loi|noi dung)",
    r"khong co (?:du|bat ky )?thong tin",
    r"khong du thong tin",
    r"thong tin (?:nay )?khong (?:duoc )?(?:de cap|cung cap|neu)",
    r"provided (?:text|context|document).*(?:does not|doesn't|do not|don't).*(?:contain|mention|provide)",
    r"(?:not mentioned|no information|cannot be answered).*(?:context|document|text)?",
)


def normalize_abstention(answer: str) -> tuple[str, bool]:
    """Map common refusal variants to the one evaluation-safe Vietnamese phrase."""
    stripped = answer.strip()
    folded = _fold(stripped)
    if any(re.search(pattern, folded) for pattern in _ABSTENTION_PATTERNS):
        return ABSTENTION, stripped != ABSTENTION
    return stripped, False


def looks_vietnamese(text: str) -> bool:
    """Cheap diagnostic only; prompts enforce language, this does not alter answers."""
    folded = _fold(text)
    words = set(re.findall(r"\w+", folded, flags=re.UNICODE))
    markers = {"la", "cua", "va", "trong", "theo", "khong", "duoc", "tai", "lieu", "quy", "dinh"}
    return len(words & markers) >= 2 or bool(re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", text.casefold()))
