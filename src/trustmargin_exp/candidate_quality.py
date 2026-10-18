"""Conservative detection of degenerate loops in D2L answers."""
from __future__ import annotations

import re
import unicodedata


def has_repetition_loop(answer: str) -> bool:
    """Detect a short phrase repeated consecutively many times.

    This intentionally ignores ordinary repeated legal terms in separate clauses.
    """
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    words = re.findall(r"\w+", normalized, flags=re.UNICODE)
    for width in range(1, 5):
        minimum = 8 if width == 1 else max(5, (12 + width - 1) // width)
        for start in range(len(words) - width * minimum + 1):
            phrase = words[start:start + width]
            if all(
                words[start + width * index:start + width * (index + 1)] == phrase
                for index in range(1, minimum)
            ):
                return True
    return False
