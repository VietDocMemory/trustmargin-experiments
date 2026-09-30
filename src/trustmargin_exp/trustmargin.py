"""User-specified experimental D2L versus RAG TrustMargin extension."""
from __future__ import annotations

import math


def arbitrate(scores: dict, lambda_bind: float = 0.5, tau: float = -1.5) -> dict:
    for view in ("q_only", "q_context", "context_only"):
        for source in ("d2l", "rag"):
            if not math.isfinite(scores[view][source]):
                raise ValueError(f"Non-finite score: {view}.{source}")
    m_prior = scores["q_only"]["rag"] - scores["q_only"]["d2l"]
    delta_d2l = scores["q_context"]["d2l"] - scores["context_only"]["d2l"]
    delta_rag = scores["q_context"]["rag"] - scores["context_only"]["rag"]
    m_bind = delta_rag - delta_d2l
    trust_score = m_prior + lambda_bind * m_bind
    selected = "rag" if trust_score > tau else "d2l"
    return {
        "m_prior": m_prior,
        "delta_d2l": delta_d2l,
        "delta_rag": delta_rag,
        "m_bind": m_bind,
        "trust_score": trust_score,
        "lambda_bind": lambda_bind,
        "tau": tau,
        "selected_source": selected,
    }
