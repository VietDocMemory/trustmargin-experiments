"""User-specified experimental D2L versus RAG TrustMargin extension."""
from __future__ import annotations

import math

from .candidate_quality import has_repetition_loop


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


def arbitrate_candidate(row: dict, lambda_bind: float, tau: float) -> dict:
    """Apply the TrustMargin score and reject a looping D2L candidate."""
    decision = arbitrate(row["scores"], lambda_bind, tau)
    diagnostics = row.get("candidate_diagnostics", {}).get("d2l", {})
    looping = bool(diagnostics.get("repetition_loop")) or has_repetition_loop(row["d2l_answer"])
    decision["d2l_repetition_loop"] = looping
    decision["selection_override_reason"] = None
    if looping and decision["selected_source"] == "d2l":
        decision["selected_source"] = "rag"
        decision["selection_override_reason"] = "d2l_repetition_loop"
    return decision
