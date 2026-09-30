"""Phase 2: make the chat actually semantic.

Pipeline: the deterministic parser + data.search narrow the candidate set
with hard constraints (city, cuisine, price, open late); the Qdrant vector
then ranks those candidates by feel. Neither alone nails a phrase like
"cozy ramen spot for date night" -- together they do.

Degrades gracefully: when the vector backend (Qdrant collection or HF
token for query embeddings) is unavailable, deterministic order is kept
and vibe_applied=False is reported.
"""
from __future__ import annotations

import logging

from qdrant_client.models import FieldCondition, Filter, MatchAny

import data
import parse
import vibe

log = logging.getLogger(__name__)

# Within the narrowed candidate set every pick already satisfies the hard
# constraints, so vibe dominates the final order; the deterministic score
# survives as a quality tiebreak (stars/reviews driven).
DET_WEIGHT = 0.3
VIBE_WEIGHT = 0.7
CANDIDATE_LIMIT = 50


def rerank_by_vibe(text: str, filters: dict, limit: int = 3):
    """Return (picks, vibe_applied).

    picks: top-`limit` picks in data.search()'s shape, reordered by the
    blended deterministic+vector score. Each pick keeps its fields and
    gains vibe_score (raw cosine similarity).
    """
    kw = {k: filters[k] for k in parse._SEARCH_KEYS}
    kw["open_only"] = True
    candidates = data.search(limit=CANDIDATE_LIMIT, **kw)
    if not candidates:
        return [], False
    try:
        vec_scores = _vector_scores(
            text, [p["record"]["business_id"] for p in candidates])
    except Exception as e:  # vector backend unavailable -> deterministic
        log.warning("vibe rerank skipped: %s", e)
        return candidates[:limit], False
    if not vec_scores:
        return candidates[:limit], False

    max_det = max(p["score"] for p in candidates) or 1.0
    ranked = []
    for p in candidates:
        vec = vec_scores.get(p["record"]["business_id"])
        if vec is None:
            continue  # all open venues are embedded; defensive only
        det_norm = p["score"] / max_det
        blended = round(DET_WEIGHT * det_norm + VIBE_WEIGHT * vec, 4)
        ranked.append((blended, vec, p))
    ranked.sort(key=lambda t: t[0], reverse=True)

    out = []
    for blended, vec, p in ranked[:limit]:
        p = dict(p)  # don't mutate the cached search rows
        p["score"] = blended
        p["vibe_score"] = round(vec, 4)
        out.append(p)
    return out, True


def _vector_scores(text: str, business_ids: list[str]) -> dict[str, float]:
    client = vibe.get_client()
    coll = vibe.collection_name()
    if not client.collection_exists(coll):
        raise RuntimeError(
            f"Qdrant collection '{coll}' not found -- run vector/embed_load.py first"
        )
    vec = vibe.embed_query(text)
    qfilter = Filter(must=[
        FieldCondition(key="business_id", match=MatchAny(any=business_ids))
    ])
    hits = client.query_points(
        coll, query=vec, query_filter=qfilter,
        limit=len(business_ids)).points
    return {h.payload.get("business_id"): float(h.score) for h in hits}
