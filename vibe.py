"""Semantic search over the Qdrant restaurant vectors.

Used by POST /vibe. Filters are deterministic (city, min stars); the vector
query ranks by vibe. Embeddings come from the HF Inference API -- the server
needs HF_TOKEN set (or the hosted custom.huggingface connector).

Env:
  QDRANT_URL        http://localhost:6333 for Docker (preferred)
  QDRANT_PATH       local file mode fallback (default: vector/qdrant_data)
  QDRANT_COLLECTION default: restaurants
  HF_TOKEN          Hugging Face token for query embeddings
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

HF_MODEL = "BAAI/bge-small-en-v1.5"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
ALLOWED = ("huggingface.co", "api-inference.huggingface.co", "router.huggingface.co")

_clients: dict[str, QdrantClient] = {}


def get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    path = os.environ.get("QDRANT_PATH", "vector/qdrant_data")
    key = url or f"path:{path}"
    if key not in _clients:
        _clients[key] = QdrantClient(url=url) if url else QdrantClient(path=path)
    return _clients[key]


def collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", "restaurants")


def embed_query(text: str) -> list[float]:
    body = json.dumps({"inputs": [text]}).encode()
    req = urllib.request.Request(
        HF_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    else:
        try:
            sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
            from dynamic_credentials import add_surrogate_to_request
            add_surrogate_to_request(req, "custom.huggingface", allowed_hosts=ALLOWED)
        except Exception as e:
            raise RuntimeError(
                "HF_TOKEN is not set and no hosted HF connector is available"
            ) from e
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF inference error: {data['error']}")
    return data[0]


def vibe_search(query: str, city: str | None = None,
                min_stars: float = 0.0, limit: int = 5) -> list[dict]:
    """Return ranked hits: score + payload (business_id, name, city, ...)."""
    client = get_client()
    coll = collection_name()
    if not client.collection_exists(coll):
        raise RuntimeError(
            f"Qdrant collection '{coll}' not found -- run vector/embed_load.py first"
        )
    vec = embed_query(query)
    must = []
    if city:
        must.append(FieldCondition(key="city", match=MatchValue(value=city)))
    if min_stars > 0:
        must.append(FieldCondition(key="stars", range=Range(gte=min_stars)))
    qfilter = Filter(must=must) if must else None
    hits = client.query_points(coll, query=vec, query_filter=qfilter, limit=limit).points
    return [
        {
            "business_id": h.payload.get("business_id"),
            "name": h.payload.get("name"),
            "city": h.payload.get("city"),
            "stars": h.payload.get("stars"),
            "review_count": h.payload.get("review_count"),
            "categories": h.payload.get("categories"),
            "address": h.payload.get("address"),
            "lat": h.payload.get("latitude"),
            "lng": h.payload.get("longitude"),
            "score": round(float(h.score), 4),
        }
        for h in hits
    ]
