#!/usr/bin/env python3
"""Semantic search over the restaurant vectors.

Usage:
  python query.py "cozy ramen spot for a date night" --city Philadelphia
  python query.py "cheap tacos" --qdrant-url http://localhost:6333 --limit 5
"""
from __future__ import annotations

import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from embed_load import embed_batch  # reuse the HF inference helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--city", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--collection", default="restaurants")
    ap.add_argument("--local-path", default="./qdrant_data")
    args = ap.parse_args()

    vec = embed_batch([args.query])[0]
    client = (QdrantClient(url=args.qdrant_url)
              if args.qdrant_url else QdrantClient(path=args.local_path))
    qfilter = (Filter(must=[FieldCondition(key="city", match=MatchValue(value=args.city))])
               if args.city else None)
    hits = client.query_points(args.collection, query=vec,
                               query_filter=qfilter, limit=args.limit).points
    for h in hits:
        p = h.payload
        print(f"{h.score:.3f}  {p['name']}  ({p['city']}, {p['stars']}*, {p['review_count']} reviews)")
        print(f"       {p['categories']}")
    if not hits:
        print("(no results -- is the collection loaded? run embed_load.py --all first)")


if __name__ == "__main__":
    main()
