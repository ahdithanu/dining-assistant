#!/usr/bin/env python3
"""Upsert JSONL shard files (from embed_load.py --out) into Qdrant.

Single-process: avoids Qdrant local-mode file-lock contention.
The embedding fan-out happens in parallel; this merge runs once after.

Usage:
  python upsert_shards.py /tmp/shard_0.jsonl /tmp/shard_1.jsonl [...] [--qdrant-url URL]
"""
from __future__ import annotations

import argparse
import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--collection", default="restaurants")
    ap.add_argument("--local-path", default="./qdrant_data")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    client = (QdrantClient(url=args.qdrant_url)
              if args.qdrant_url else QdrantClient(path=args.local_path))
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
        print(f"dropped stale collection {args.collection}")
    client.create_collection(args.collection,
                             vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    print(f"created collection {args.collection} (384d, cosine)")

    total = 0
    buf: list[PointStruct] = []
    for shard in args.shards:
        with open(shard, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                buf.append(PointStruct(id=d["id"], vector=d["vector"], payload=d["payload"]))
                if len(buf) >= args.batch:
                    client.upsert(args.collection, buf)
                    total += len(buf)
                    buf = []
                    print(f"  upserted {total}", flush=True)
    if buf:
        client.upsert(args.collection, buf)
        total += len(buf)
    print(f"done: {total} points in {args.collection}")


if __name__ == "__main__":
    main()
