#!/usr/bin/env python3
"""Embed restaurants with HF Inference API and load them into Qdrant.

Full-scope loader: stable point IDs (hash of business_id), retries with
backoff, resumable via --offset.

Usage:
  python embed_load.py --all                                   # all 64k venues
  python embed_load.py --city Philadelphia                    # one city
  python embed_load.py --all --qdrant-url http://localhost:6333

Without --qdrant-url, uses local file mode at ./qdrant_data (no Docker needed).
With --qdrant-url, talks to a Qdrant container (see docker-compose.yml).

Embeddings: BAAI/bge-small-en-v1.5 (384 dims, cosine) via HF Inference API.
Auth: HF_TOKEN env var, else the stored custom.huggingface connector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    from dynamic_credentials import read_json_response
except ImportError:  # plain laptop: minimal fallback
    def read_json_response(resp):
        import json as _j
        return _j.loads(resp.read().decode())

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

HF_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, fast, feature-extraction on HF router
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
ALLOWED = ("huggingface.co", "api-inference.huggingface.co", "router.huggingface.co")
CSV_PATH = os.environ.get("DINING_DATA_PATH", "dataset/restaurants.csv")


def stable_id(business_id: str) -> int:
    """Deterministic 63-bit int from business_id -- safe to re-run / resume."""
    h = hashlib.sha256(business_id.encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


def embed_batch(texts: list[str], tries: int = 5) -> list[list[float]]:
    body = json.dumps({"inputs": texts}).encode()
    last_err = None
    for attempt in range(tries):
        req = urllib.request.Request(
            HF_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            req.add_header("Authorization", f"Bearer {hf_token}")
        else:
            sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
            from dynamic_credentials import add_surrogate_to_request
            add_surrogate_to_request(req, "custom.huggingface", allowed_hosts=ALLOWED)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = read_json_response(resp)
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"HF inference error: {data['error']}")
            return data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 502, 503) and attempt < tries - 1:
                wait = 2 ** attempt * 5
                print(f"    HTTP {e.code}, retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            if attempt < tries - 1:
                wait = 2 ** attempt * 5
                print(f"    {type(e).__name__}, retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"embed failed after {tries} tries: {last_err}")


def load_rows(city: str | None, limit: int | None, offset: int):
    rows = []
    skipped = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("is_open") not in ("1", "True", "true"):
                continue
            if city and r.get("city", "").lower() != city.lower():
                continue
            if skipped < offset:
                skipped += 1
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def doc_text(r: dict) -> str:
    cats = (r.get("categories") or "").replace(",", ", ")
    parts = [r.get("name") or "", cats, r.get("city") or ""]
    return " -- ".join(p for p in parts if p).strip()


def make_point(r: dict, vector: list[float]) -> PointStruct:
    return PointStruct(
        id=stable_id(r.get("business_id") or r.get("name", "")),
        vector=vector,
        payload={
            "business_id": r.get("business_id"),
            "name": r.get("name"),
            "city": r.get("city"),
            "stars": float(r.get("stars") or 0),
            "review_count": int(float(r.get("review_count") or 0)),
            "categories": r.get("categories"),
            "address": r.get("address"),
            "latitude": float(r.get("latitude") or 0),
            "longitude": float(r.get("longitude") or 0),
            "price": r.get("price"),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None, help="filter to one city (default: all)")
    ap.add_argument("--all", action="store_true", help="load every open venue in the CSV")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="resume from row N")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--collection", default="restaurants")
    ap.add_argument("--local-path", default="./qdrant_data")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index (0-based) for parallel embedding")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total shards; this worker handles rows[i %% N == shard]")
    ap.add_argument("--out", default=None,
                    help="write JSONL {id, vector, payload} instead of upserting "
                         "(for multi-agent fan-out; upsert separately)")
    args = ap.parse_args()

    if not args.city and not args.all and not args.limit:
        ap.error("pass --all, --city NAME, or --limit N")
    city = None if args.all else args.city

    rows = load_rows(city, args.limit, args.offset)
    scope = city or "ALL CITIES"
    if args.shard is not None:
        rows = [r for i, r in enumerate(rows)
                if i % args.num_shards == args.shard]
        scope += f" [shard {args.shard}/{args.num_shards}]"
    print(f"loaded {len(rows)} open venues ({scope}) from {CSV_PATH}")
    if not rows:
        return

    if args.out:
        # multi-agent mode: embed and dump JSONL, no Qdrant writes
        with open(args.out, "w", encoding="utf-8") as f:
            bs = args.batch_size
            t0 = time.time()
            for i in range(0, len(rows), bs):
                chunk = rows[i:i + bs]
                vecs = embed_batch([doc_text(r) for r in chunk])
                for r, v in zip(chunk, vecs):
                    p = make_point(r, v)
                    f.write(json.dumps({"id": p.id, "vector": p.vector,
                                        "payload": p.payload}) + "\n")
                done = i + len(chunk)
                print(f"  wrote {done}/{len(rows)} "
                      f"({done / max(time.time() - t0, 1):.1f}/s)", flush=True)
        print(f"done -> {args.out}")
        return

    import os
    client = (QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))
              if args.qdrant_url else QdrantClient(path=args.local_path))
    if not client.collection_exists(args.collection):
        client.create_collection(args.collection,
                                 vectors_config=VectorParams(size=384, distance=Distance.COSINE))
        print(f"created collection {args.collection} (384d, cosine)")
    else:
        print(f"collection {args.collection} exists -- upserting (stable IDs, safe to re-run)")

    bs = args.batch_size
    t0 = time.time()
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts = [doc_text(r) for r in chunk]
        vecs = embed_batch(texts)
        client.upsert(args.collection, [make_point(r, v) for r, v in zip(chunk, vecs)])
        done = i + len(chunk)
        rate = done / max(time.time() - t0, 1)
        print(f"  upserted {done}/{len(rows)} ({rate:.1f}/s)", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
