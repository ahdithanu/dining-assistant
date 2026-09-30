# Vector search for the dining assistant (Qdrant + BGE embeddings)

Semantic search over restaurant names + categories. Complements the
deterministic filters in `data.py`: filters narrow, vectors rank by vibe.

## Quick start (Mac)

```bash
cd vector

# 1. start Qdrant (dashboard at http://localhost:6333/dashboard)
docker compose up -d

# 2. install the client
pip install qdrant-client

# 3. set your Hugging Face token (free, read-only is fine)
export HF_TOKEN=hf_...

# 4. embed + load Philadelphia (~4.4k venues, a few minutes)
export DINING_DATA_PATH=../dataset/restaurants.csv
python embed_load.py --city Philadelphia --qdrant-url http://localhost:6333

# 5. search
python query.py "cozy ramen spot for a date night" --qdrant-url http://localhost:6333
```

## What gets embedded

`"<name> -- <categories> -- <city>"`, embedded with
`BAAI/bge-small-en-v1.5` (384 dims, cosine) via the Hugging Face Inference API.
Each point stores `business_id`, `name`, `stars`, `review_count`,
`categories`, `address`, `latitude`, `longitude` as payload so you can
filter + join back to the CSV rows.

## Options

- `--city Tampa --limit 500` — load a different slice
- `--collection restaurants_v2` — version your collections
- Without `--qdrant-url`, uses local file mode (`./qdrant_data`, no Docker)

## Wiring it into the app

`POST /vibe` is live in `api.py` (see `vibe.py`):

```bash
curl -X POST http://localhost:8000/vibe \
  -H "Content-Type: application/json" \
  -d '{"q": "romantic french dinner with wine", "city": "Philadelphia"}'
```

Deterministic filters (`city`, `min_stars`) narrow; the vector ranks.
The server needs `QDRANT_URL` (Docker) or `QDRANT_PATH`, plus `HF_TOKEN`
for embedding the query. Missing Qdrant or token returns 503 with a
plain-English reason.
