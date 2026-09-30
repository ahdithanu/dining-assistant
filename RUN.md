# Running the Dining Assistant API (Mac + Docker Desktop)

The service is a FastAPI app over the Yelp restaurant dataset (64,629 venues,
all 1,000 cities). The Docker image is self-contained: code + 61 MB dataset,
no volumes, no keys.

## Layout

```
dining-assistant/
  api.py              # FastAPI service: /search /ask /explain /invite /social /cities /health
  data.py             # deterministic query layer (load/search/explain/estimate_cost/social_signals)
  parse.py            # natural-language -> filters
  demo.py             # chat-style turn loop (also runnable standalone)
  invite.py           # invitation draft
  dataset/
    restaurants.csv   # baked into the image
  Dockerfile
  requirements.txt
```

## Build & run

```bash
cd dining-assistant
# one-time: fetch the 61 MB dataset (public Hugging Face mirror)
mkdir -p dataset
curl -sL "https://huggingface.co/datasets/jaimik69/Yelp-Restaurant-Dataset/resolve/main/restaurants.csv" -o dataset/restaurants.csv
docker build -t dining-assistant .
docker run -p 8000:8000 dining-assistant
```

First boot takes ~10s (dataset ingest), then:

- `GET  http://localhost:8000/health` → `{"ok": true, "venues": 64629}`
- `GET  http://localhost:8000/docs` → interactive Swagger UI for all endpoints

## Without Docker (quick check)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
DINING_DATA_PATH=dataset/restaurants.csv .venv/bin/uvicorn api:app --port 8000
```

## Frontend wiring

The frontend (built separately) calls these endpoints with the API base URL
as a config variable — `http://localhost:8000` for local Docker. CORS is
open (`*`) for the demo. See `claude-design-prompt.md` for the exact
request/response shapes the frontend was designed against.

## Demo cities

`/cities` returns all 1,000 cities ranked by open-venue count. Densest for
rehearsal: Philadelphia (4,372) → Tampa (2,503) → Indianapolis (2,371).

## Social signals ("Peep these too")

`dataset/social_signals.json` holds 10 Philadelphia spots with real Instagram
buzz (public posts gathered 2026-09-29 via social.search), matched to dataset
records. `GET /social?city=Philadelphia` serves them; `/search` and `/ask`
also bundle them so the frontend can render the "👀 Peep these too" section
under each result set. The Docker image bakes the file in.

## Typo-tolerant cuisines

`data.correct_cuisine()` fuzzy-matches cuisine queries against the real
category vocabulary ("vietnmse" → "vietnamese"). `/search` returns the
correction in `notes`; the chat parser (`parse.py`) applies the same fix to
typed phrases.

## Vector search (Qdrant + BGE embeddings)

`vector/` holds the semantic layer: every open venue embedded as
`"<name> -- <categories> -- <city>"` with `BAAI/bge-small-en-v1.5`
(384 dims, cosine) via the Hugging Face Inference API.

```bash
cd vector
docker compose up -d                       # Qdrant at :6333 (+ dashboard)
pip install qdrant-client
export HF_TOKEN=hf_...                     # free Hugging Face token
export DINING_DATA_PATH=../dataset/restaurants.csv
python embed_load.py --all --qdrant-url http://localhost:6333
python query.py "cozy ramen spot for a date night" --city Philadelphia \
  --qdrant-url http://localhost:6333
```

`POST /vibe` exposes it from the API: `{"q": "...", "city": "...",
"min_stars": 0, "limit": 5}` → ranked hits with cosine scores. Deterministic
filters narrow; the vector ranks. The API needs `QDRANT_URL` (or `QDRANT_PATH`
for local file mode) and `HF_TOKEN`; without them `/vibe` returns 503 with
a plain-English reason. Without `--qdrant-url`, the loader uses local file
mode (`./qdrant_data`, no Docker needed).

## Semantic chat (Phase 2: the big bet)

`POST /ask` now fuses both systems (`blend.py`): the parser extracts hard
constraints (city, cuisine, price, open late), `data.search` narrows to the
top 50 candidates, and the Qdrant vector reranks them by feel
(0.3 deterministic + 0.7 vibe). The response carries
`"vibe_applied": true` plus a "Ranked by vibe" note; send `"vibe": false`
to get pure deterministic order. When Qdrant or `HF_TOKEN` is unavailable,
`/ask` degrades gracefully to deterministic ranking (`vibe_applied: false`)
-- the chat never breaks.

Run the API with the vector stack:

```bash
export HF_TOKEN=<real token> QDRANT_URL=http://localhost:6333
DINING_DATA_PATH=dataset/restaurants.csv uvicorn api:app --port 8000
```
