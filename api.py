#!/usr/bin/env python3
"""FastAPI service for the dining-assistant hackathon entry.

Endpoints mirror the modules:
  POST /search  -> data.search      (ranked picks, with lat/lng for the map)
  POST /ask     -> demo.Session     (natural-language chat turn)
  POST /vibe    -> vibe.vibe_search (semantic search over Qdrant vectors)
  POST /explain -> data.explain     (structured reasons per pick)
  POST /invite  -> invite.draft_invitation (dinner invitation draft)
  GET  /cities  -> city picker data (all 1,000 dataset cities, by venue count)
  GET  /health  -> liveness check for the frontend's loading/error states

Run locally:
  DINING_DATA_PATH=../datasets/jaimik69-Yelp-Restaurant-Dataset/restaurants.csv \
      uvicorn api:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import data
from invite import draft_invitation
from demo import Session
import parse
import vibe
import blend


@asynccontextmanager
async def lifespan(app: FastAPI):
    records = data.load()  # cached in data module; ~9s on first boot
    app.state.by_id = {r["business_id"]: r for r in records}
    app.state.records = records
    yield


app = FastAPI(title="Dining Assistant API", lifespan=lifespan)

# Hackathon demo: the frontend (built separately) calls from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ models

class SearchRequest(BaseModel):
    city: str | None = None
    cuisines: list[str] = []
    price_max: int | None = None
    min_stars: float = 0.0
    min_reviews: int = 10
    open_late: bool = False
    day: str | None = None
    flags: list[str] = []
    limit: int = Field(default=5, le=20)


class PickOut(BaseModel):
    business_id: str
    name: str
    address: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    lat: float | None
    lng: float | None
    stars: float
    review_count: int
    price_band: int | None
    categories: list[str]
    score: float
    matched_cuisines: list[str]


class SocialSpotOut(BaseModel):
    business_id: str
    name: str
    address: str | None
    city: str | None
    state: str | None
    lat: float | None
    lng: float | None
    stars: float
    review_count: int
    price_band: int | None
    cuisine: str | None
    buzz: str


class SearchResponse(BaseModel):
    picks: list[PickOut]
    notes: list[str] = []
    social: list[SocialSpotOut] = []


class ExplainRequest(BaseModel):
    business_id: str
    matched_cuisines: list[str] = []
    filters: dict = {}


class ExplainResponse(BaseModel):
    reasons: list[str]


class InviteRequest(BaseModel):
    business_id: str
    matched_cuisines: list[str] = []
    party_size: int = Field(default=2, ge=1, le=50)
    day: str = "Friday"
    time: str = "7:30 PM"


class InviteResponse(BaseModel):
    text: str
    restaurant: str
    address: str
    time: str
    party_size: int
    cost_estimate: dict
    disclaimer: str


class CityOut(BaseModel):
    name: str
    venue_count: int


class CitiesResponse(BaseModel):
    cities: list[CityOut]


class AskRequest(BaseModel):
    text: str
    filters: dict | None = None  # previous turn's filters; omitted = fresh
    vibe: bool = True  # semantic rerank when the vector backend is available


class AskResponse(BaseModel):
    filters: dict
    picks: list[PickOut]
    summary: str
    notes: list[str]
    social: list[SocialSpotOut] = []
    vibe_applied: bool = False  # True when the vector reranked the picks


# ------------------------------------------------------------------ helpers

def _pick_out(pick: dict) -> dict:
    r = pick["record"]
    return {
        "business_id": r["business_id"],
        "name": r["name"],
        "address": r["address"],
        "city": r["city"],
        "state": r["state"],
        "postal_code": r["postal_code"],
        "lat": r["lat"],
        "lng": r["lng"],
        "stars": r["stars"],
        "review_count": r["review_count"],
        "price_band": r["price_band"],
        "categories": r["categories"],
        "score": pick["score"],
        "matched_cuisines": pick.get("matched_cuisines") or [],
    }


def _record_or_404(business_id: str) -> dict:
    rec = app.state.by_id.get(business_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown business_id")
    return rec


# ------------------------------------------------------------------ routes

@app.get("/health")
def health():
    return {"ok": True, "venues": len(app.state.records)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    notes = []
    fixed = []
    for c in req.cuisines:
        canon = data.correct_cuisine(c)
        if canon and canon != c.strip().lower():
            notes.append(f"Showing {canon} (corrected from \u2018{c}\u2019).")
        fixed.append(canon or c)
    picks = data.search(
        city=req.city,
        cuisines=fixed,
        price_max=req.price_max,
        min_stars=req.min_stars,
        min_reviews=req.min_reviews,
        open_late=req.open_late,
        day=req.day,
        flags=req.flags,
        limit=req.limit,
    )
    return {"picks": [_pick_out(p) for p in picks], "notes": notes,
            "social": [SocialSpotOut(**s) for s in data.social_signals(req.city)]}


class SocialResponse(BaseModel):
    city: str | None
    source: str
    spots: list[SocialSpotOut]


@app.get("/social", response_model=SocialResponse)
def social(city: str | None = None):
    """'Peep these too' — spots with real Instagram buzz for the city,
    matched to dataset records. Source: public Instagram posts gathered
    2026-09-29."""
    return {
        "city": city,
        "source": "Instagram public posts (Sep 2026)",
        "spots": [SocialSpotOut(**s) for s in data.social_signals(city)],
    }


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    rec = _record_or_404(req.business_id)
    pick = {"record": rec, "score": 0.0,
            "matched_cuisines": req.matched_cuisines}
    return {"reasons": data.explain(pick, req.filters)}


@app.post("/invite", response_model=InviteResponse)
def invite(req: InviteRequest):
    rec = _record_or_404(req.business_id)
    pick = {"record": rec, "score": 0.0,
            "matched_cuisines": req.matched_cuisines}
    return draft_invitation(pick, party_size=req.party_size,
                            day=req.day, time=req.time)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Natural-language turn: parse the phrase, merge with the previous
    turn's filters when given, and return structured picks.

    Phase 2: when the vector backend is available (and vibe=True), the
    deterministic picks are reranked by semantic similarity -- the parser
    narrows, the vector ranks by feel. Degrades to deterministic order
    when Qdrant or the HF token is unavailable.

    This is the primary product interaction -- the user types phrases,
    the deterministic parser + query layer does the rest.
    """
    s = Session()
    if req.filters:
        # sanitize/complete the client-supplied filters through the parser
        s.filters = parse.parse_request("", defaults=req.filters)
    msg = s.ask(req.text)
    picks = s.picks
    vibe_applied = False
    if req.vibe and picks:
        try:
            picks, vibe_applied = blend.rerank_by_vibe(
                req.text, s.filters, limit=3)
            s.picks = picks
        except Exception:
            picks, vibe_applied = s.picks, False  # never break the chat
    # summary = the head line of the assistant message (before the pick list),
    # or the whole message when nothing matched.
    summary = msg.split("\n\n")[0] if picks else msg
    notes = s.filters.get("notes") or []
    if vibe_applied:
        # The vector consumed the whole phrase, so bare "unrecognized word"
        # notes are noise now -- keep only the informative corrections.
        notes = [n for n in notes if "corrected from" in n]
        notes = notes + ["Ranked by vibe: semantic match on your phrase."]
    return {
        "filters": s.filters,
        "picks": [_pick_out(p) for p in picks],
        "summary": summary,
        "notes": notes,
        "social": [SocialSpotOut(**x)
                   for x in data.social_signals(s.filters.get("city"))],
        "vibe_applied": vibe_applied,
    }


@app.get("/cities", response_model=CitiesResponse)
def cities():
    counts: dict[str, int] = {}
    for r in app.state.records:
        if r["is_open"] and r["city"]:
            counts[r["city"]] = counts.get(r["city"], 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {"cities": [{"name": n, "venue_count": c} for n, c in ranked]}


class VibeRequest(BaseModel):
    q: str = Field(min_length=2, max_length=300)
    city: str | None = None
    min_stars: float = 0.0
    limit: int = Field(default=5, le=20)


class VibeHit(BaseModel):
    business_id: str | None
    name: str | None
    city: str | None
    stars: float | None
    review_count: int | None
    categories: str | None
    address: str | None
    lat: float | None
    lng: float | None
    score: float


class VibeResponse(BaseModel):
    query: str
    hits: list[VibeHit]


@app.post("/vibe", response_model=VibeResponse)
def vibe_search(req: VibeRequest):
    """Semantic search: embed the vibe phrase, rank Qdrant vectors.

    Deterministic filters (city, min_stars) narrow; the vector ranks.
    Needs QDRANT_URL (Docker) or QDRANT_PATH, plus HF_TOKEN for embeddings.
    """
    try:
        hits = vibe.vibe_search(req.q, city=req.city,
                                min_stars=req.min_stars, limit=req.limit)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"query": req.q, "hits": hits}
