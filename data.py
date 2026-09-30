#!/usr/bin/env python3
"""
Data layer for the dining-assistant hackathon entry.

Loads the Yelp restaurants CSV into memory once and exposes a small,
deterministic query API:

    load()                          -> list of clean records
    search(...)                     -> ranked picks (dicts)
    explain(pick, filters)          -> list of human-readable reasons
    estimate_cost(pick, party_size) -> per-person / total cost ranges

Design: the LLM agent parses natural language into structured filters;
all retrieval and ranking happens here, so a demo can never hallucinate
a restaurant. The layer is stateless -- conversation state (shortlist,
applied filters) lives in the agent.
"""

import ast
import csv
import difflib
import json
import math
import os

DEFAULT_DATA_PATH = os.path.expanduser(
    "~/workspace/datasets/jaimik69-Yelp-Restaurant-Dataset/restaurants.csv"
)

PRICE_LABELS = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
BAND_MEANING = {1: "budget-friendly", 2: "casual", 3: "upscale", 4: "fine dining"}
# Per-person heuristic ranges by price band (labeled as estimates, not menu prices)
PER_PERSON = {1: (10, 20), 2: (21, 40), 3: (41, 70), 4: (71, 120)}

# Raw attribute keys -> normalized flag names used by search(flags=[...])
_FLAG_MAP = {
    "outdoor_seating": "OutdoorSeating",
    "delivery": "RestaurantsDelivery",
    "takeout": "RestaurantsTakeOut",
    "good_for_kids": "GoodForKids",
    "good_for_groups": "RestaurantsGoodForGroups",
    "wheelchair_accessible": "WheelchairAccessible",
    "takes_reservations": "RestaurantsReservations",
}
_FLAG_LABELS = {
    "outdoor_seating": "outdoor seating",
    "delivery": "delivery",
    "takeout": "takeout",
    "good_for_kids": "good for kids",
    "good_for_groups": "good for groups",
    "wheelchair_accessible": "wheelchair accessible",
    "takes_reservations": "takes reservations",
}

# Query term -> category substrings to match (extends plain substring matching)
_CUISINE_ALIASES = {
    "sushi": ["sushi"],
    "japanese": ["japanese", "sushi"],
    "tacos": ["tacos", "mexican"],
    "pizza": ["pizza", "italian"],
    "burgers": ["burgers", "american"],
    "ramen": ["ramen", "japanese"],
    "thai": ["thai"],
    "indian": ["indian"],
    "chinese": ["chinese"],
    "bbq": ["bbq", "barbeque"],
    "seafood": ["seafood"],
    "brunch": ["brunch", "breakfast"],
    "vegan": ["vegan", "vegetarian"],
    "vegetarian": ["vegetarian", "vegan"],
}

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- ingestion

def _parse_dict(raw):
    """Parse the stringified Python dicts in the attributes/hours columns."""
    if not raw or not raw.strip():
        return {}
    try:
        d = ast.literal_eval(raw)
        return d if isinstance(d, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _parse_hhmm(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _parse_hours(raw):
    """'{Day: (open_min, close_min) | None}' -- None means closed/unknown.

    '0:0-0:0' is treated as closed. Overnight ranges (close <= open)
    are kept as-is and handled at query time.
    """
    out = {}
    for day, span in _parse_dict(raw).items():
        if not span or not isinstance(span, str) or "-" not in span:
            out[day] = None
            continue
        try:
            o, c = (_parse_hhmm(x.strip()) for x in span.split("-", 1))
        except ValueError:
            out[day] = None
            continue
        out[day] = None if (o == 0 and c == 0) else (o, c)
    return out


def _is_true(v):
    return str(v).strip().lower() == "true"


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clean(row):
    attrs = _parse_dict(row.get("attributes"))
    try:
        price = int(attrs.get("RestaurantsPriceRange2"))
        price = price if 1 <= price <= 4 else None
    except (TypeError, ValueError):
        price = None
    cats = [c.strip().lower()
            for c in (row.get("categories") or "").split(",") if c.strip()]
    try:
        reviews = int(float(row.get("review_count") or 0))
    except (TypeError, ValueError):
        reviews = 0
    return {
        "business_id": row.get("business_id"),
        "name": row.get("name"),
        "address": row.get("address"),
        "city": row.get("city"),
        "state": row.get("state"),
        "postal_code": row.get("postal_code"),
        "lat": _to_float(row.get("latitude")),
        "lng": _to_float(row.get("longitude")),
        "stars": _to_float(row.get("stars")) or 0.0,
        "review_count": reviews,
        "is_open": (row.get("is_open") or "").strip() == "1",
        "price_band": price,
        "categories": cats,
        "hours": _parse_hours(row.get("hours")),
        "flags": {k: _is_true(attrs.get(v)) for k, v in _FLAG_MAP.items()},
    }


_records = None


def load(path=None):
    """Load and clean the CSV once; subsequent calls return the cache."""
    global _records
    if _records is None:
        p = path or os.environ.get("DINING_DATA_PATH", DEFAULT_DATA_PATH)
        with open(p, newline="", encoding="utf-8") as f:
            _records = [_clean(r) for r in csv.DictReader(f)]
    return _records


# ------------------------------------------------------------------- query

def _match_cuisine(rec_cats, query):
    q = query.strip().lower()
    terms = _CUISINE_ALIASES.get(q, [q])
    hits = [c for c in rec_cats if any(t in c or c in t for t in terms)]
    if hits:
        return hits
    # Typo tolerance: fuzzy-match against the real category vocabulary so
    # "vietnmse" still finds Vietnamese spots instead of matching nothing.
    corrected = correct_cuisine(q)
    if corrected:
        terms = _CUISINE_ALIASES.get(corrected, [corrected])
        return [c for c in rec_cats if any(t in c or c in t for t in terms)]
    return []


_categories_cache = None

def _all_categories():
    """Distinct category strings across the dataset (lowercase)."""
    global _categories_cache
    if _categories_cache is None:
        cats = set()
        for r in load():
            cats.update(r["categories"])
        _categories_cache = sorted(cats)
    return _categories_cache


def correct_cuisine(query):
    """Return the canonical category for a (possibly mistyped) cuisine query,
    or None when nothing close exists. Used to correct typos like
    'vietnmse' -> 'vietnamese' and to surface 'did you mean' hints."""
    q = query.strip().lower()
    if not q:
        return None
    if q in _CUISINE_ALIASES:
        return q
    if q in _all_categories():
        return q
    close = difflib.get_close_matches(q, _all_categories(), n=1, cutoff=0.8)
    if close:
        return close[0]
    # also try against the alias keys ("sushi", "bbq", ...)
    close = difflib.get_close_matches(q, list(_CUISINE_ALIASES), n=1, cutoff=0.8)
    return close[0] if close else None


def _open_past(rec, day, minutes):
    """True if the venue is open at/past `minutes` on `day` (handles overnight)."""
    span = rec["hours"].get(day)
    if not span:
        return False
    o, c = span
    if c <= o:  # overnight, e.g. 18:00 -> 02:00
        c += 24 * 60
    return c >= minutes


def _score(r):
    # Rating adjusted for review volume: 4.5*/800 reviews beats 5.0*/3 reviews.
    return r["stars"] + 0.3 * math.log10(r["review_count"] + 1)


def search(city=None, cuisines=None, price_max=None, min_stars=0.0,
           min_reviews=10, open_only=True, day=None, open_late=False,
           flags=None, limit=5):
    """Filter the dataset and return ranked picks.

    Each pick: {"record": clean_record, "score": float,
                "matched_cuisines": [category strings]}.

    Notes / trade-offs:
    - price_max excludes rows with unknown price band: every pick is
      *verified* within budget rather than possibly over.
    - min_reviews=10 default keeps single-review 5.0s from topping results.
    - open_late means open past 22:00 on `day` (default Friday).
    """
    recs = load()
    scored = []
    for r in recs:
        if open_only and not r["is_open"]:
            continue
        if city and (r["city"] or "").lower() != city.strip().lower():
            continue
        if price_max is not None:
            if r["price_band"] is None or r["price_band"] > price_max:
                continue
        if r["stars"] < min_stars or r["review_count"] < min_reviews:
            continue
        matched = []
        if cuisines:
            ok = True
            for q in cuisines:
                hits = _match_cuisine(r["categories"], q)
                if not hits:
                    ok = False
                    break
                matched.extend(hits)
            if not ok:
                continue
        if open_late and not _open_past(r, day or "Friday", 22 * 60):
            continue
        if flags and not all(r["flags"].get(f) for f in flags):
            continue
        scored.append({"record": r, "score": round(_score(r), 3),
                       "matched_cuisines": sorted(set(matched))})
    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored[:limit]


def explain(pick, filters=None):
    """Structured reasons for a pick -- the agent renders these as prose.

    Building explanation as a function (not improvised text) guarantees
    the demo requirement 'explain its recommendations' is always met.
    """
    r = pick["record"]
    f = filters or {}
    reasons = []
    mc = pick.get("matched_cuisines") or []
    if mc:
        reasons.append("Matches what you asked for: " + ", ".join(mc[:3]))
    reasons.append(f"Rated {r['stars']} stars across {r['review_count']} reviews")
    if r["price_band"]:
        reasons.append(
            f"Price band {'$' * r['price_band']} "
            f"({BAND_MEANING[r['price_band']]})")
    else:
        reasons.append("No price data listed")
    if f.get("open_late"):
        reasons.append("Open late (past 10pm)")
    for fl in f.get("flags") or []:
        reasons.append(f"{_FLAG_LABELS.get(fl, fl)} available")
    return reasons


def estimate_cost(pick, party_size=2):
    """Cost heuristic from the price band. Always labeled an estimate."""
    band = pick["record"]["price_band"]
    if band is None:
        return {"per_person": None, "total": None,
                "note": "No price data for this venue -- estimate unavailable."}
    lo, hi = PER_PERSON[band]
    return {"per_person": (lo, hi),
            "total": (lo * party_size, hi * party_size),
            "note": f"Estimate from the {'$' * band} price band, not the menu."}


# ------------------------------------------------------- social signals

_social_cache = None

def _social_path():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "dataset", "social_signals.json")
    if os.path.exists(cand):
        return cand
    # fall back next to wherever the CSV lives
    d = os.path.dirname(os.path.expanduser(
        os.environ.get("DINING_DATA_PATH", DEFAULT_DATA_PATH)))
    cand = os.path.join(d, "social_signals.json")
    return cand if os.path.exists(cand) else None


def social_signals(city=None, limit=6):
    """Spots with real social-media buzz (Instagram public posts, gathered
    2026-09-29), matched to dataset records. Returns dicts with business_id,
    name, stars, review_count, price_band, lat/lng, cuisine, and a `buzz`
    line describing the social proof. Empty list when no data for the city."""
    global _social_cache
    if _social_cache is None:
        p = _social_path()
        _social_cache = json.load(open(p, encoding="utf-8")) if p else {"spots": []}
    spots = _social_cache.get("spots", [])
    if city:
        spots = [s for s in spots
                 if (s.get("city") or "").lower() == city.strip().lower()]
    return spots[:limit]


def trending(city, limit=6, exclude_ids=()):
    """Region-aware 'trending' picks for cities without curated social data.

    Honest signal only: the most-loved open venues in the city, ranked by
    stars x review volume. Returns social-spot-shaped dicts (with a `buzz`
    line describing the data proof), so the frontend renders them exactly
    like the Instagram section. Empty list when the city has no coverage.
    """
    import math
    excluded = set(exclude_ids or ())
    recs = [r for r in load()
            if r.get("is_open") and (r.get("city") or "").lower() == city.strip().lower()
            and r["business_id"] not in excluded]
    recs.sort(key=lambda r: (r.get("stars") or 0) * math.log10((r.get("review_count") or 0) + 10),
              reverse=True)
    out = []
    for r in recs[:limit]:
        cats = r.get("categories") or []
        out.append({
            "business_id": r["business_id"],
            "name": r["name"],
            "address": r.get("address"),
            "city": r.get("city"),
            "state": r.get("state"),
            "lat": r.get("lat"),
            "lng": r.get("lng"),
            "stars": r.get("stars") or 0,
            "review_count": r.get("review_count") or 0,
            "price_band": r.get("price_band"),
            "cuisine": (cats[0] if cats else None),
            "buzz": (f"{r.get('stars')} stars across "
                     f"{(r.get('review_count') or 0):,} reviews -- "
                     f"one of {r.get('city')}'s most-loved spots"),
        })
    return out


def top_cities(n=5):
    """Most-covered cities by open-venue count, for coverage messaging."""
    counts = {}
    for r in load():
        if r.get("is_open") and r.get("city"):
            counts[r["city"]] = counts.get(r["city"], 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]]


# ------------------------------------------------------------------ self-test

if __name__ == "__main__":
    recs = load()
    print(f"loaded {len(recs)} records")
    picks = search(city="Philadelphia", cuisines=["sushi"], price_max=2,
                   open_late=True, day="Friday", min_reviews=20, limit=3)
    print(f"\nquery: cheap sushi, open late, Philadelphia -> {len(picks)} picks")
    for p in picks:
        r = p["record"]
        print(f"\n{r['name']} -- {r['stars']} stars "
              f"({r['review_count']} reviews, score={p['score']})")
        for line in explain(p, {"open_late": True}):
            print("  -", line)
        c = estimate_cost(p, party_size=2)
        print("  cost for 2:", c["total"], "|", c["note"])
