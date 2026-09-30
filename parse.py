#!/usr/bin/env python3
"""
Deterministic, stdlib-only natural-language request parser for the
dining-assistant hackathon entry.

    parse_request(text, defaults=None) -> dict
    merge_filters(base, update)        -> dict

Turns "cheap sushi open late in Philadelphia" into the structured
filters that data.search() expects. Deliberately rule-based -- no ML,
no network -- so demo behavior is reproducible run to run.

Tokens the parser cannot map (but that look substantive) land in
`notes` instead of being silently dropped, so the agent can see --
and disclose -- what it failed to understand.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import _CUISINE_ALIASES, correct_cuisine, load  # noqa: E402

# ------------------------------------------------------------------ lookups

_cities_by_len = None


def _cities():
    """All distinct cities from the dataset, longest name first."""
    global _cities_by_len
    if _cities_by_len is None:
        seen = {}
        for r in load():
            c = (r.get("city") or "").strip()
            if c and c not in seen:
                seen[c] = c
        _cities_by_len = sorted(seen.values(), key=len, reverse=True)
    return _cities_by_len


_EXTRA_CUISINES = {
    "italian", "mexican", "japanese", "korean", "vietnamese", "french",
    "greek", "spanish", "mediterranean", "american", "steak", "cajun",
    "creole", "caribbean", "german", "turkish", "lebanese", "ethiopian",
    "filipino", "hawaiian", "cuban", "peruvian", "diner", "breakfast",
    "bakery", "cafe", "noodles", "sandwiches", "salad",
}
_KNOWN_CUISINES = set(_CUISINE_ALIASES) | _EXTRA_CUISINES

_DAYS = {
    "monday": "Monday", "mon": "Monday",
    "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
    "wednesday": "Wednesday", "wed": "Wednesday",
    "thursday": "Thursday", "thu": "Thursday",
    "thur": "Thursday", "thurs": "Thursday",
    "friday": "Friday", "fri": "Friday",
    "saturday": "Saturday", "sat": "Saturday",
    "sunday": "Sunday", "sun": "Sunday",
}

# Ordered: check "cheapest" before "cheap" so it wins.
_PRICE_PATTERNS = [
    (re.compile(r"\bcheapest\b"), 1),
    (re.compile(r"\bcheap\b|\bbudget\b|\baffordable\b|\binexpensive\b"), 2),
    (re.compile(r"\bupscale\b|\bfancy\b|\bfine dining\b|\bluxury\b"
                r"|\bexpensive\b|\bsplurge\b"), 4),
    (re.compile(r"\bmoderate\b|\bmid-?range\b|\bmid priced\b"), 3),
]

_RATING_RE = re.compile(
    r"\bhighly rated\b|\btop rated\b|\btop-rated\b|\bbest\b"
    r"|\bwell[ -]reviewed\b")

_LATE_RE = re.compile(r"\bopen late\b|\blate-?night\b")

_FLAG_PATTERNS = [
    (re.compile(r"\boutdoor seating\b|\bpatio\b"), "outdoor_seating"),
    (re.compile(r"\bkids?-friendly\b|\bfamily-friendly\b|\bgood for kids\b"),
     "good_for_kids"),
    (re.compile(r"\bwheelchair(?:\s+accessible|\s+access)?\b"),
     "wheelchair_accessible"),
    (re.compile(r"\b(?:takes?|accepts?)\s+reservations?\b|\breservations?\b"),
     "takes_reservations"),
    (re.compile(r"\bgood for groups\b"), "good_for_groups"),
    (re.compile(r"\bdeliver(?:y|s)?\b"), "delivery"),
    (re.compile(r"\btake-?out\b"), "takeout"),
]

_PARTY_RES = [
    re.compile(r"\bparty of (\d{1,2})\b"),
    re.compile(r"\btable for (\d{1,2})\b"),
    re.compile(r"\bfor (\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}) (?:people|persons|guests)\b"),
]

_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "with", "and", "or", "of",
    "to", "me", "my", "it", "its", "is", "are", "be", "i", "we", "us",
    "our", "you", "your", "find", "finds", "show", "shows", "get", "gets",
    "looking", "look", "want", "wants", "need", "needs", "some", "any",
    "please", "place", "places", "restaurant", "restaurants", "spot",
    "spots", "dinner", "lunch", "eat", "eating", "good", "nice", "great",
    "like", "that", "this", "there", "here", "can", "where", "what",
    "which", "s", "t", "re", "ll", "ve", "don", "not", "no",
    "food", "foods", "meal", "meals", "dish", "dishes",
    "cuisine", "cuisines", "craving", "cravings",
}

# In merge_filters, these default-equivalent values count as "no opinion"
# so a follow-up like "make it cheap" never clobbers an earlier "best".
_UNSET_NUMERICS = {"min_stars": 0.0, "min_reviews": 10}

_SEARCH_KEYS = ["city", "cuisines", "price_max", "min_stars", "min_reviews",
                "open_late", "day", "flags"]


def _blank():
    return {"city": None, "cuisines": [], "price_max": None,
            "min_stars": 0.0, "min_reviews": 10, "open_late": False,
            "day": None, "flags": [], "party_size": None, "notes": []}


def _words(s):
    return re.findall(r"[a-z]+", s.lower().replace("-", " "))


# ------------------------------------------------------------------- parsing

def _parse(text):
    """Internal parse. open_late is None (not False) when unmentioned,
    so merge_filters can tell 'not said' apart from an explicit False."""
    q = (text or "").lower()
    consumed = set()
    result = _blank()
    result["open_late"] = None

    # City: case-insensitive substring, longest match wins
    # (so "New Orleans" beats any shorter overlapping name).
    for city in _cities():
        if city.lower() in q:
            result["city"] = city
            consumed.update(city.lower().split())
            break

    # Party size: "party of 6", "table for 4", "for 4", "2 people".
    for rx in _PARTY_RES:
        m = rx.search(q)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 50:
                result["party_size"] = n
                consumed.update(
                    ["party", "of", "table", "for", "people", "persons",
                     "guests", m.group(1)])
            break

    # Price band.
    for rx, band in _PRICE_PATTERNS:
        m = rx.search(q)
        if m:
            result["price_max"] = band
            consumed.update(_words(m.group(0)))
            break

    # Rating boost.
    m = _RATING_RE.search(q)
    if m:
        result["min_stars"] = 4.0
        consumed.update(_words(m.group(0)))

    # Open late.
    m = _LATE_RE.search(q)
    if m:
        result["open_late"] = True
        consumed.update(_words(m.group(0)))

    # Weekday.
    for tok in re.findall(r"[a-z]+", q):
        if tok in _DAYS:
            result["day"] = _DAYS[tok]
            consumed.add(tok)
            break

    # Feature flags -> data.py's normalized names.
    for rx, flag in _FLAG_PATTERNS:
        m = rx.search(q)
        if m and flag not in result["flags"]:
            result["flags"].append(flag)
            consumed.update(_words(m.group(0)))

    # Cuisines: single-token match against known cuisine vocabulary,
    # with typo tolerance ("vietnmse" -> "vietnamese").
    notes_fixed = []
    for tok in re.findall(r"[a-z]+", q):
        if tok in _KNOWN_CUISINES and tok not in result["cuisines"]:
            result["cuisines"].append(tok)
            consumed.add(tok)
        elif len(tok) >= 5 and tok not in consumed:
            fixed = correct_cuisine(tok)
            if fixed and fixed not in result["cuisines"]:
                result["cuisines"].append(fixed)
                consumed.add(tok)
                notes_fixed.append((tok, fixed))

    # Notes: leftover substantive tokens, order-preserved, de-duplicated.
    notes = []
    for tok, fixed in notes_fixed:
        notes.append(f"showing {fixed} (corrected from \u2018{tok}\u2019)")
    for tok in re.findall(r"[a-z]+", q):
        if tok in consumed or tok in _STOPWORDS or len(tok) < 2:
            continue
        if tok not in notes:
            notes.append(tok)
    result["notes"] = notes

    return result


def parse_request(text, defaults=None):
    """Parse `text` into search filters.

    `defaults` (e.g. filters from an earlier turn) seed the result;
    anything actually parsed from `text` overrides them.
    """
    result = _blank()
    if defaults:
        result = merge_filters(result, defaults)
    merged = merge_filters(result, _parse(text))
    if merged["open_late"] is None:  # normalize the internal unset marker
        merged["open_late"] = False
    return merged


# ------------------------------------------------------------------- merging

def _is_meaningful(key, value):
    if value is None:
        return False
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    if isinstance(value, str) and value == "":
        return False
    if key in _UNSET_NUMERICS and value == _UNSET_NUMERICS[key]:
        return False
    return True


def merge_filters(base, update):
    """Return a new dict where `update`'s meaningful values override `base`.

    Meaningful = non-None, non-empty. Default-equivalent numerics
    (min_stars 0.0, min_reviews 10) count as "no opinion" so follow-up
    constraints never accidentally clear earlier ones. Neither input
    is mutated.
    """
    merged = dict(base)
    for key, value in update.items():
        if _is_meaningful(key, value):
            merged[key] = value
    return merged


# ----------------------------------------------------------------- self-test

if __name__ == "__main__":
    import data

    queries = [
        "cheap sushi open late in Philadelphia",
        "romantic italian place for 4 in Tampa",
        "bbq with outdoor seating in Indianapolis",
        "best brunch in New Orleans",
        "fancy french restaurant with wheelchair access for 2 in Philadelphia",
        "moderate thai place that takes reservations in Nashville",
        "good for groups pizza on Friday in Saint Louis",
        "cheapest ramen open late in Tucson",
        "vegan brunch in Philadelphia",
        "tacos for 6 with takeout in Tampa",
    ]

    for q in queries:
        parsed = parse_request(q)
        kwargs = {k: parsed[k] for k in _SEARCH_KEYS}
        picks = data.search(limit=3, **kwargs)
        top = (f"{picks[0]['record']['name']} "
               f"({picks[0]['record']['stars']}*, "
               f"{picks[0]['record']['review_count']} reviews)"
               if picks else "(no results)")
        print(f"QUERY: {q}")
        print(f"PARSED: {json.dumps(parsed)}")
        print(f"TOP: {top}  [n={len(picks)}]")
        print()

    # Follow-up constraint merge: "best sushi in Philadelphia" -> "make it cheap"
    base = parse_request("best sushi in Philadelphia")
    update = parse_request("make it cheap with outdoor seating")
    merged = merge_filters(base, update)
    print("MERGE TEST")
    print(f"  base:   {json.dumps(base)}")
    print(f"  update: {json.dumps(update)}")
    print(f"  merged: {json.dumps(merged)}")
    assert merged["city"] == "Philadelphia"
    assert merged["cuisines"] == ["sushi"]
    assert merged["min_stars"] == 4.0, "follow-up must not clear 'best'"
    assert merged["price_max"] == 2
    assert merged["flags"] == ["outdoor_seating"]

    # defaults seeding
    seeded = parse_request("cheap sushi", defaults={"city": "Tampa"})
    assert seeded["city"] == "Tampa" and seeded["price_max"] == 2
    print("\nDEFAULTS TEST OK: 'cheap sushi' + defaults city=Tampa ->",
          json.dumps(seeded))
    print("\nALL SELF-TESTS PASSED")
