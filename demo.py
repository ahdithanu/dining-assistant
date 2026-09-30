#!/usr/bin/env python3
"""
Tier 1 core: the assistant turn loop for the dining-assistant hackathon entry.

    Session.ask(text)       -- fresh dining request -> top-3 picks + explanations
    Session.constrain(text) -- follow-up constraint -> re-search, diff, explain

Parsing delegates to parse.py (rule-based, deterministic). Retrieval and
ranking delegate to data.py (deterministic). This module is the glue:
session state, one-shot relaxation, kept/added/dropped diffing, and
chat-style rendering.

Key adaptation to parse.py's real interface:
  - parse_request(text, defaults=self.filters) both parses AND merges, and
    keeps "sticky" filters (e.g. open_late=True) across turns, because its
    internal _parse() marks unmentioned optionals as None.
  - merge never mutates; Session always reassigns self.filters.
"""

import re

import data
import parse

# parse._SEARCH_KEYS covers data.search()'s filter params except these two,
# which the Session owns (not the parser).
_EXTRA_SEARCH_KEYS = ["open_only", "limit"]


class Session:
    def __init__(self):
        # Blank parse seeds every key so self.filters is always complete.
        self.filters = parse.parse_request("")
        self.picks = []
        self.history = []

    # ------------------------------------------------------------ internals

    def _search_kwargs(self, filters=None):
        f = filters if filters is not None else self.filters
        kw = {k: f[k] for k in parse._SEARCH_KEYS}
        kw["open_only"] = True
        kw["limit"] = 3
        return kw

    def _describe(self, filters=None):
        f = filters if filters is not None else self.filters
        parts = []
        if f["cuisines"]:
            parts.append(" / ".join(f["cuisines"]))
        if f["price_max"]:
            parts.append(f"{'$' * f['price_max']} and under")
        if f["open_late"]:
            parts.append("open late" + (f" ({f['day']})" if f["day"] else ""))
        for fl in f["flags"]:
            parts.append(data._FLAG_LABELS.get(fl, fl))
        if f["city"]:
            parts.append(f"in {f['city']}")
        return ", ".join(parts) if parts else "restaurants"

    def _render_picks(self, picks, filters=None):
        f = filters if filters is not None else self.filters
        lines = []
        for i, p in enumerate(picks, 1):
            r = p["record"]
            price = ("$" * r["price_band"]) if r["price_band"] else "price n/a"
            lines.append(
                f"{i}. **{r['name']}** — {r['stars']} stars "
                f"({r['review_count']} reviews) · {price}")
            lines.append(f"   {r['address']}, {r['city']}")
            reasons = data.explain(p, f)
            lines.append("   Why: " + "; ".join(reasons) + ".")
        return "\n".join(lines)

    def _describe_change(self, new_filters):
        old, new = self.filters, new_filters
        bits = []
        if new.get("price_max") != old.get("price_max") and new.get("price_max"):
            bits.append(f"a {'$' * new['price_max']} cap")
        bits.extend(
            data._FLAG_LABELS.get(fl, fl) for fl in new.get("flags", [])
            if fl not in old.get("flags", []))
        if new.get("open_late") and not old.get("open_late"):
            bits.append("open late")
        bits.extend(c for c in new.get("cuisines", [])
                    if c not in old.get("cuisines", []))
        if new.get("city") != old.get("city") and new.get("city"):
            bits.append(f"in {new['city']}")
        return ", ".join(bits)

    def _drop_reason(self, record, new_filters):
        old, new = self.filters, new_filters
        for fl in new.get("flags", []):
            if fl not in old.get("flags", []) and not record["flags"].get(fl):
                return f"no {data._FLAG_LABELS.get(fl, fl)} listed"
        if (new.get("price_max") != old.get("price_max")
                and record["price_band"]
                and new.get("price_max") is not None
                and record["price_band"] > new["price_max"]):
            return f"over the {'$' * new['price_max']} cap"
        return "doesn't match the new filters"

    # ------------------------------------------------------------------ ask

    def ask(self, text):
        self.filters = parse.parse_request(text, defaults=self.filters)
        picks = data.search(**self._search_kwargs())

        relaxed = []
        if not picks and self.filters["min_reviews"] > 0:
            self.filters["min_reviews"] = 0
            relaxed.append("dropping the 10-review minimum")
            picks = data.search(**self._search_kwargs())
        if not picks and self.filters["price_max"] is not None:
            self.filters["price_max"] = None
            relaxed.append("dropping the price cap")
            picks = data.search(**self._search_kwargs())

        if not picks:
            msg = (f"I couldn't find anything for {self._describe()}, even "
                   f"after loosening the filters. Want to try a different "
                   f"cuisine or city?")
        else:
            if relaxed:
                head = ("Nothing matched at first, so I tried " +
                        " and ".join(relaxed) + " — here's what came up:")
            else:
                head = f"Here's what I found for {self._describe()}:"
            msg = head + "\n\n" + self._render_picks(picks)

        self.picks = picks
        self.history.append((text, msg))
        return msg

    # ------------------------------------------------------------ constrain

    def constrain(self, text):
        old_kw = self._search_kwargs()
        new_filters = parse.parse_request(text, defaults=self.filters)
        new_kw = self._search_kwargs(new_filters)

        if new_kw == old_kw:
            # The constraint adds nothing new: either already satisfied or
            # not something we can filter on. Say which.
            if (re.search(r"\$|under|budget|cheap|price|cost", text, re.I)
                    and self.filters.get("price_max")):
                msg = (f"You're already inside that budget — current picks "
                       f"are {'$' * self.filters['price_max']} and under, "
                       f"so the shortlist is unchanged:\n\n" +
                       self._render_picks(self.picks))
            else:
                alone = parse.parse_request(text)
                notes = alone.get("notes") or []
                extra = (f" I did note {', '.join(repr(n) for n in notes)} "
                         f"but can't filter on that yet." if notes else "")
                msg = (f"That fits your current filters "
                       f"({self._describe()}), so the shortlist is "
                       f"unchanged.{extra}\n\n" +
                       self._render_picks(self.picks))
            self.history.append((text, msg))
            return msg

        new_picks = data.search(**new_kw)
        if not new_picks:
            msg = (f"Adding \"{text.strip()}\" leaves zero matches, so I'm "
                   f"keeping your current shortlist for {self._describe()}. "
                   f"Want to loosen a different constraint instead?")
            self.history.append((text, msg))
            return msg

        old_ids = {p["record"]["business_id"] for p in self.picks}
        new_ids = {p["record"]["business_id"] for p in new_picks}
        kept = [p for p in new_picks
                if p["record"]["business_id"] in old_ids]
        added = [p for p in new_picks
                 if p["record"]["business_id"] not in old_ids]
        dropped = [p for p in self.picks
                   if p["record"]["business_id"] not in new_ids]

        change = self._describe_change(new_filters) or f'"{text.strip()}"'
        lines = [f"Adding {change} — here's what changed:"]
        if kept:
            lines.append("Kept: " +
                         ", ".join(p["record"]["name"] for p in kept) +
                         " — still match.")
        for p in dropped:
            lines.append(f"Dropped: {p['record']['name']} — "
                         f"{self._drop_reason(p['record'], new_filters)}.")
        for p in added:
            r = p["record"]
            lines.append(f"New: {r['name']} — {r['stars']} stars "
                         f"({r['review_count']} reviews).")
        lines.append("")
        lines.append("Updated shortlist:")
        lines.append(self._render_picks(new_picks, new_filters))

        msg = "\n".join(lines)
        self.filters = new_filters
        self.picks = new_picks
        self.history.append((text, msg))
        return msg


# --------------------------------------------------------------- demo arc

if __name__ == "__main__":
    s = Session()
    arc = [
        ("ask", "cheap sushi open late in Philadelphia"),
        ("constrain", "somewhere with outdoor seating"),
        ("constrain", "actually keep it under $30 a person"),
    ]
    for kind, text in arc:
        print(f"You: {text}\n")
        msg = s.ask(text) if kind == "ask" else s.constrain(text)
        print(f"Assistant:\n{msg}\n")
        print("-" * 64 + "\n")
    print(f"history entries: {len(s.history)}")
