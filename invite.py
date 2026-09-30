#!/usr/bin/env python3
"""
Invitation-draft extension for the dining-assistant hackathon entry.

    draft_invitation(pick, party_size=2, day="Friday", time="7:30 PM") -> dict

Takes a pick from data.search() and drafts a warm, text-message-style
dinner invitation. Cost is ALWAYS labeled an estimate derived from the
price band -- never a real price.
"""

import data


def _money(n):
    return f"${n:,}"


def draft_invitation(pick, party_size=2, day="Friday", time="7:30 PM"):
    """Draft a dinner invitation for a search pick.

    pick: {"record": clean_record, "score": float,
           "matched_cuisines": [str]} as returned by data.search().
    Returns dict with keys: text, restaurant, address, time, party_size,
    cost_estimate, disclaimer.
    """
    r = pick["record"]
    cost = data.estimate_cost(pick, party_size=party_size)

    addr_parts = [r.get("address"), r.get("city"), r.get("state")]
    address = ", ".join(p for p in addr_parts if p)
    if r.get("postal_code"):
        address += f" {r['postal_code']}"
    when = f"{day} at {time}"

    cuisines = pick.get("matched_cuisines") or []
    cuisine_bit = f"{cuisines[0]} " if cuisines else ""

    # --- cost sentence: estimate only, never a real price ---
    if cost["total"] is not None:
        lo, hi = cost["total"]
        cost_sentence = (
            f"Rough guess {_money(lo)}-{_money(hi)} total for all "
            f"{party_size} of us \u2014 estimated from the price band, not the menu."
        )
        disclaimer = (
            "Cost is an estimate derived from the venue's price band "
            f"({'$' * r['price_band']}), not actual menu prices."
        )
    else:
        cost_sentence = (
            "No price info listed for this one, so we'll figure out cost "
            "when we get there."
        )
        disclaimer = "No price data available for this venue; no estimate given."

    text = (
        f"Hey \u2014 dinner {day}? I'm thinking {r['name']} in {r['city']} "
        f"({address}) around {time}. "
        f"It's a {r['stars']}-star {cuisine_bit}spot with "
        f"{r['review_count']:,} reviews. "
        f"{cost_sentence} "
        f"You in?"
    )

    return {
        "text": text,
        "restaurant": r["name"],
        "address": address,
        "time": when,
        "party_size": party_size,
        "cost_estimate": cost,
        "disclaimer": disclaimer,
    }


# ------------------------------------------------------------------ self-test

if __name__ == "__main__":
    picks = data.search(city="Philadelphia", cuisines=["italian"],
                        price_max=2, limit=1)
    assert picks, "self-test query returned no picks"
    invite = draft_invitation(picks[0], party_size=4)
    print(invite["text"])
    print()
    print("restaurant :", invite["restaurant"])
    print("address    :", invite["address"])
    print("time       :", invite["time"])
    print("party_size :", invite["party_size"])
    print("cost       :", invite["cost_estimate"])
    print("disclaimer :", invite["disclaimer"])
