# Dining Assistant — 3-Minute Demo Script

**Total: ~3:00 · Four beats, ~45s each.** The assistant runs on the Yelp restaurant dataset (64,629 venues). Any of the 1,000 cities works — the data layer handles them all. Rehearse in one of these three, ranked by density (density = a live query never comes back empty):

1. **Philadelphia** — 4,372 open venues, densest in the dataset; saturates every demo query shape; strong bars/nightlife for "open late" queries.
2. **Tampa** — 2,503 open venues; also saturates the cheap-sushi-open-late query. The backup.
3. **Indianapolis** — 2,371 open venues; deep enough, but thinner on late-night coverage.

---

## Beat 1 — The dining request (0:00–0:45)

Say, verbatim:

> "Find me cheap sushi that's open late in Philadelphia."

The assistant parses this into filters (cuisine: sushi, price ≤ $$, open past 10pm, Philadelphia) and returns ranked picks from real data. Let the results land on screen for a beat — don't rush past them.

**Say:** "Every pick you see came from the dataset — the assistant can't invent a restaurant, because ranking happens in code, not in the model's head."

## Beat 2 — Explain the picks (0:45–1:30)

Point at the reasons shown under each pick. Say:

> "It doesn't just list names — it tells you *why*. This one matched sushi bars and Japanese, it's 5.0 stars across 155 reviews, price band $$, and it's open past 10pm."

**Say:** "The explanations are generated, not improvised — same input, same reasons, every run."

## Beat 3 — The new constraint (1:30–2:15)

Say, verbatim:

> "Actually — somewhere with outdoor seating."

The assistant re-queries with the added filter, drops picks that don't have it, and explains what changed. Point at the diff.

**Say:** "Watch what happened: it didn't start over, it *adapted* — same shortlist, new constraint, and it tells you which picks fell off and why."

## Beat 4 — The invitation (2:15–3:00)

Say, verbatim:

> "Draft an invitation for two, Friday at 7:30."

The assistant takes the top pick, estimates cost from the price band, and drafts the invite: restaurant, time, estimated total.

**Say:** "The cost is labeled an estimate from the $$ band — not the menu. We'd rather show our work than fake precision."

**Closer:** "Real data, explained picks, live adaptation, and a finished artifact — in three minutes."

---

## Fallbacks (only if you need them)

- **A query returns thin results:** "Good — watch this. The assistant tells you when the data is thin instead of making things up. Let me loosen one filter…" *(drop `min_reviews` or widen the price band)*
- **Someone shouts a sparse city:** "That city's thin in this dataset — the assistant says so rather than hallucinating. Let's run it in Philadelphia, where the data is dense."
- **The constraint kills every pick:** "It dropped all three because none has outdoor seating listed — that's the filter working, not failing. Let me try a different constraint."
- **Anything stalls:** "The data layer is a local module — no network, nothing to time out. Let's re-run the last step."
