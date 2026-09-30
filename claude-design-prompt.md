Design the frontend for a hackathon dining assistant. The backend is a Python data layer over the Yelp restaurant dataset (64k venues) exposed as a FastAPI service. The core layout is a **list + map side-by-side**: the assistant returns ranked restaurant picks, the list shows them with explanations, and the map shows them as pins.

## API contract (base URL must be a config variable)

- `POST /search` — body: `{ city, cuisines[], price_max, min_stars, min_reviews, open_late, day, flags[], limit }` → `{ picks: [{ business_id, name, address, city, state, stars, review_count, price_band (1-4 or null), categories[], lat, lng, score, matched_cuisines[] }] }`
- `POST /explain` — body: `{ business_id, matched_cuisines[], filters }` → `{ reasons[] }`
- `POST /invite` — body: `{ business_id, matched_cuisines[], party_size, day, time }` → `{ text, restaurant, address, time, party_size, cost_estimate, disclaimer }`
- `GET /cities` — → `{ cities: [{ name, venue_count }] }` for the city picker

## Map requirements

1. Numbered pins (1, 2, 3) matching the ranked list order; the top pick visually distinct.
2. Bidirectional selection: clicking a pin highlights its list card and vice versa.
3. Auto-fit map bounds to the current pins.
4. Use Leaflet + OpenStreetMap — no API keys, no paid services — with proper tile attribution.
5. Pin popup shows: name, stars × review count, price band ($–$$$$), matched cuisine.

## Edge cases you must handle

- **Missing/null lat/lng**: venue appears in the list only; the map never breaks.
- **Overlapping pins** (same or near-identical coordinates): cluster them or spider-offset so every pick is tappable.
- **Single pick**: center on it with a sensible default zoom, not max zoom.
- **Zero results**: friendly empty state on both map and list (not a blank map).
- **Picks update mid-session** (user adds a constraint): remove stale pins, animate new ones in; visually distinguish added vs. kept picks.
- **Selected pick gets filtered out**: selection clears gracefully, no dangling highlight.
- **Loading**: skeleton/spinner state on map and list while `/search` is in flight.
- **Backend unreachable**: friendly error state with a retry button, not a broken page.
- **Mobile**: stack map and list with a map/list toggle; pins need touch-friendly hit areas.
- **Invitation view**: emphasize the chosen restaurant's pin and include a directions link built from lat/lng (Google Maps URL).
- **Scale**: never attempt to render more than the returned picks — no plotting thousands of venues.
- **Accessibility**: the list must be keyboard navigable; pin selection must be reachable through the list.

## Deliverable

A working frontend (React or plain HTML/CSS/JS, your call) wired to the endpoints above. All copy should assume a 3-minute live demo: the UI must make the recommendations, the reasons, and the map instantly legible to someone seeing it for the first time.
