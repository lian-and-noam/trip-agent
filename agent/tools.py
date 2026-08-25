"""Tool layer for the ReAct Planner, plus a deny-by-default dispatcher.

The LLM is never treated as a trust boundary. `run_tool` gates every call: unknown
tools are refused, side-effecting (Tier 2) tools are refused unless an explicit
approval is passed, tool inputs are filtered to a per-tool allowlist, and all outbound
HTTP is restricted to an egress host allowlist. Tier-2 safety therefore does not depend
on the tool bodies being stubs.

Tools:
- weather_tool: real, free, no API key (Open-Meteo); fails loud (typed errors) on 4xx/5xx.
- maps_tool: real (Geoapify) — address, coordinates and OpenStreetMap opening_hours.
  Returns open_hours=None where OSM has none, rather than inventing plausible times.
- search_tool / reviews_tool: real (Tavily) — summarised web text with source links.
  Each of these degrades to sample data flagged `fictive: true` when its key is unset,
  so a bare checkout still runs without ever passing off invented data as real.
- flight_search_tool / booking_tool: fictive by design; never make a real reservation/purchase.
  flight_search_tool has no free real-data source available — see its docstring.
- booking_confirm_tool / flight_book_tool: Tier 2, gated, never auto-fire.
"""
import os
from urllib.parse import urlparse

import requests

from . import obs

# Egress allowlist: outbound HTTP is limited to these hosts.
ALLOWED_HOSTS = {"geocoding-api.open-meteo.com", "api.open-meteo.com",
                 "api.geoapify.com", "api.tavily.com"}
_HTTP_TIMEOUT = 6  # small, so several tool calls still fit the function time budget


class ToolError(Exception):
    """Typed tool failure, so callers can degrade on a reason rather than a stack trace."""
    def __init__(self, error_type, note):
        super().__init__(note)
        self.error_type = error_type
        self.note = note


def _http_get(url, params):
    """Single guarded GET: enforces the egress allowlist, a timeout, and status checks.
    Raises ToolError with a typed reason on any failure, never a silent empty success."""
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise ToolError("blocked_egress", f"Egress to {host} is not allowed")
    try:
        r = requests.get(url, params=params, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        raise ToolError("timeout", "Upstream request timed out")
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", 0)
        raise ToolError("rate_limited" if code == 429 else f"http_{code}", "Upstream returned an error")
    except requests.exceptions.RequestException:
        raise ToolError("network", "Upstream unreachable")
    except ValueError:
        raise ToolError("bad_response", "Upstream returned invalid JSON")


def _split_place(place):
    """Split "Prague, Czech Republic" into ("Prague", "czech republic").

    Open-Meteo's geocoder matches on the settlement name alone, so the qualifier must be
    kept out of the query. It is still useful for picking the right hit afterwards.
    """
    p = " ".join((place or "").split())
    if not p:
        return "", ""
    head, _, rest = p.partition(",")
    return head.strip(), rest.strip().lower()


def _pick_hit(results, qualifier):
    """Choose the geocoder hit that best matches the qualifier, else the first result.

    The API ranks by population, so the first hit is the sensible default. The qualifier
    only overrides that when it actually matches a country or admin region — which is what
    makes "Springfield, Illinois" resolve differently from plain "Springfield".
    """
    if not results:
        return None
    if qualifier:
        for r in results:
            fields = [str(r.get(k, "")).lower()
                      for k in ("country", "country_code", "admin1", "admin2")]
            if any(qualifier == f or qualifier in f or (f and f in qualifier) for f in fields):
                return r
    return results[0]


def _geocode(place):
    """Resolve a place name to coordinates. Returns a dict hit or None (never raises).

    One request: the bare settlement name (which is all the API understands), asking for
    several candidates so a qualifier like "Czech Republic" can disambiguate locally.
    """
    head, qualifier = _split_place(place)
    if not head:
        return None
    try:
        data = _http_get("https://geocoding-api.open-meteo.com/v1/search",
                         {"name": head, "count": 5})
    except ToolError:
        return None
    hit = _pick_hit(data.get("results") or [], qualifier)
    if not hit:
        return None
    return {"lat": hit["latitude"], "lon": hit["longitude"],
            "name": hit["name"], "country": hit.get("country", "")}


def geocode_place(place):
    """Public geocode helper used by the orchestrator to validate the destination."""
    return _geocode(place)


# ---- Real ----
def weather_tool(location=None, date=None, lat=None, lon=None, **_):
    # Coordinates win when the caller already has them: the orchestrator geocodes the
    # destination to validate it, and re-resolving the same place name here was a second
    # HTTP round trip on the critical path for an answer already in hand.
    if lat is not None and lon is not None:
        geo = {"lat": lat, "lon": lon, "name": location or "", "country": ""}
    else:
        geo = _geocode(location or "")
    if not geo:
        return {"ok": False, "error_type": "not_found", "note": f'Could not geocode "{location}"'}
    try:
        data = _http_get(
            "https://api.open-meteo.com/v1/forecast",
            {"latitude": geo["lat"], "longitude": geo["lon"],
             "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
             "forecast_days": 7, "timezone": "auto"})
    except ToolError as e:
        # Fail loud: the planner must know weather is unavailable, not assume success.
        return {"ok": False, "error_type": e.error_type, "note": f"weather unavailable ({e.error_type})"}
    d = data.get("daily", {}) or {}
    times = d.get("time", []) or []
    tmax = d.get("temperature_2m_max") or [None] * len(times)
    tmin = d.get("temperature_2m_min") or [None] * len(times)
    rain = d.get("precipitation_probability_max") or [None] * len(times)
    if not times:
        return {"ok": False, "error_type": "empty", "note": "Upstream returned no forecast data"}
    return {"ok": True, "location": f'{geo["name"]}, {geo["country"]}',
            "daily": [{"date": t, "max_c": tmax[i], "min_c": tmin[i], "rain_pct": rain[i]}
                      for i, t in enumerate(times)]}


# ---- Mock (structured, query-aware; replace the body with a real API) ----
GEOAPIFY_KEY = os.environ.get("GEOAPIFY_API_KEY", "").strip()
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "").strip()


def _geoapify_lookup(query, near=None):
    """Resolve a place through Geoapify and return whatever real detail it holds.

    Two calls: geocode/search to find the place, then place-details for the OpenStreetMap
    fields (opening_hours, website, phone). The second is best-effort — plenty of POIs are
    simply not tagged with hours, and "unknown" is an honest answer where the old mock
    invented "09:00-17:00" for everything.
    """
    text = f"{query} {near}".strip() if near else (query or "")
    data = _http_get("https://api.geoapify.com/v1/geocode/search",
                     {"text": text, "limit": 2, "format": "json", "apiKey": GEOAPIFY_KEY})
    hits = data.get("results") or []
    if not hits:
        return []

    out = []
    for i, hit in enumerate(hits[:2]):
        entry = {"name": hit.get("name") or hit.get("address_line1") or query,
                 "address": hit.get("formatted", ""),
                 "lat": hit.get("lat"), "lon": hit.get("lon"),
                 "category": hit.get("category", ""),
                 "open_hours": None, "website": None}
        # Details for the BEST match only. Fetching them for every hit made one maps_tool
        # call cost four HTTP round trips; at 6s each that is 24s, and a planner that checks
        # hours for five venues spent two minutes on HTTP alone and hit the run deadline
        # before it could write anything. The runner-up is context, not a place we schedule.
        if i == 0 and hit.get("place_id"):
            try:
                det = _http_get("https://api.geoapify.com/v2/place-details",
                                {"id": hit["place_id"], "features": "details",
                                 "apiKey": GEOAPIFY_KEY})
                for feat in (det.get("features") or []):
                    props = feat.get("properties") or {}
                    entry["open_hours"] = entry["open_hours"] or props.get("opening_hours")
                    entry["website"] = entry["website"] or props.get("website")
            except ToolError:
                pass        # details are a bonus; the located place is still useful
        out.append(entry)
    return out


def maps_tool(query=None, near=None, **_):
    """Look up a real place: address, coordinates, and opening hours when OSM has them.

    Falls back to clearly-labelled sample data when no GEOAPIFY_API_KEY is configured, so
    the agent still runs in a bare checkout — but never presents invented hours as real.
    """
    if not (query or "").strip():
        return {"ok": False, "error_type": "bad_input",
                "note": "maps_tool needs a place name in 'query', e.g. "
                        "{\"query\": \"Colosseum\", \"near\": \"Rome\"}."}
    if not GEOAPIFY_KEY:
        return {"ok": True, "source": "sample", "fictive": True, "query": query, "near": near,
                "note": "No GEOAPIFY_API_KEY set — sample data, hours are NOT real.",
                "results": [{"name": query or "point of interest", "open_hours": None,
                             "est_visit_min": 90}]}
    try:
        results = _geoapify_lookup(query, near)
    except ToolError as e:
        return {"ok": False, "error_type": e.error_type, "note": e.note}
    if not results:
        return {"ok": True, "source": "geoapify", "query": query, "results": [],
                "note": f'No place found for "{query}".'}
    return {"ok": True, "source": "geoapify", "query": query, "near": near,
            "results": results,
            "note": "open_hours is null where OpenStreetMap has no hours recorded — "
                    "treat that as unknown, not as closed."}


def search_tool(query=None, **_):
    """Real web search via Tavily, for facts no other tool covers."""
    if not TAVILY_KEY:
        return {"ok": True, "source": "sample", "fictive": True, "query": query,
                "note": "No TAVILY_API_KEY set — sample snippets, not real search results.",
                "snippets": [f'Book popular venues for "{query}" ahead to avoid queues.']}
    try:
        data = _tavily_search(query)
    except ToolError as e:
        return {"ok": False, "error_type": e.error_type, "note": e.note}
    return {"ok": True, "source": "tavily", "query": query,
            "answer": data.get("answer"),
            "snippets": [{"title": x.get("title"), "url": x.get("url"),
                          "content": (x.get("content") or "")[:400]}
                         for x in (data.get("results") or [])[:4]]}


def _tavily_search(query, max_results=4, depth="basic"):
    """POST to Tavily. _http_get is GET-only, so the allowlist, timeout and typed errors are
    applied here directly rather than skipped."""
    host = urlparse("https://api.tavily.com/search").hostname
    if host not in ALLOWED_HOSTS:
        raise ToolError("blocked_egress", f"Egress to {host} is not allowed")
    try:
        r = requests.post("https://api.tavily.com/search", timeout=_HTTP_TIMEOUT, json={
            "api_key": TAVILY_KEY, "query": query or "", "max_results": max_results,
            "search_depth": depth, "include_answer": True})
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        raise ToolError("timeout", "Search timed out")
    except requests.exceptions.RequestException:
        raise ToolError("network", "Search request failed")


def reviews_tool(place=None, **_):
    """What visitors say about a place, from a real web search.

    There is no free structured-reviews API — Google Places and TripAdvisor both gate theirs
    behind billing or partner approval. Tavily's summary plus cited snippets is the honest
    substitute. Deliberately returns NO numeric rating: the old mock answered "4.4 stars from
    1280 reviews" for every place on earth, and a fabricated number reads as authoritative in
    a way a quoted sentence does not.
    """
    if not place:
        return {"ok": False, "error_type": "bad_input", "note": "No place given"}
    if not TAVILY_KEY:
        return {"ok": True, "source": "sample", "fictive": True, "place": place,
                "note": "No TAVILY_API_KEY set — sample data, not real reviews.",
                "summary": None, "snippets": []}
    try:
        data = _tavily_search(f"{place} reviews what visitors say")
    except ToolError as e:
        return {"ok": False, "error_type": e.error_type, "note": e.note}
    return {"ok": True, "source": "tavily", "place": place,
            "summary": data.get("answer"),
            "snippets": [{"title": x.get("title"), "url": x.get("url"),
                          "content": (x.get("content") or "")[:300]}
                         for x in (data.get("results") or [])[:4]],
            "note": "Summarised from web sources; no verified star rating is available."}


# ---- Fictive (never real) ----
def flight_search_tool(from_=None, to=None, date=None, **kw):
    """Flight search. Returns fictive options — deliberately, and likely to stay that way.

    Amadeus Self-Service was the obvious free source for real fares; that portal was
    decommissioned in July 2026 and its API keys disabled. What remains (Duffel, Kiwi/Tequila,
    Skyscanner) is partner-approval gated, so there is no key a student project can obtain.

    The output is therefore synthetic and labelled `fictive: true` so neither the planner nor
    the traveller can mistake it for a real quote. Swapping in a provider later only means
    replacing the body: keep the {carrier, depart, arrive, stops, price_eur} shape and no
    prompt has to change.

    Booking stays separate and gated regardless of data source: see flight_book_tool (tier 2).
    """
    from_ = from_ or kw.get("from")  # 'from' is a Python keyword, so accept both keys
    return {"ok": True, "fictive": True, "from": from_, "to": to, "date": date, "options": [
        {"carrier": "Demo Air", "depart": "08:10", "arrive": "12:40", "stops": 0, "price_eur": 320},
        {"carrier": "Sample Wings", "depart": "14:25", "arrive": "20:05", "stops": 1, "price_eur": 248}],
        "note": "Fictive results — no booking is made."}


def booking_tool(kind=None, name=None, date=None, **_):
    return {"ok": True, "fictive": True, "kind": kind, "name": name, "date": date,
            "available": True, "price_eur": 140 if kind == "hotel" else 35,
            "note": "Fictive availability — no reservation is made."}


# Tier 2 (gated) — present for completeness, never auto-fire.
def booking_confirm_tool(**_):
    return {"ok": False, "gated": True, "note": "Requires explicit user approval (Tier 2). Not executed."}


def flight_book_tool(**_):
    return {"ok": False, "gated": True, "note": "Requires explicit user approval (Tier 2). Not executed."}


# Registry: function, description, tier (default 1), and the only input params accepted.
TOOLS = {
    "maps_tool":    {"fn": maps_tool,    "params": ["query", "near"],
                     "desc": "Look up a real place: address, coordinates and opening hours "
                             "(OpenStreetMap via Geoapify). Hours may be null = unknown."},
    "search_tool":  {"fn": search_tool,  "params": ["query"],
                     "desc": "Real web search (Tavily) for anything the other tools miss."},
    "reviews_tool": {"fn": reviews_tool, "params": ["place"],
                     "desc": "What visitors say about a place (web-sourced; no star rating)."},
    "weather_tool": {"fn": weather_tool, "params": ["location", "date"],
                     "desc": "Real 7-day forecast for a location."},
    "flight_search_tool": {"fn": flight_search_tool, "params": ["from_", "from", "to", "date"],
                           "desc": "Flight search/compare (fictive data, no purchase)."},
    "booking_tool": {"fn": booking_tool, "params": ["kind", "name", "date"],
                     "desc": "Fictive hotel/restaurant availability (no reservation)."},
    "booking_confirm_tool": {"fn": booking_confirm_tool, "params": [], "tier": 2,
                             "desc": "Gated: reserve hotel/restaurant."},
    "flight_book_tool":     {"fn": flight_book_tool, "params": [], "tier": 2,
                             "desc": "Gated: purchase flight."},
}


def run_tool(name, tool_input, approvals=None):
    """Deny-by-default dispatch; every decision is logged.

    - Unknown tool: refused.
    - Tier >= 2: refused unless `name` is in `approvals` (never passed from the loop).
    - tool_input: coerced to a dict and filtered to the tool's declared params.
    - A raising tool body: caught and returned as a typed error, so the pipeline never crashes.
    """
    approvals = approvals or set()
    t = TOOLS.get(name)
    if not t:
        obs.log("tool_denied", tool=name, reason="unknown")
        return {"ok": False, "note": f"Unknown tool: {name}"}
    if t.get("tier", 1) >= 2 and name not in approvals:
        obs.log("tool_denied", tool=name, reason="tier2_no_approval")
        return {"ok": False, "gated": True,
                "note": f"'{name}' requires explicit user approval (Tier 2) and was not executed."}

    raw = tool_input if isinstance(tool_input, dict) else {}
    allowed = t.get("params", [])
    # A key the tool already accepts is passed through untouched. Aliasing unconditionally
    # renames valid arguments — weather_tool really does take "location", and mapping it to
    # "query" dropped it and broke every forecast.
    safe_input = {}
    for key, value in raw.items():
        name_out = key if key in allowed else _ALIASES.get(key, key)
        if name_out in allowed:
            safe_input[name_out] = value

    # Dropping every argument silently is how a lookup becomes a call with no query: the
    # upstream rejects it, the planner sees an opaque failure, retries the same thing and
    # burns the run. Say what the tool accepts so the next attempt can be right.
    if raw and not safe_input:
        obs.log("tool_bad_params", tool=name, got=sorted(raw))
        return {"ok": False, "error_type": "bad_input",
                "note": f"{name} takes {allowed}; you sent {sorted(raw)}. "
                        f"Call it again using {allowed}."}
    try:
        return t["fn"](**safe_input)
    except Exception as e:  # last-resort guard; typed tools handle their own upstream errors
        obs.log("tool_error", tool=name, error=type(e).__name__)
        return {"ok": False, "note": f"Tool {name} failed"}


# Names a model reaches for that mean the same thing as a tool's real parameter. Accepting
# them costs nothing and saves a wasted round trip each time one is used.
_ALIASES = {
    "place": "query", "name": "query", "venue": "query", "location": "query",
    "search": "query", "q": "query", "text": "query",
    "city": "near", "country": "near", "area": "near", "region": "near", "in": "near",
}


# Catalog string shown to the planner. Tier-2 tools are excluded so the loop can never call them.
TOOL_CATALOG = "\n".join(
    f'{name} — {t["desc"]}' for name, t in TOOLS.items() if t.get("tier") != 2
)


def route_matrix(points, mode="walk"):
    """Travel times between consecutive points, in one Geoapify Route Matrix POST.

    `points` is a list of (lat, lon). Returns a list of minutes for each consecutive leg —
    one shorter than `points` — or None when unavailable.

    One request for a whole day rather than one per leg: a 6-stop day is a single call, not
    five. That matters because this runs inside the same time budget as the LLM calls, and
    an earlier version of the maps lookup nearly starved the planner by making four HTTP
    round trips where one would do.
    """
    if not GEOAPIFY_KEY or not points or len(points) < 2:
        return None
    host = urlparse("https://api.geoapify.com/v1/routematrix").hostname
    if host not in ALLOWED_HOSTS:
        raise ToolError("blocked_egress", f"Egress to {host} is not allowed")
    body = {"mode": mode,
            "sources": [{"location": [lon, lat]} for lat, lon in points],
            "targets": [{"location": [lon, lat]} for lat, lon in points]}
    try:
        r = requests.post(f"https://api.geoapify.com/v1/routematrix?apiKey={GEOAPIFY_KEY}",
                          timeout=_HTTP_TIMEOUT, json=body)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        raise ToolError("timeout", "Route matrix timed out")
    except requests.exceptions.RequestException:
        raise ToolError("network", "Route matrix request failed")

    # sources_to_targets[i][j] is the leg from point i to point j.
    grid = data.get("sources_to_targets") or []
    out = []
    for i in range(len(points) - 1):
        try:
            seconds = grid[i][i + 1].get("time")
        except (IndexError, KeyError, AttributeError, TypeError):
            return None
        if seconds is None:
            return None
        out.append(max(1, round(seconds / 60)))
    return out
