"""Tool layer for the ReAct Planner, plus a deny-by-default dispatcher.

The LLM is never treated as a trust boundary. `run_tool` gates every call: unknown
tools are refused, side-effecting (Tier 2) tools are refused unless an explicit
approval is passed, tool inputs are filtered to a per-tool allowlist, and all outbound
HTTP is restricted to an egress host allowlist. Tier-2 safety therefore does not depend
on the tool bodies being stubs.

Tools:
- weather_tool: real, free, no API key (Open-Meteo); fails loud (typed errors) on 4xx/5xx.
- maps_tool / search_tool / reviews_tool: structured mocks (replace with a real API).
- flights_tool / booking_tool: fictive by design; never make a real reservation/purchase.
- calendar_tool: builds an .ics string locally.
- booking_confirm_tool / flight_book_tool: Tier 2, gated, never auto-fire.
"""
from urllib.parse import urlparse

import requests

from . import obs

# Egress allowlist: outbound HTTP is limited to these hosts.
ALLOWED_HOSTS = {"geocoding-api.open-meteo.com", "api.open-meteo.com"}
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


def _geocode(place):
    """Resolve a place name to coordinates. Returns a dict hit or None (never raises)."""
    try:
        data = _http_get("https://geocoding-api.open-meteo.com/v1/search",
                         {"name": place or "", "count": 1})
    except ToolError:
        return None
    hit = (data.get("results") or [None])[0]
    if not hit:
        return None
    return {"lat": hit["latitude"], "lon": hit["longitude"],
            "name": hit["name"], "country": hit.get("country", "")}


def geocode_place(place):
    """Public geocode helper used by the orchestrator to validate the destination."""
    return _geocode(place)


# ---- Real ----
def weather_tool(location=None, date=None, **_):
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
def maps_tool(query=None, near=None, **_):
    base = query or "point of interest"  # real API: Google Places / Overpass
    return {"ok": True, "source": "mock", "query": query, "near": near, "results": [
        {"name": f"{base} — central landmark", "type": "attraction",
         "open_hours": "09:00-17:00", "est_visit_min": 90, "walk_min_from_center": 8},
        {"name": f"{base} — popular spot", "type": "attraction",
         "open_hours": "24h", "est_visit_min": 60, "walk_min_from_center": 15},
        {"name": f"{base} — hidden gem", "type": "attraction",
         "open_hours": "10:00-18:00", "est_visit_min": 75, "walk_min_from_center": 22}]}


def search_tool(query=None, **_):  # real API: SerpAPI / Tavily / Bing
    return {"ok": True, "source": "mock", "query": query, "snippets": [
        f'Tip: book popular venues for "{query}" a few days ahead to avoid queues.',
        f'Local note: many sites near "{query}" close on Mondays.',
        f'Seasonal: evenings are pleasant for walking tours around "{query}".']}


def reviews_tool(place=None, **_):  # real API: Google Places Details / TripAdvisor
    return {"ok": True, "source": "mock", "place": place, "rating": 4.4,
            "reviews_count": 1280, "highlights": ["well rated", "good value", "can be busy midday"]}


# ---- Fictive (never real) ----
def flights_tool(from_=None, to=None, date=None, **kw):
    from_ = from_ or kw.get("from")  # 'from' is a Python keyword, so accept both keys
    return {"ok": True, "fictive": True, "from": from_, "to": to, "date": date, "options": [
        {"carrier": "Demo Air", "depart": "08:10", "arrive": "12:40", "stops": 0, "price_eur": 320},
        {"carrier": "Sample Wings", "depart": "14:25", "arrive": "20:05", "stops": 1, "price_eur": 248}],
        "note": "Fictive results — no booking is made."}


def booking_tool(kind=None, name=None, date=None, **_):
    return {"ok": True, "fictive": True, "kind": kind, "name": name, "date": date,
            "available": True, "price_eur": 140 if kind == "hotel" else 35,
            "note": "Fictive availability — no reservation is made."}


def calendar_tool(title=None, items=None, **_):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//TripAgent//EN"]
    for i, it in enumerate(items or []):
        it = it if isinstance(it, dict) else {}
        lines += ["BEGIN:VEVENT", f"SUMMARY:{it.get('summary', f'Item {i+1}')}",
                  f"DESCRIPTION:{(it.get('desc', '') or '').replace(chr(10), ' ')}", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return {"ok": True, "ics": "\n".join(lines), "title": title or "Trip"}


# Tier 2 (gated) — present for completeness, never auto-fire.
def booking_confirm_tool(**_):
    return {"ok": False, "gated": True, "note": "Requires explicit user approval (Tier 2). Not executed."}


def flight_book_tool(**_):
    return {"ok": False, "gated": True, "note": "Requires explicit user approval (Tier 2). Not executed."}


# Registry: function, description, tier (default 1), and the only input params accepted.
TOOLS = {
    "maps_tool":    {"fn": maps_tool,    "params": ["query", "near"],
                     "desc": "Find POIs, routes, distances near a place."},
    "search_tool":  {"fn": search_tool,  "params": ["query"],
                     "desc": "Web search for local tips & events."},
    "reviews_tool": {"fn": reviews_tool, "params": ["place"],
                     "desc": "Ratings/highlights for a venue."},
    "weather_tool": {"fn": weather_tool, "params": ["location", "date"],
                     "desc": "Real 7-day forecast for a location."},
    "flights_tool": {"fn": flights_tool, "params": ["from_", "from", "to", "date"],
                     "desc": "Fictive flight search/compare (no purchase)."},
    "booking_tool": {"fn": booking_tool, "params": ["kind", "name", "date"],
                     "desc": "Fictive hotel/restaurant availability (no reservation)."},
    "calendar_tool": {"fn": calendar_tool, "params": ["title", "items"],
                      "desc": "Build an .ics itinerary string."},
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
    safe_input = {k: v for k, v in raw.items() if k in allowed}
    try:
        return t["fn"](**safe_input)
    except Exception as e:  # last-resort guard; typed tools handle their own upstream errors
        obs.log("tool_error", tool=name, error=type(e).__name__)
        return {"ok": False, "note": f"Tool {name} failed"}


# Catalog string shown to the planner. Tier-2 tools are excluded so the loop can never call them.
TOOL_CATALOG = "\n".join(
    f'{name} — {t["desc"]}' for name, t in TOOLS.items() if t.get("tier") != 2
)
