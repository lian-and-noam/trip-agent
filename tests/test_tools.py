"""Deny-by-default dispatcher gating and weather-tool error handling."""
from agent import tools
from agent.tools import run_tool, ToolError


def test_unknown_tool_refused():
    r = run_tool("definitely_not_a_tool", {})
    assert r["ok"] is False and "Unknown tool" in r["note"]


def test_tier2_denied_by_default():
    # Gating is enforced by the dispatcher, not by the tool body.
    r = run_tool("flight_book_tool", {})
    assert r["ok"] is False and r.get("gated") is True


def test_tier2_allowed_only_with_explicit_approval():
    r = run_tool("flight_book_tool", {}, approvals={"flight_book_tool"})
    assert r.get("gated") is True  # the stub itself is still inert, but it WAS reached deliberately


def test_input_filtered_to_declared_params(monkeypatch):
    captured = {}

    def fake_maps(query=None, near=None, **_):
        captured.update({"query": query, "near": near, "extra_seen": "evil" in _})
        return {"ok": True}

    monkeypatch.setitem(tools.TOOLS["maps_tool"], "fn", fake_maps)
    run_tool("maps_tool", {"query": "kyoto", "evil": "rm -rf", "near": "x"})
    assert captured["query"] == "kyoto" and captured["near"] == "x"
    assert captured["extra_seen"] is False  # unexpected key was stripped before the call


def test_non_dict_tool_input_is_safe():
    """It must not raise. It also must not pretend to succeed: a lookup with no query is a
    failure, and saying so is what lets the planner correct itself."""
    out = run_tool("maps_tool", "not-a-dict")
    assert out["ok"] is False and out["error_type"] == "bad_input"


def test_weather_reports_failure_loud(monkeypatch):
    monkeypatch.setattr(tools, "_geocode", lambda p: {"lat": 1, "lon": 2, "name": "X", "country": "Y"})

    def boom(url, params):
        raise ToolError("rate_limited", "429")

    monkeypatch.setattr(tools, "_http_get", boom)
    r = tools.weather_tool(location="Tokyo")
    assert r["ok"] is False and r["error_type"] == "rate_limited"  # NOT a silent ok:True


def test_weather_success(monkeypatch):
    monkeypatch.setattr(tools, "_geocode", lambda p: {"lat": 1, "lon": 2, "name": "Kyoto", "country": "JP"})
    monkeypatch.setattr(tools, "_http_get", lambda url, params: {"daily": {
        "time": ["2026-07-01", "2026-07-02"],
        "temperature_2m_max": [30, 31], "temperature_2m_min": [20, 21],
        "precipitation_probability_max": [10, 5]}})
    r = tools.weather_tool(location="Kyoto")
    assert r["ok"] is True and len(r["daily"]) == 2 and r["daily"][0]["max_c"] == 30


def test_egress_allowlist_blocks_unknown_host():
    try:
        tools._http_get("https://evil.example.com/x", {})
        assert False, "should have raised"
    except ToolError as e:
        assert e.error_type == "blocked_egress"


def test_tier2_excluded_from_catalog():
    assert "flight_book_tool" not in tools.TOOL_CATALOG
    assert "booking_confirm_tool" not in tools.TOOL_CATALOG
    assert "weather_tool" in tools.TOOL_CATALOG


# --- Geocoding: qualified place names -------------------------------------------------
# "Prague, Czech Republic" returns no results if sent whole, because the geocoder matches
# on the settlement name alone. The planner then burned a whole LLM turn retrying by hand.
def test_split_place_separates_the_qualifier():
    from agent.tools import _split_place
    assert _split_place("Prague, Czech Republic") == ("Prague", "czech republic")
    assert _split_place("  Prague  ") == ("Prague", "")
    assert _split_place("") == ("", "")


def test_pick_hit_uses_the_qualifier_to_disambiguate():
    from agent.tools import _pick_hit
    results = [
        {"name": "Springfield", "country": "United States", "admin1": "Missouri"},
        {"name": "Springfield", "country": "United States", "admin1": "Illinois"},
    ]
    assert _pick_hit(results, "illinois")["admin1"] == "Illinois"
    # No qualifier: keep the upstream ranking (by population) rather than guessing.
    assert _pick_hit(results, "")["admin1"] == "Missouri"
    assert _pick_hit([], "illinois") is None


def test_removed_and_renamed_tools_are_consistent():
    from agent.tools import TOOLS
    assert "flight_search_tool" in TOOLS
    assert "flights_tool" not in TOOLS      # renamed
    assert "calendar_tool" not in TOOLS     # removed: nothing ever called it


# --- Real place data: hours must never be invented --------------------------------------
def test_maps_tool_without_a_key_flags_its_data_as_not_real(monkeypatch):
    """The old mock returned "09:00-17:00" for every query, so the planner's opening-hours
    rule was being checked against fiction. Sample data must now announce itself."""
    from agent import tools
    monkeypatch.setattr(tools, "GEOAPIFY_KEY", "")
    out = tools.maps_tool(query="Prague Castle", near="Prague")
    assert out["fictive"] is True and out["source"] == "sample"
    assert out["results"][0]["open_hours"] is None      # unknown, not invented


def test_maps_tool_reports_unknown_hours_rather_than_guessing(monkeypatch):
    from agent import tools
    monkeypatch.setattr(tools, "GEOAPIFY_KEY", "test-key")
    monkeypatch.setattr(tools, "_geoapify_lookup",
                        lambda q, near=None: [{"name": "Lokál Dlouhá", "address": "Praha",
                                               "lat": 50.09, "lon": 14.42,
                                               "open_hours": None, "website": None}])
    out = tools.maps_tool(query="Lokál Dlouhá", near="Prague")
    assert out["source"] == "geoapify"
    assert out["results"][0]["open_hours"] is None
    assert "unknown" in out["note"].lower()             # the planner is told how to read it


def test_maps_tool_passes_through_real_hours(monkeypatch):
    from agent import tools
    monkeypatch.setattr(tools, "GEOAPIFY_KEY", "test-key")
    monkeypatch.setattr(tools, "_geoapify_lookup",
                        lambda q, near=None: [{"name": "Prague Castle", "address": "Praha",
                                               "lat": 50.09, "lon": 14.40,
                                               "open_hours": "Mo-Su 06:00-22:00",
                                               "website": "https://example.org"}])
    out = tools.maps_tool(query="Prague Castle", near="Prague")
    assert out["results"][0]["open_hours"] == "Mo-Su 06:00-22:00"


def test_geoapify_and_tavily_hosts_are_allowed():
    from agent.tools import ALLOWED_HOSTS
    assert "api.geoapify.com" in ALLOWED_HOSTS and "api.tavily.com" in ALLOWED_HOSTS


def test_search_tool_without_a_key_is_flagged(monkeypatch):
    from agent import tools
    monkeypatch.setattr(tools, "TAVILY_KEY", "")
    assert tools.search_tool(query="Prague tips")["fictive"] is True


def test_reviews_tool_returns_sourced_text_not_an_invented_rating(monkeypatch):
    """The mock returned a confident 4.4 stars from 1280 reviews for every place on earth.
    A fabricated number reads as authoritative in a way a quoted sentence does not."""
    from agent import tools
    monkeypatch.setattr(tools, "TAVILY_KEY", "test-key")
    monkeypatch.setattr(tools, "_tavily_search", lambda q, max_results=4, depth="basic": {
        "answer": "Visitors praise the views but mention long queues.",
        "results": [{"title": "Guide", "url": "https://example.org", "content": "Busy at noon."}]})
    out = tools.reviews_tool(place="Prague Castle")
    assert out["source"] == "tavily"
    assert "rating" not in out and "reviews_count" not in out
    assert out["snippets"][0]["url"] == "https://example.org"


def test_reviews_tool_without_a_key_is_flagged(monkeypatch):
    from agent import tools
    monkeypatch.setattr(tools, "TAVILY_KEY", "")
    out = tools.reviews_tool(place="Prague Castle")
    assert out["fictive"] is True and out["snippets"] == []


def test_reviews_tool_surfaces_upstream_failure(monkeypatch):
    from agent import tools
    monkeypatch.setattr(tools, "TAVILY_KEY", "test-key")

    def boom(*a, **k):
        raise tools.ToolError("timeout", "Search timed out")
    monkeypatch.setattr(tools, "_tavily_search", boom)
    assert tools.reviews_tool(place="x")["ok"] is False


def test_synonym_parameters_are_accepted():
    """A planner writing {"place": "Colosseum", "country": "Italy"} means the same thing as
    {"query": ..., "near": ...}. Silently dropping both produced a lookup with no query, an
    opaque upstream 400, and an identical retry that ended the run."""
    out = run_tool("maps_tool", {"place": "Colosseum", "country": "Italy"})
    assert out["ok"] is True
    assert out.get("query") == "Colosseum" and out.get("near") == "Italy"


def test_unusable_parameters_say_what_the_tool_wants():
    """An error the planner can act on beats one it can only repeat."""
    out = run_tool("maps_tool", {"landmark": "Colosseum"})
    assert out["ok"] is False and out["error_type"] == "bad_input"
    assert "query" in out["note"] and "landmark" in out["note"]


def test_a_valid_parameter_is_never_renamed_by_an_alias():
    """weather_tool really does take "location". Aliasing it to "query" unconditionally
    dropped it, and every forecast failed with a bad_input error naming the field it had
    just been given. An alias may only apply where the key is NOT already valid."""
    from agent.tools import TOOLS, _ALIASES
    for name, spec in TOOLS.items():
        for param in spec.get("params", []):
            resolved = param if param in spec["params"] else _ALIASES.get(param, param)
            assert resolved in spec["params"], f"{name}: '{param}' is aliased away"
    out = run_tool("weather_tool", {"location": "Rome, Italy"})
    assert out.get("error_type") != "bad_input"      # the argument reached the tool
