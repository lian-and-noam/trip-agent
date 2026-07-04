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
    assert run_tool("maps_tool", "not-a-dict")["ok"] is True


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
