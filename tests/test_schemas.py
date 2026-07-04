"""Deterministic guards + coercion (agent/schemas.py). These are the crash-proofing layer."""
from agent import schemas as s


def test_days_clamped_and_recorded():
    p = s.validate_profile({"days": 100, "destination": "Kyoto", "budget": "gold"})
    assert p["days"] == s.DAYS_MAX
    assert p["budget"] == "mid-range"
    assert any("clamped" in a for a in p["assumptions"])


def test_profile_total_on_junk_input():
    for junk in (None, [1, 2, 3], "nope", 42):
        p = s.validate_profile(junk)
        assert isinstance(p, dict)
        assert p["days"] >= s.DAYS_MIN and p["budget"] in s.BUDGET_LEVELS


def test_classify_turn():
    assert s.classify_turn([1, 2])[0] == "invalid"
    assert s.classify_turn("x")[0] == "invalid"
    assert s.classify_turn({"done": True, "draft_plan": {"days": []}})[0] == "done"
    kind, tool, ti = s.classify_turn({"tool": "maps_tool", "tool_input": {"query": "x"}})
    assert (kind, tool, ti) == ("tool", "maps_tool", {"query": "x"})
    # non-dict tool_input is coerced to {}
    assert s.classify_turn({"tool": "maps_tool", "tool_input": "oops"})[2] == {}


def test_draft_plan_recompute_and_none():
    dp = s.validate_draft_plan({"days": [{"day": 1, "items": [
        {"name": "A", "cost_eur": 10}, {"name": "B", "cost_eur": 5}]}], "total_cost_eur": 999})
    assert dp["total_cost_eur"] == 15                      # model total is not trusted
    assert s.validate_draft_plan({"days": [{"items": []}]}) is None
    assert s.validate_draft_plan("nope") is None


def test_minimal_plan_never_empty():
    mp = s.minimal_plan({"days": 3, "destination": "Rome"})
    assert len(mp["days"]) == 3 and mp["degraded"] is True
    assert all(day["items"] for day in mp["days"])


def test_verdict_fail_safe():
    assert s.validate_verdict("???")["verdict"] == "FAIL"      # unreadable critic must NOT pass
    assert s.validate_verdict({"verdict": "pass"})["verdict"] == "PASS"
    assert s.validate_verdict(None)["issues"] == []


def test_budget_ceiling_scales():
    assert s.budget_ceiling_eur({"budget": "mid-range", "days": 3}) == 780
    assert s.budget_ceiling_eur({"budget": "luxury", "days": 1}) == 650


def test_optional_fields_captured():
    p = s.validate_profile({"days": 3, "destination": "Rome", "budget": "mid-range",
                            "group": "couple", "style": "food", "when": "May",
                            "origin": "Tel Aviv", "dietary": ["vegetarian"]})
    assert p["when"] == "May" and p["origin"] == "Tel Aviv"
    assert p["dietary"] == ["vegetarian"]


def test_compact_profile_drops_empty_optionals():
    p = s.validate_profile({"days": 3, "destination": "Rome", "budget": "budget",
                            "group": "solo", "style": "art"})
    c = s.compact_profile(p)
    assert set(c) == {"days", "destination", "budget", "group", "style"}  # no empty extras
    c2 = s.compact_profile({**p, "origin": "Tel Aviv", "dietary": ["vegan"]})
    assert c2["origin"] == "Tel Aviv" and c2["dietary"] == ["vegan"]


def test_walking_coercion():
    assert s._as_walking("a lot") == "high"
    assert s._as_walking("unlimited") == "unlimited"
    assert s._as_walking("small") == "light"
    assert s._as_walking(20) == "high"
    assert s._as_walking(3) == "light"
    assert s._as_walking("nonsense") == "moderate"      # safe default
    p = s.validate_profile({"days": 2, "destination": "Kyoto", "budget": "budget",
                            "group": "solo", "style": "temples", "walking": "unlimited"})
    assert p["walking"] == "unlimited"
