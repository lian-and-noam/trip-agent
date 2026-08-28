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


def test_open_issues_survive_the_store_round_trip():
    """A plan reloaded from Supabase must still know what was left unfixed, or the next
    turn cannot act on the caveat the traveller was shown."""
    body = {"days": [{"day": 1, "items": [{"name": "A", "cost_eur": 10}]}]}
    assert "open_issues" not in s.validate_draft_plan(body)
    dp = s.validate_draft_plan(dict(body, open_issues=["Day 1 has no dinner.", "", None]))
    assert dp["open_issues"] == ["Day 1 has no dinner."]     # blanks dropped, not carried


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
                            "start_point": "Prague Bank Hotel", "dietary": ["vegetarian"]})
    assert p["when"] == "May" and p["start_point"] == "Prague Bank Hotel"
    assert p["dietary"] == ["vegetarian"]


def test_compact_profile_drops_empty_optionals():
    p = s.validate_profile({"days": 3, "destination": "Rome", "budget": "budget",
                            "group": "solo", "style": "art"})
    c = s.compact_profile(p)
    assert set(c) == {"days", "destination", "budget", "group", "style"}  # no empty extras
    c2 = s.compact_profile({**p, "end_point": "Václav Havel Airport", "dietary": ["vegan"]})
    assert c2["end_point"] == "Václav Havel Airport" and c2["dietary"] == ["vegan"]


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


# --- Trip anchors: where and when the trip physically starts and ends ------------------
def test_profile_keeps_start_and_end_anchors():
    """Regression: "from 17/8 15:00 in Prague Bank Hotel to a 20/8 18:30 flight" was reduced
    to destination=Prague, so the planner started day 1 at a made-up hour and never routed
    to the airport."""
    from agent.schemas import validate_profile, compact_profile
    p = validate_profile({
        "destination": "Prague", "days": 4, "budget": "mid-range", "group": "2 friends",
        "start_point": "Prague Bank Hotel", "start_time": "17/8 15:00",
        "end_point": "Václav Havel Airport Prague", "end_time": "20/8 18:30",
        "lodging": "Prague Bank Hotel",
    })
    for field in ("start_point", "start_time", "end_point", "end_time", "lodging"):
        assert p[field], field
    # And they must survive compaction, or the planner never sees them.
    c = compact_profile(p)
    assert c["start_point"] == "Prague Bank Hotel"
    assert c["end_point"] == "Václav Havel Airport Prague"
    assert c["end_time"] == "20/8 18:30"


def test_absent_anchors_are_dropped_from_the_compact_profile():
    """They are optional: an unset anchor must not add empty keys to every prompt."""
    from agent.schemas import validate_profile, compact_profile
    c = compact_profile(validate_profile(
        {"destination": "Rome", "days": 3, "budget": "low", "group": "solo"}))
    for field in ("start_point", "end_point", "start_time", "end_time", "lodging"):
        assert field not in c


def test_advisories_are_tidied_but_never_truncated():
    from agent.schemas import validate_verdict
    long_one = ("Note: " + "book the castle tour well in advance because summer slots "
                "sell out quickly and queues are long (advisory, cannot verify here)")
    v = validate_verdict({"verdict": "PASS", "be_aware": [long_one, "  ", "second tip"]})
    assert len(v["be_aware"]) == 2                      # blanks dropped, nothing capped
    assert v["be_aware"][0].startswith("Book the castle tour")   # label stripped
    assert "cannot verify" not in v["be_aware"][0]               # hedge stripped
    assert v["be_aware"][0].endswith("queues are long")          # but not truncated


# --- A stated amount is better information than a tier ---------------------------------
def test_group_size_read_from_free_text():
    from agent.schemas import group_size
    assert group_size({"group": "2 friends"}) == 2
    assert group_size({"group": "a couple"}) == 2
    assert group_size({"group": "solo traveller"}) == 1
    assert group_size({"group": "family"}) == 1        # unknown: never guess high


def test_stated_budget_overrides_the_tier_ceiling():
    """Regression: "2000 euro" was not recognised as a budget at all, and the guard measured
    against a generic mid-range allowance instead of the figure the traveller gave."""
    from agent.schemas import validate_profile, budget_ceiling_eur
    base = {"destination": "Prague", "days": 4, "group": "2 friends", "budget": "mid-range"}
    # A party total is divided: every cost in the plan is quoted per person.
    total = validate_profile({**base, "budget_amount_eur": 2000, "budget_basis": "total"})
    assert budget_ceiling_eur(total) == 1000
    per_person = validate_profile({**base, "budget_amount_eur": 2000,
                                   "budget_basis": "per person"})
    assert budget_ceiling_eur(per_person) == 2000
    # With no amount it still falls back to the tier ceiling.
    assert budget_ceiling_eur(validate_profile(base)) == 1040


def test_budget_amount_survives_compaction():
    from agent.schemas import validate_profile, compact_profile
    c = compact_profile(validate_profile(
        {"destination": "Rome", "days": 3, "budget": "low", "group": "solo",
         "budget_amount_eur": 800, "budget_basis": "per person"}))
    assert c["budget_amount_eur"] == 800 and c["budget_basis"] == "per person"


# --- Broken values, caught deterministically rather than left to the critic ---------------
def test_one_minute_visits_are_raised_to_something_real():
    """'Explore Prague Castle' at duration_min 1 is a broken number, not a judgement call —
    the model drops a digit or emits hours. Asking the critic to spot it worked sometimes."""
    from agent.schemas import validate_draft_plan
    plan = validate_draft_plan({"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Explore Prague Castle", "duration_min": 1},
        {"time": "11:00", "name": "Jewish Museum visit", "duration_min": 0},
        {"time": "12:00", "name": "Walk to Old Town", "duration_min": 1},
        {"time": "13:00", "name": "Lunch", "duration_min": 60}]}]})
    d = [i["duration_min"] for i in plan["days"][0]["items"]]
    assert d[0] == 15 and d[1] == 15        # visits get a real floor
    assert d[2] == 5                        # a leg may genuinely be short
    assert d[3] == 60                       # a sane value is untouched


def test_broken_unicode_is_stripped_from_displayed_text():
    """A model emitting mangled \\u escapes produced 'Na P\\x01Y\\x00edkop' — boxes on screen,
    and rubbish inside the generated map URL."""
    from agent.schemas import validate_draft_plan, clean_text
    assert clean_text("Na P\x01\x59\x00edkop\x01\x1b") == "Na PYedkop"
    assert clean_text("Na Příkopě") == "Na Příkopě"      # real accents survive
    plan = validate_draft_plan({"days": [{"day": 1, "title": "Day\x001", "items": [
        {"time": "09:00", "name": "Stroll\x07", "venue": "Na P\x01edkop", "duration_min": 30}]}]})
    item = plan["days"][0]["items"][0]
    assert "\x07" not in item["name"] and "\x01" not in item["venue"]
    assert "\x00" not in plan["days"][0]["title"]


def test_venues_are_cleaned_into_searchable_place_names():
    """A venue is fed to a map search. "Prague (flight)" describes the activity, not an
    address, and searching for it lands nowhere useful."""
    from agent.schemas import clean_venue, validate_draft_plan
    assert clean_venue("Prague (flight)") == "Prague"
    assert clean_venue("Hotel Ibis [check-in]") == "Hotel Ibis"
    assert clean_venue("the Old Town Square") == "Old Town Square"
    assert clean_venue("Václav Havel Airport Prague") == "Václav Havel Airport Prague"
    item = validate_draft_plan({"days": [{"day": 1, "items": [
        {"time": "14:15", "name": "Travel to airport", "venue": "Prague (flight)",
         "venue_from": "the Old Town Square", "duration_min": 75}]}]})["days"][0]["items"][0]
    assert item["venue"] == "Prague" and item["venue_from"] == "Old Town Square"


def test_literal_escape_sequences_are_decoded_in_displayed_text():
    """A model sometimes writes the text of an escape instead of the character, so an en dash
    arrives as "u2013" and an accent as "u00e1"."""
    from agent.schemas import clean_text, validate_verdict
    assert clean_text("14:00u201315:00") == "14:00–15:00"
    assert clean_text("Vu00e1clav Havel Airport") == "Václav Havel Airport"
    # Ordinary words that merely contain "u" plus digits are left alone.
    assert clean_text("you2019re here") == "you2019re here"
    assert clean_text("Route u66 bus") == "Route u66 bus"
    v = validate_verdict({"verdict": "FAIL",
                          "must_fix": ["Day 4: To Vu00e1clav Havel u2014 inconsistent"]})
    assert "Václav" in v["must_fix"][0] and "—" in v["must_fix"][0]


def test_a_missing_cost_is_recorded_as_missing_not_as_free():
    """A cost the planner never gave used to become 0, which is indistinguishable from a
    genuinely free item — so nothing downstream could tell a skipped price from a real one."""
    day = lambda item: {"days": [{"day": 1, "items": [dict({"name": "Dinner"}, **item)]}]}
    for absent in ({}, {"cost_eur": None}, {"cost_eur": "varies"}, {"cost_eur": "depends"}):
        it = s.validate_draft_plan(day(absent))["days"][0]["items"][0]
        assert it["cost_eur"] == 0 and it["cost_missing"] is True

    for given in ({"cost_eur": 0}, {"cost_eur": "0"}, {"cost_eur": 25}, {"cost_eur": "€25"}):
        it = s.validate_draft_plan(day(given))["days"][0]["items"][0]
        assert "cost_missing" not in it          # a stated 0 is a price, not an omission
    # The total still sums cleanly with an unpriced item present.
    assert s.validate_draft_plan(day({}))["total_cost_eur"] == 0


def test_internal_names_never_reach_the_traveller():
    """From a real plan: notes cited our own field names and tool names. The planner prompt
    forbids both and the model did it anyway, so it is enforced here instead."""
    from agent import schemas
    out = schemas.clean_text("Colosseum standard ticket ~EUR18 (source summary). Timed entry: "
                             "book in advance (maps_tool hours: Mar 30-Sep 30 08:30-19:15).")
    assert "source summary" not in out
    assert "maps_tool" not in out
    assert "hours: Mar 30-Sep 30 08:30-19:15" in out, "the hours themselves are worth keeping"
    assert out.startswith("Colosseum standard ticket ~EUR18. Timed entry")
