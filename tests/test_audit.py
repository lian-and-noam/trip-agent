"""Tests for the deterministic plan audit.

These cover the checks that used to be the critic's job. The critic is an LLM asked to
verify arithmetic, which it does inconsistently; everything here is computed, so it either
works every time or fails a test.
"""
from agent import audit


# --- Food costs ------------------------------------------------------------------------
def test_food_floors_raise_implausible_meal_costs():
    """Models lowball restaurant prices: they recall cheap averages and omit drinks and
    service. A mid-range sit-down dinner is not EUR12."""
    plan = {"days": [{"day": 1, "items": [
        {"name": "Lunch in Malá Strana", "cost_eur": 8, "duration_min": 60},
        {"name": "Dinner in Old Town", "cost_eur": 12, "duration_min": 90},
        {"name": "Prague Castle", "cost_eur": 15, "duration_min": 180}]}]}
    fixed, corrections = audit.apply_food_floors(plan, "mid-range")
    items = fixed["days"][0]["items"]
    assert items[0]["cost_eur"] == 18 and items[1]["cost_eur"] == 26
    assert items[2]["cost_eur"] == 15          # not a meal, untouched
    assert len(corrections) == 2
    assert fixed["total_cost_eur"] == 18 + 26 + 15


def test_food_floors_never_lower_a_price():
    """A planner quoting EUR40 for dinner is making a choice, not a mistake."""
    plan = {"days": [{"day": 1, "items": [{"name": "Dinner", "cost_eur": 40}]}]}
    fixed, corrections = audit.apply_food_floors(plan, "mid-range")
    assert fixed["days"][0]["items"][0]["cost_eur"] == 40 and corrections == []


def test_food_floors_scale_with_the_budget_tier():
    plan = {"days": [{"day": 1, "items": [{"name": "Dinner", "cost_eur": 1}]}]}
    assert audit.apply_food_floors(plan, "low")[0]["days"][0]["items"][0]["cost_eur"] == 14
    assert audit.apply_food_floors(plan, "luxury")[0]["days"][0]["items"][0]["cost_eur"] == 65


# --- Meal venues -----------------------------------------------------------------------
def test_business_names_are_told_apart_from_areas():
    assert audit.looks_like_a_business("Lokál Dlouhá Restaurant")
    assert audit.looks_like_a_business("U Fleku")
    assert audit.looks_like_a_business("Kafka's Bistro")
    assert not audit.looks_like_a_business("Malá Strana")
    assert not audit.looks_like_a_business("Old Town Square")
    assert not audit.looks_like_a_business("")


def test_meal_items_are_recognised():
    assert audit.is_meal("Lunch in Vinohrady") and audit.is_meal("Coffee break")
    assert not audit.is_meal("Prague Castle") and not audit.is_meal("Transfer to airport")


# --- Schedule arithmetic ----------------------------------------------------------------
def test_overlapping_items_are_caught():
    plan = {"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Castle", "duration_min": 180},
        {"time": "11:00", "name": "Lunch", "duration_min": 60}]}]}
    found = audit.check_schedule(plan)
    assert any("runs to 12:00" in f and "11:00" in f for f in found)


def test_days_running_past_midnight_are_caught():
    plan = {"days": [{"day": 2, "items": [
        {"time": "22:00", "name": "Dinner", "duration_min": 180}]}]}
    assert any("next morning" in f for f in audit.check_schedule(plan))


def test_unparseable_times_are_skipped_not_guessed():
    """A false "these overlap" sends the planner to rewrite a day that was fine."""
    plan = {"days": [{"day": 1, "items": [
        {"time": "morning", "name": "Castle", "duration_min": 180},
        {"time": "", "name": "Lunch", "duration_min": 60}]}]}
    assert audit.check_schedule(plan) == []


def test_trip_anchors_are_enforced():
    plan = {"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Castle", "duration_min": 120}]}]}
    found = audit.check_schedule(plan, {"start_time": "17/8 15:00"})
    assert any("does not begin until 15:00" in f for f in found)


def test_last_day_must_reach_the_departure_in_time():
    plan = {"days": [{"day": 1, "items": [
        {"time": "16:00", "name": "Museum", "duration_min": 180}]}]}
    found = audit.check_schedule(plan, {"end_time": "20/8 18:30", "end_point": "Airport"})
    assert any("past the 18:30 departure" in f for f in found)


def test_a_tight_but_legal_departure_is_flagged_not_failed():
    plan = {"days": [{"day": 1, "items": [
        {"time": "16:00", "name": "Museum", "duration_min": 60}]}]}
    found = audit.check_schedule(plan, {"end_time": "20/8 18:30", "end_point": "Airport"})
    assert any("too tight" in f for f in found)


# --- Opening hours ----------------------------------------------------------------------
def test_items_scheduled_outside_known_hours_are_caught():
    plan = {"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Castle", "venue": "Prague Castle", "duration_min": 180}]}]}
    found = audit.check_opening_hours(plan, {"Prague Castle": "Mo-Su 06:00-10:00"})
    assert len(found) == 1 and "opens" in found[0]


def test_hours_within_range_pass():
    plan = {"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Castle", "venue": "Prague Castle", "duration_min": 60}]}]}
    assert audit.check_opening_hours(plan, {"Prague Castle": "Mo-Su 06:00-22:00"}) == []


def test_unknown_hours_are_never_treated_as_closed():
    """OpenStreetMap does not tag everything. Absent data must not move a stop that is fine."""
    plan = {"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Cafe", "venue": "Somewhere", "duration_min": 60}]}]}
    assert audit.check_opening_hours(plan, {}) == []
    assert audit.check_opening_hours(plan, {"Somewhere": None}) == []
    # An opening_hours format we cannot parse is also unknown, not closed.
    assert audit.check_opening_hours(plan, {"Somewhere": "sunrise-sunset"}) == []


def test_always_open_venues_pass():
    plan = {"days": [{"day": 1, "items": [
        {"time": "23:00", "name": "Bridge", "venue": "Charles Bridge", "duration_min": 60}]}]}
    assert audit.check_opening_hours(plan, {"Charles Bridge": "24/7"}) == []


def test_open_ended_blocks_are_recognised():
    """The traveller picks where to go, so there is no single place to pin."""
    assert audit.is_flexible("Free time / shopping")
    assert audit.is_flexible("Free afternoon — Náplavka or shopping")
    assert audit.is_flexible("Optional Old Town Tower")
    assert not audit.is_flexible("Prague Castle")
    assert not audit.is_flexible("Walk to Charles Bridge")


def test_salvage_lays_out_the_places_the_run_actually_found():
    """When the planner cannot finish, a page of "Explore <city> (self-guided)" throws away
    every venue the run looked up. Those places are real and worth keeping."""
    plan = audit.salvage_plan({"destination": "Rome", "days": 3},
                              ["Colosseum", "Maximo", "Vatican Museums",
                               "Trevi Fountain", "Pantheon", "Roman Forum"])
    assert plan["salvaged"] is True
    names = [i["name"] for d in plan["days"] for i in d["items"]]
    assert "Colosseum" in names and "Vatican Museums" in names
    assert len(plan["days"]) == 3
    for day in plan["days"]:
        for item in day["items"]:
            assert item["time"] and item["venue"] == item["name"]


def test_salvage_gives_up_when_nothing_was_found():
    """With no real venues there is nothing to salvage, so the caller keeps its skeleton."""
    assert audit.salvage_plan({"destination": "Rome", "days": 2}, []) is None
    assert audit.salvage_plan({"destination": "Rome", "days": 2}, ["", "  "]) is None


def test_salvage_does_not_repeat_a_venue():
    plan = audit.salvage_plan({"destination": "Rome", "days": 2},
                              ["Colosseum", "Colosseum", "Pantheon"])
    names = [i["name"] for d in plan["days"] for i in d["items"]]
    assert len(names) == len(set(names))
