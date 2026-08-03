"""Tests for the deterministic plan audit.

These cover the checks that used to be the critic's job. The critic is an LLM asked to
verify arithmetic, which it does inconsistently; everything here is computed, so it either
works every time or fails a test.
"""
from agent import audit





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


# --- Cost plausibility --------------------------------------------------------------------
def _one_item(**item):
    return {"days": [{"day": 2, "items": [{"time": "19:30", "duration_min": 90, **item}]}]}


def test_a_silently_free_meal_is_flagged():
    """The reported bug: a dinner at EUR0 whose note never mentions money passed every
    check, because the existing one only fires when the note contradicts the number."""
    found = audit.check_costs(_one_item(name="Dinner in Trastevere", cost_eur=0,
                                        note="Trattorias line the side streets."))
    assert len(found) == 1
    assert "Dinner in Trastevere" in found[0] and "day 2" in found[0]


def test_a_priced_meal_is_left_alone():
    assert audit.check_costs(_one_item(name="Dinner in Trastevere", cost_eur=28,
                                       note="Trattorias line the side streets.")) == []


def test_a_zero_the_note_accounts_for_is_left_alone():
    """A meal the hotel includes looks identical from here, so the note decides."""
    for note in ("Breakfast is included with the room.",
                 "Complimentary buffet at the hotel.",
                 "Free tasting at the market stalls."):
        assert audit.check_costs(_one_item(name="Breakfast", cost_eur=0, note=note)) == []


def test_a_note_that_names_a_price_is_left_to_the_verified_check():
    """_check_free_but_priced already reports this one as a defect; reporting it twice
    would put the same item in front of the critic as both fact and question."""
    plan = _one_item(name="Lunch", cost_eur=0, note="About €12 for a set menu.")
    assert audit.check_costs(plan) == []
    assert any("Lunch" in f for f in audit.check_schedule(plan))


def test_non_meal_items_are_not_judged_on_cost():
    """Plenty of real stops are free, and guessing at those was what price floors did."""
    assert audit.check_costs(_one_item(name="Charles Bridge", cost_eur=0, note="")) == []
    assert audit.check_costs(_one_item(name="Free afternoon — coffee if you like",
                                       cost_eur=0, note="")) == []


def test_a_wholly_uncosted_itinerary_is_flagged():
    plan = {"days": [{"day": 1, "items": [
        {"name": "Castle", "cost_eur": 0}, {"name": "Bridge", "cost_eur": 0},
        {"name": "Museum", "cost_eur": 0}, {"name": "Park", "cost_eur": 0}]}]}
    assert any("Nothing in this itinerary is costed" in f for f in audit.check_costs(plan))


def test_one_paid_item_is_enough_to_show_prices_were_applied():
    plan = {"days": [{"day": 1, "items": [
        {"name": "Castle", "cost_eur": 15}, {"name": "Bridge", "cost_eur": 0},
        {"name": "Museum", "cost_eur": 0}, {"name": "Park", "cost_eur": 0}]}]}
    assert audit.check_costs(plan) == []


def test_cost_suspicions_are_not_shaped_like_computed_defects():
    """The orchestrator drops "Day N:" findings it cannot re-derive before delivery, and
    these cannot be re-derived by the schedule checks."""
    from agent.agent import _is_computed_defect
    found = audit.check_costs(_one_item(name="Dinner", cost_eur=0, note="Nice area."))
    assert found and not any(_is_computed_defect(f) for f in found)


def test_cost_suspicions_are_kept_out_of_the_verified_defect_list():
    """check_schedule output is re-asserted into must_fix verbatim; suspicions must not be."""
    plan = _one_item(name="Dinner", cost_eur=0, note="Nice area.")
    assert audit.check_schedule(plan) == []
    assert audit.check_costs(plan) != []


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


def test_a_free_item_whose_note_mentions_a_price_is_flagged():
    """The plan contradicting itself: an airport transfer at EUR0 whose note says "taxi or
    Leonardo Express", or a coffee described as EUR1.50 inside a free walking block. Flagged,
    not corrected — only the planner knows which option it meant."""
    plan = {"days": [{"day": 3, "items": [
        {"time": "19:45", "name": "Travel to airport", "duration_min": 45, "cost_eur": 0,
         "note": "taxi or Leonardo Express; final travel cost depends on chosen transfer"},
        {"time": "15:00", "name": "Walking loop", "duration_min": 150, "cost_eur": 0,
         "note": "Short coffee break included (coffee ~ €1.50)."},
        {"time": "12:30", "name": "St Peter's Basilica", "duration_min": 60, "cost_eur": 0,
         "note": "Free to enter; the security queue can be long."},
        {"time": "09:00", "name": "Colosseum", "duration_min": 90, "cost_eur": 22,
         "note": "Ticket is €22 — book ahead."}]}]}
    flagged = [f for f in audit.check_schedule(plan, {}) if "EUR0" in f]
    assert len(flagged) == 2
    assert any("Travel to airport" in f for f in flagged)
    assert any("Walking loop" in f for f in flagged)
    # A genuinely free item, and a priced one, are both left alone.
    assert not any("Basilica" in f or "Colosseum" in f for f in flagged)


def test_an_unpriced_item_is_flagged_whatever_it_is_called():
    """The recorded fact needs none of the meal-name or note heuristics below it."""
    plan = {"days": [{"day": 3, "items": [
        {"name": "Ferry to Capri", "cost_eur": 0, "cost_missing": True, "note": "lovely"}]}]}
    found = audit.check_costs(plan)
    assert len(found) == 1 and "never priced" in found[0] and "Ferry to Capri" in found[0]


def test_a_stated_zero_on_a_non_meal_is_still_left_alone():
    """Only the omission is a fact; a planner that wrote 0 for a free sight meant it."""
    plan = {"days": [{"day": 3, "items": [
        {"name": "Ferry to Capri", "cost_eur": 0, "note": "lovely"}]}]}
    assert audit.check_costs(plan) == []


def test_overnight_lodging_is_not_scheduled_time():
    """Regression from a real NYC run: the planner ended each day with "Lodging overnight,
    22:00, 480 min". The schedule arithmetic read that as an eight-hour engagement, so every
    night produced "Day N ends at 06:00 the next morning" plus "runs 22h end to end" — nine
    defects the planner was then sent to fix, none of them real."""
    plan = {"days": [{"day": 1, "title": "D1", "items": [
        {"time": "09:00", "name": "Central Park walk", "venue": "Central Park",
         "duration_min": 120, "cost_eur": 0},
        {"time": "19:00", "name": "Dinner", "venue": "Upper West Side",
         "duration_min": 60, "cost_eur": 14},
        {"time": "22:00", "name": "Lodging overnight", "venue": "HI NYC Hostel",
         "duration_min": 480, "cost_eur": 64},
    ]}]}
    problems = audit.check_schedule(plan)
    assert not [p for p in problems if "next morning" in p or "end to end" in p], problems


def test_an_unnamed_evening_block_of_hours_is_still_a_night():
    plan = {"days": [{"day": 2, "title": "D2", "items": [
        {"time": "09:00", "name": "Museum", "venue": "MoMA", "duration_min": 120},
        {"time": "21:00", "name": "Back at the hostel", "venue": "HI NYC Hostel",
         "duration_min": 540},
    ]}]}
    assert not [p for p in audit.check_schedule(plan) if "next morning" in p]


def test_a_genuine_late_night_is_still_reported():
    """The fix must not silence real defects: a long evening activity is not a night."""
    plan = {"days": [{"day": 1, "title": "D1", "items": [
        {"time": "09:00", "name": "Walk", "venue": "Park", "duration_min": 60},
        {"time": "21:00", "name": "Rooftop bar", "venue": "Trastevere", "duration_min": 300},
    ]}]}
    assert [p for p in audit.check_schedule(plan) if "next morning" in p or "very late" in p]
