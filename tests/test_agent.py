"""Orchestrator behavior: intake branching, contract shape, crash-proofing, no tier-2 dispatch."""
import json


# ---- Conversational Intake branching (A / B / C) -----------------------------------
def test_branch_A_missing_info_asks_and_skips_planner(patched_agent, scripted_chat):
    intake = {"profile": {"destination": "Kyoto"}, "confirmed": False,
              "question": "How many days are you planning to travel?"}
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent("User: I want to visit Kyoto")
    assert "how many days" in out["response"].lower()
    mods = [s["module"] for s in out["steps"]]
    assert mods == ["Conversational Intake"]           # planner/reflection NEVER ran


def test_branch_B_complete_but_unconfirmed_asks_confirmation(patched_agent, scripted_chat):
    intake = {"profile": {"destination": "Kyoto", "days": 3, "budget": "mid-range",
                          "group": "couple", "style": "temples"}, "confirmed": False, "question": ""}
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent("User: 3 days in Kyoto, couple, mid-range, temples")
    # Asserts on what the card must convey, not its wording: the collected values are shown
    # back and the user is told how to proceed.
    body = out["response"].lower()
    assert "yes" in body
    assert all(v in body for v in ("kyoto", "3", "couple", "mid-range", "temples"))
    assert "```" not in out["response"]          # humanised card, not a raw JSON dump
    assert [s["module"] for s in out["steps"]] == ["Conversational Intake"]   # still no planning


def test_branch_C_confirmed_runs_planner(patched_agent, scripted_chat):
    patched_agent.install(scripted_chat())   # default intake is complete + confirmed
    out = patched_agent.module.run_agent(
        "User: 2 days in Kyoto, couple, mid-range, temples\nAgent: ...confirm?\nUser: yes")
    mods = [s["module"] for s in out["steps"]]
    assert "ReAct Planner" in mods and "Output Formatter" in mods
    # Steps describe LLM calls only. The intake call is traced under its own name, and the
    # deterministic profiler — which makes no model call — is not a step at all.
    assert mods[0] == "Conversational Intake"
    assert "Preference Profiler" not in mods


def test_intake_only_costs_one_llm_call(patched_agent):
    """Budget guard: Branch A/B must make EXACTLY one LLM call (no plan/reflect/format)."""
    calls = {"n": 0}

    def counting_chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        calls["n"] += 1
        return json.dumps({"profile": {"destination": "Kyoto"}, "confirmed": False,
                           "question": "How many days?"})

    patched_agent.install(counting_chat)
    patched_agent.module.run_agent("User: visit Kyoto")
    assert calls["n"] == 1     # the expensive loops were skipped entirely


# ---- Branch C internals (contract, crash-proofing, tiers) --------------------------
def test_happy_path_shape(patched_agent, scripted_chat):
    patched_agent.install(scripted_chat())
    out = patched_agent.module.run_agent("2 days in Kyoto, mid-range, love temples")
    assert set(out.keys()) == {"response", "steps", "state", "branch", "usage", "ms"}
    assert isinstance(out["response"], str) and out["response"]
    assert isinstance(out["steps"], list) and out["steps"]
    assert out["branch"] == "plan"
    assert out["state"]["plan"]["days"]          # the plan is handed back for revision


def test_step_schema_and_module_names(patched_agent, scripted_chat, diagram_modules):
    patched_agent.install(scripted_chat())
    out = patched_agent.module.run_agent("2 days in Kyoto")
    for step in out["steps"]:
        assert set(step.keys()) == {"module", "prompt", "response"}   # EXACT contract
        assert isinstance(step["prompt"], dict) and isinstance(step["response"], dict)
        assert step["module"] in diagram_modules                       # consistent with diagram


def test_malformed_planner_never_crashes(patched_agent, scripted_chat):
    # Planner returns pure garbage on every turn -> pipeline must degrade, not raise.
    patched_agent.install(scripted_chat(planner_reply="this is not json at all"))
    out = patched_agent.module.run_agent("2 days in Kyoto")
    assert set(out.keys()) == {"response", "steps", "state", "branch", "usage", "ms"}
    assert "caveats" in out["response"].lower()   # degradation is surfaced to the user


def test_over_budget_warning_folded_into_response(patched_agent, scripted_chat):
    pricey = json.dumps({"thought": "done", "done": True, "draft_plan": {"days": [
        {"day": 1, "title": "Lux", "items": [
            {"time": "09:00", "name": "Suite", "duration_min": 60, "cost_eur": 5000, "note": "x"}]}],
        "total_cost_eur": 5000}})
    patched_agent.install(scripted_chat(planner_reply=pricey))
    out = patched_agent.module.run_agent("2 days in Kyoto, budget trip")
    assert "caveats" in out["response"].lower() and "exceeds" in out["response"].lower()


def test_unknown_destination_warns(patched_agent, scripted_chat, monkeypatch):
    monkeypatch.setattr(patched_agent.module, "geocode_place", lambda name: None)  # geocode fails
    patched_agent.install(scripted_chat())
    out = patched_agent.module.run_agent("2 days on Mars")
    assert "could not locate" in out["response"].lower()


def test_no_tier2_tool_in_trace(patched_agent, scripted_chat):
    patched_agent.install(scripted_chat())
    out = patched_agent.module.run_agent("2 days in Kyoto")
    blob = json.dumps(out["steps"]).lower()
    assert "flight_book_tool" not in blob and "booking_confirm_tool" not in blob


def test_map_links_are_built_from_venue_not_the_activity_label():
    """Regression: links were built from `name`, a human label like "Popular spot — stroll
    and photos". Maps turned that into a keyword search that pinned nothing in particular."""
    from agent.agent import _with_map_links
    plan = {"days": [{"day": 1, "items": [
        {"name": "Popular spot — stroll and photos", "venue": "Kinkaku-ji"}]}]}
    url = _with_map_links(plan, "Kyoto")["days"][0]["items"][0]["map_url"]
    assert url.startswith("https://www.google.com/maps/search/?api=1&query=")
    assert "Kinkaku-ji" in url and "Kyoto" in url
    assert "stroll" not in url and "Popular" not in url


def test_items_without_a_venue_get_no_link(patched_agent):
    """No link beats a link to the wrong place."""
    from agent.agent import _with_map_links
    plan = {"days": [{"day": 1, "items": [
        {"name": "Hotel check-in / drop luggage", "venue": ""},
        {"name": "Free afternoon — wander the old town", "venue": ""},
        {"name": "Castle visit", "venue": "Karlštejn Castle"},
    ]}]}
    items = _with_map_links(plan, "Prague")["days"][0]["items"]
    assert "map_url" not in items[0]
    assert "map_url" not in items[1]
    assert "map_url" in items[2]


def test_no_official_site_links_are_fabricated(patched_agent):
    """These used to be Google *search* URLs labelled "official site" — which they were not."""
    from agent.agent import _with_map_links
    plan = {"days": [{"day": 1, "items": [
        {"name": "Castle", "venue": "Karlštejn Castle", "site_url": "https://stale.example"}]}]}
    item = _with_map_links(plan, "Prague")["days"][0]["items"][0]
    assert "site_url" not in item


def test_transfers_link_to_directions_from_the_previous_stop(patched_agent):
    """A pin on "Transfer to Old Town" is useless; the way there from the hotel is not.

    Both signals are required: a venue_from AND a name that describes movement.
    """
    from agent.agent import _with_map_links
    plan = {"days": [{"day": 1, "items": [
        {"name": "Check in", "venue": "Prague Bank Hotel"},
        {"name": "Transfer to Old Town", "venue_from": "Prague Bank Hotel",
         "venue": "Old Town Square"},
        {"name": "Dinner", "venue": "Lokál Dlouhá"},
    ]}]}
    items = _with_map_links(plan, "Prague")["days"][0]["items"]
    assert not items[0].get("is_leg")
    assert items[1]["is_leg"] and "/maps/dir/" in items[1]["map_url"]
    assert "Prague+Bank+Hotel" in items[1]["map_url"]      # from where they just were
    assert "Old+Town+Square" in items[1]["map_url"]
    assert not items[2].get("is_leg")                      # a meal is a place, not a leg



def test_no_area_map_link_is_emitted(patched_agent):
    """Dropped: a map centred on Prague tells someone already in Prague nothing, and the
    per-day route already shows the stops."""
    from agent.agent import _with_map_links
    out = _with_map_links({"days": [{"day": 1, "items": [{"name": "x", "venue": "Charles Bridge"}]}]},
                          "Prague", {"lat": 50.0755, "lon": 14.4378})
    assert "area_map_url" not in out



def test_confirmation_is_readable_not_a_json_dump(patched_agent):
    from agent.agent import _confirmation_message, CONFIRM_MARKER
    msg = _confirmation_message({"destination": "Prague", "days": 4,
                                 "group": "2 friends", "budget": "mid-range"})
    assert "```" not in msg and '"destination"' not in msg
    assert "**Destination:** Prague" in msg
    assert CONFIRM_MARKER in msg
    # Unset interests must NOT appear as a row: listing it as "not specified" made the model
    # read the summary as incomplete and refuse to accept "yes". It is an invitation after
    # the question instead.
    assert "- **Interests:**" not in msg
    assert "food, culture, nature" in msg


def test_confirmation_lists_interests_when_supplied(patched_agent):
    from agent.agent import _confirmation_message
    msg = _confirmation_message({"destination": "Prague", "days": 4, "group": "2 friends",
                                 "budget": "mid-range", "style": "nature"})
    assert "- **Interests:** nature" in msg
    assert "food, culture, nature" not in msg          # no nudge once it is known


def test_plain_yes_confirms_even_if_the_model_says_otherwise(patched_agent, scripted_chat):
    """Regression: the agent looped forever on 'yes', re-showing the summary each turn.
    The deterministic backstop must accept an unambiguous agreement."""
    from agent.agent import _is_affirmative, _asked_to_confirm, _latest_user_message
    convo = ("User: 4 days in Prague, mid-range\n"
             "Agent: **Here's your trip so far**\n\nDoes this look right? Reply **yes**…\n"
             "User: yes")
    assert _latest_user_message(convo) == "yes"
    assert _asked_to_confirm(convo)
    assert _is_affirmative("yes") and _is_affirmative("Yes.") and _is_affirmative("go ahead")
    # An agreement carrying an edit is not a bare confirmation; the model handles those.
    assert not _is_affirmative("yes but make it cheaper")


# --- Planner is bounded by time, not by a step count -----------------------------------

def test_leg_uses_the_declared_start_not_just_the_previous_item(patched_agent):
    from agent.agent import _with_map_links
    plan = {"days": [{"day": 1, "items": [
        {"name": "Check in", "venue": "Prague Bank Hotel"},
        {"name": "Ride to the castle", "venue_from": "Malostranská", "venue": "Prague Castle"},
    ]}]}
    leg = _with_map_links(plan, "Prague")["days"][0]["items"][1]
    assert leg["is_leg"] and "/maps/dir/" in leg["map_url"]
    assert "Malostransk" in leg["map_url"]          # the declared start wins
    assert "Bank+Hotel" not in leg["map_url"]



def test_map_query_does_not_repeat_the_city(patched_agent):
    """Planners return "Václav Havel Airport Prague"; appending the city gave "Prague Prague"."""
    from agent.agent import _map_query
    assert _map_query("Václav Havel Airport Prague", "Prague").count("Prague") == 1
    assert _map_query("Old Town Square", "Prague").endswith("Prague")


def test_free_form_details_survive_into_the_planner_profile(patched_agent):
    from agent.schemas import validate_profile, compact_profile
    p = validate_profile({"destination": "Prague", "days": 3, "budget": "low", "group": "solo",
                          "details": ["we have a rental car", "no early mornings", "  "]})
    assert p["details"] == ["we have a rental car", "no early mornings"]   # blanks dropped
    assert compact_profile(p)["details"] == p["details"]


def test_no_day_level_route_link_is_emitted(patched_agent):
    """Dropped: every stop already links to itself and transfers link to their own
    directions, so a third link at day level restated the bullets above it."""
    from agent.agent import _with_map_links
    day = _with_map_links({"days": [{"day": 1, "items": [
        {"name": "Check in", "venue": "Prague Bank Hotel"},
        {"name": "Walk to the square", "venue_from": "Prague Bank Hotel",
         "venue": "Old Town Square"}]}]}, "Prague")["days"][0]
    assert "route_url" not in day


def test_only_movement_links_to_directions(patched_agent):
    """A route link belongs on a journey; everything else gets a pin on one place."""
    from agent.agent import _with_map_links
    items = _with_map_links({"days": [{"day": 1, "items": [
        {"name": "Prague Castle visit", "venue": "Prague Castle"},
        {"name": "Dinner", "venue": "Lokál Dlouhá"},
        {"name": "Transfer to the airport", "venue_from": "Lokál Dlouhá",
         "venue": "Václav Havel Airport Prague"},
    ]}]}, "Prague")["days"][0]["items"]
    assert "/maps/search/" in items[0]["map_url"] and not items[0].get("is_leg")
    assert "/maps/search/" in items[1]["map_url"] and not items[1].get("is_leg")
    assert "/maps/dir/" in items[2]["map_url"] and items[2]["is_leg"]


def test_prompts_forbid_leaking_tool_names_to_the_traveller(patched_agent):
    """The agent wrote "maps returned UNKNOWN hours — check official site". The advice is
    fine; naming the lookup is not. The traveller has no idea what maps_tool is."""
    import inspect
    from agent import agent as mod
    src = inspect.getsource(mod._plan)
    assert "never mention tools or lookups" in src
    assert "check opening" in src and "before you go" in src   # the traveller-facing wording


def test_confirmation_card_folds_the_anchors_into_two_lines(patched_agent):
    from agent.agent import _confirmation_message
    msg = _confirmation_message({
        "destination": "Prague", "days": 4, "group": "2 friends", "budget": "mid-range",
        "start_time": "17/8 15:00", "end_time": "20/8 18:30",
        "start_point": "Prague Bank Hotel", "lodging": "Prague Bank Hotel",
        "end_point": "Václav Havel Airport Prague",
        "details": ["we have a rental car", "no early mornings"]})
    assert "- **Trip window:** 17/8 15:00 → 20/8 18:30" in msg
    assert "- **Start point:** Prague Bank Hotel" in msg
    assert "- **End point:** Václav Havel Airport Prague" in msg
    assert "- **Also noted:** we have a rental car; no early mornings" in msg
    # The anchors no longer get a row each.
    for gone in ("Starts:", "Ends:", "Starting at:", "Ending at:", "Staying at:",
                 "Departing from:"):
        assert gone not in msg
    assert len([l for l in msg.splitlines() if l.startswith("- ")]) == 8


def test_only_movement_gets_a_route_even_with_venue_from(patched_agent):
    """Regression: venue_from alone triggered a route, and the planner sets it on ordinary
    stops too — so museum visits rendered as driving directions."""
    from agent.agent import _with_map_links
    items = _with_map_links({"days": [{"day": 1, "items": [
        {"name": "Prague Castle visit", "venue_from": "Hotel", "venue": "Prague Castle"},
        {"name": "Head back to Old Town for dinner", "venue_from": "Prague Castle",
         "venue": "Lokál Dlouhá"},
        {"name": "Walk to Charles Bridge", "venue_from": "Lokál Dlouhá",
         "venue": "Charles Bridge"},
    ]}]}, "Prague")["days"][0]["items"]
    assert not items[0].get("is_leg")          # a visit is a place
    assert not items[1].get("is_leg")          # "head back to" describes a stop, not a journey
    assert items[2]["is_leg"]                  # "walk to" is movement


def test_origin_is_gone_from_the_profile(patched_agent):
    from agent.schemas import validate_profile, compact_profile
    p = validate_profile({"destination": "Prague", "days": 3, "budget": "low",
                          "group": "solo", "origin": "Tel Aviv"})
    assert "origin" not in p and "origin" not in compact_profile(p)


def test_card_shows_walking_and_the_stated_amount(patched_agent):
    """Both were captured but invisible, so the traveller assumed they were ignored."""
    from agent.agent import _confirmation_message
    msg = _confirmation_message({
        "destination": "Prague", "days": 4, "group": "2 friends", "budget": "mid-range",
        "budget_amount_eur": 2000, "budget_basis": "total", "walking": "high"})
    assert "- **Budget:** mid-range (~€2,000 total)" in msg
    assert "- **Walking:** high" in msg


def test_card_hides_the_default_walking_level(patched_agent):
    """Showing "moderate" implies the traveller chose it; it is just the default."""
    from agent.agent import _confirmation_message
    msg = _confirmation_message({"destination": "Prague", "days": 4, "group": "solo",
                                 "budget": "low", "walking": "moderate"})
    assert "Walking" not in msg



def test_the_planner_prompt_stays_small(patched_agent):
    """A long instruction set makes every planner call slower, because a reasoning model
    spends longer thinking before it answers. Rules the deterministic audit already enforces
    — budget ceiling, meal costs, gaps, day balance — were removed for that reason: the
    audit still catches them and the critic still reports them.
    """
    import inspect
    from agent import agent as mod
    src = inspect.getsource(mod._plan)
    prompt = src[src.index("sys = ("):src.index('user = "Traveller profile:')]
    assert len(prompt) < 3200, f"planner prompt is {len(prompt)} chars and calls will be slow"
    # What the audit cannot check must still be instructed.
    for rule in ("start_time", "lodging", "details", "venue_from", "live prices"):
        assert rule in prompt, f"dropped a rule nothing else enforces: {rule}"

def test_confirmation_card_shows_every_field_the_traveller_supplied(patched_agent):
    """The card is what the traveller checks before planning starts, so anything they said
    must be visible on it. This test fails when a new profile field is added without a row —
    that is the point: silently dropping input is how "a lot of walking" looked ignored.
    """
    from agent.schemas import validate_profile
    from agent.agent import _confirmation_message
    prof = validate_profile({
        "destination": "Prague", "days": 4, "group": "2 friends", "budget": "mid-range",
        "budget_amount_eur": 2000, "budget_basis": "total", "style": "nature",
        "when": "August", "start_point": "Prague Bank Hotel", "start_time": "17/8 15:00",
        "end_point": "Václav Havel Airport", "end_time": "20/8 18:30",
        "lodging": "Hotel Ibis Old Town", "details": ["rental car"],
        "dietary": ["vegetarian"], "walking": "high", "accessibility": True,
        "priorities": ["Charles Bridge"], "avoid": ["crowds"]})
    card = _confirmation_message(prof).lower()

    for key, value in prof.items():
        if not value or key == "assumptions":
            continue
        if isinstance(value, bool):                    # rendered as prose, not a raw value
            assert key.replace("_", " ") in card, key
            continue
        for item in (value if isinstance(value, list) else [value]):
            text = str(item).lower()
            # Numbers are formatted with thousands separators on the card.
            assert text in card or f"{int(text):,}" in card if text.isdigit() else text in card, key


def test_exact_times_survive_alongside_a_season(patched_agent):
    """Regression: "August" suppressed the Trip window row, so a stated arrival hour that
    the planner is required to honour was invisible to the traveller."""
    from agent.agent import _confirmation_message
    msg = _confirmation_message({"destination": "Prague", "days": 4, "group": "solo",
                                 "budget": "low", "when": "August",
                                 "start_time": "17/8 15:00", "end_time": "20/8 18:30"})
    assert "- **Dates:** August" in msg
    assert "- **Trip window:** 17/8 15:00 → 20/8 18:30" in msg


def test_lodging_shows_only_when_it_is_not_already_the_start_point(patched_agent):
    from agent.agent import _confirmation_message
    same = _confirmation_message({"destination": "Prague", "days": 2, "group": "solo",
                                  "budget": "low", "start_point": "Hotel Ibis",
                                  "lodging": "Hotel Ibis"})
    assert same.count("Hotel Ibis") == 1                 # no duplicate row
    other = _confirmation_message({"destination": "Prague", "days": 2, "group": "solo",
                                   "budget": "low", "start_point": "Main Station",
                                   "lodging": "Hotel Ibis"})
    assert "- **Staying at:** Hotel Ibis" in other



def test_duplicate_date_rows_collapse_to_the_precise_one(patched_agent):
    from agent.agent import _confirmation_message
    dupe = _confirmation_message({"destination": "Prague", "days": 4, "group": "2 friends",
                                  "budget": "low", "when": "17/8 - 20/8",
                                  "start_time": "17/8 15:00", "end_time": "20/8 18:30"})
    assert "- **Dates:**" not in dupe                      # the vaguer copy goes
    assert "- **Trip window:** 17/8 15:00 → 20/8 18:30" in dupe

    # A season is not a duplicate of exact times; both are kept.
    both = _confirmation_message({"destination": "Prague", "days": 4, "group": "2 friends",
                                  "budget": "low", "when": "August",
                                  "start_time": "17/8 15:00", "end_time": "20/8 18:30"})
    assert "- **Dates:** August" in both and "- **Trip window:**" in both


def test_intake_records_the_group_verbatim(patched_agent):
    """The model turned "2 friends" into "user + 2 friends (3 people)" — plausible
    arithmetic, but it rewrites what the traveller said and reads as a correction."""
    import inspect
    from agent import agent as mod
    src = inspect.getsource(mod._profile)
    assert "Record group EXACTLY as the traveller phrased it" in src
    assert "Never add the traveller to the count" in src


def test_planner_completes_a_real_plan_at_realistic_call_speeds(patched_agent):
    """End-to-end guard on the time budget, with the clock faked so it runs instantly.

    This is the test that would have caught the placeholder regression: with the reserve at
    110s against a 180s budget, even a fast 25s/call run broke out of the loop after one
    step and returned a skeleton.
    """
    from unittest.mock import patch
    from agent import agent as mod

    for call_seconds in (25, 40):
        calls, clock = {"n": 0}, {"t": 0.0}

        def fake_chat(msgs, **kw):
            calls["n"] += 1
            clock["t"] += call_seconds
            if calls["n"] >= 4:                      # three tool calls, then finalize
                return {"thought": "done", "done": True, "draft_plan": {
                    "days": [{"day": 1, "title": "D1", "items": [
                        {"time": "09:00", "name": "Castle", "venue": "Prague Castle",
                         "duration_min": 120, "cost_eur": 10}]}],
                    "total_cost_eur": 10}}
            return {"thought": "check", "tool": "weather_tool",
                    "tool_input": {"location": f"Prague{calls['n']}"}}

        with patch.object(mod, "_chat_json", fake_chat), \
             patch.object(mod.time, "monotonic", lambda: clock["t"]):
            plan = mod._plan({"destination": "Prague", "days": 4, "group": "2 friends",
                              "budget": "mid-range"}, [], run_id="t",
                             deadline=mod.MAX_RUN_SECONDS)

        assert not plan.get("degraded"), f"{call_seconds}s/call produced placeholder days"
        assert plan["total_cost_eur"] > 0
        assert calls["n"] >= 4, f"only {calls['n']} calls at {call_seconds}s/call"


def test_slow_real_tools_cannot_starve_the_plan(patched_agent):
    """Regression: tools became real HTTP calls, and a planner told to verify opening hours
    spent the whole run on the network and returned placeholder days.

    Guards the combination that actually broke: a planner that keeps wanting lookups, and
    lookups that are slow. A real itinerary must still come back.
    """
    from unittest.mock import patch
    from agent import agent as mod

    done = {"thought": "done", "done": True, "draft_plan": {
        "days": [{"day": 1, "title": "D1", "items": [
            {"time": "09:00", "name": "Castle", "venue": "Prague Castle",
             "duration_min": 120, "cost_eur": 10}]}], "total_cost_eur": 10}}

    for call_s, tool_s in ((30, 24), (35, 12), (40, 6)):
        clock, calls = {"t": 0.0}, {"n": 0}

        def fake_chat(msgs, **kw):
            calls["n"] += 1
            clock["t"] += call_s
            last = msgs[-1]["content"]
            # A compliant model finalizes when told to stop or when research is cut off.
            if "Stop now" in last or "budget for this run is used up" in last:
                return done
            return {"thought": "look", "tool": "maps_tool",
                    "tool_input": {"query": f"site{calls['n']}"}}

        def fake_tool(tool, tool_input):
            clock["t"] += tool_s
            return {"ok": True, "results": [{"name": "x", "open_hours": None}]}

        with patch.object(mod, "_chat_json", fake_chat), \
             patch.object(mod, "run_tool", fake_tool), \
             patch.object(mod.time, "monotonic", lambda: clock["t"]):
            plan = mod._plan({"destination": "Prague", "days": 4, "group": "2 friends",
                              "budget": "mid-range"}, [], run_id="t",
                             deadline=mod.MAX_RUN_SECONDS)

        assert not plan.get("timed_out"), f"{call_s}s/{tool_s}s ran out of time"
        assert not plan.get("degraded"), f"{call_s}s/{tool_s}s produced placeholders"
        assert plan["total_cost_eur"] > 0


def test_maps_tool_makes_at_most_two_http_calls(monkeypatch):
    """Details were fetched for every hit: four round trips per lookup, ~24s of network for
    one venue. Only the best match gets a details call now."""
    from agent import tools
    monkeypatch.setattr(tools, "GEOAPIFY_KEY", "k")
    seen = []

    def fake_get(url, params):
        seen.append(url)
        if "geocode" in url:
            return {"results": [{"name": "A", "place_id": "p1", "lat": 1, "lon": 1},
                                {"name": "B", "place_id": "p2", "lat": 2, "lon": 2}]}
        return {"features": [{"properties": {"opening_hours": "Mo-Su 09:00-17:00"}}]}

    monkeypatch.setattr(tools, "_http_get", fake_get)
    out = tools.maps_tool(query="Prague Castle", near="Prague")
    assert len(seen) == 2, seen
    assert out["results"][0]["open_hours"] == "Mo-Su 09:00-17:00"


# --- Deterministic audit is wired into the pipeline -------------------------------------
def test_meal_venues_that_name_a_restaurant_get_no_map_link(patched_agent):
    """A map pin asserts the place is real and findable. For a restaurant the model recalled
    from memory, we do not know that — it may be closed, renamed, or invented."""
    from agent.agent import _with_map_links
    items = _with_map_links({"days": [{"day": 1, "items": [
        {"name": "Lunch", "venue": "Lokál Dlouhá Restaurant"},
        {"name": "Dinner in Malá Strana", "venue": "Malá Strana"},
        {"name": "Prague Castle", "venue": "Prague Castle"},
    ]}]}, "Prague")["days"][0]["items"]
    assert "map_url" not in items[0]              # unverifiable business name
    assert "map_url" in items[1]                  # an area is real by construction
    assert "map_url" in items[2]                  # not a meal; unaffected


def test_deterministic_findings_reach_the_critic_and_survive_it(patched_agent):
    """The critic may drop or reword them, so computed defects are re-asserted afterwards."""
    from unittest.mock import patch
    from agent import agent as mod

    draft = {"days": [{"day": 1, "items": [
        {"time": "09:00", "name": "Castle", "venue": "Prague Castle", "duration_min": 180}]}],
        "total_cost_eur": 15}
    seen = {}

    def fake_chat(msgs, **kw):
        seen["user"] = msgs[-1]["content"]
        return {"verdict": "PASS", "must_fix": [], "be_aware": [], "fixes": []}

    with patch.object(mod, "_chat_json", fake_chat):
        verdict = mod._reflect({"destination": "Prague"}, draft, [], run_id="t",
                               found=["Day 1: something is wrong"])

    assert "VERIFIED defects" in seen["user"]
    assert "Day 1: something is wrong" in verdict["must_fix"]
    assert verdict["verdict"] == "FAIL"           # a real defect cannot be a PASS


def test_forecast_is_handed_to_the_planner(patched_agent):
    """Fetched once off the geocode we already did, so weather shapes the plan without the
    planner spending a tool turn asking for it."""
    from unittest.mock import patch
    from agent import agent as mod
    seen = {}

    def fake_chat(msgs, **kw):
        seen["user"] = msgs[-1]["content"]
        return {"thought": "done", "done": True,
                "draft_plan": {"days": [], "total_cost_eur": 0}}

    with patch.object(mod, "_chat_json", fake_chat):
        mod._plan({"destination": "Prague", "days": 2, "group": "solo", "budget": "low"}, [],
                  run_id="t", forecast=[{"date": "17/8", "rain_pct": 80}])
    assert "Forecast" in seen["user"] and "80" in seen["user"]


def test_opening_hours_are_remembered_from_tool_observations(patched_agent):
    from agent.agent import _remember_hours
    cache = {}
    _remember_hours(cache, {"results": [{"name": "Prague Castle", "open_hours": "Mo-Su 09:00-17:00"},
                                        {"name": "Unknown Place", "open_hours": None}]})
    assert cache == {"Prague Castle": "Mo-Su 09:00-17:00"}   # unknown hours are not stored
    _remember_hours(cache, None)                              # must not raise
    _remember_hours(None, {"results": []})


# --- Measured travel times --------------------------------------------------------------
def _routing_day():
    return {"days": [{"day": 1, "title": "D1", "items": [
        {"time": "09:00", "name": "Castle", "venue": "Prague Castle",
         "duration_min": 120, "cost_eur": 15},
        {"time": "11:00", "name": "Walk to Charles Bridge", "venue_from": "Prague Castle",
         "venue": "Charles Bridge", "duration_min": 10, "cost_eur": 0, "note": "Short stroll"},
        {"time": "11:10", "name": "Charles Bridge", "venue": "Charles Bridge",
         "duration_min": 45, "cost_eur": 0}]}], "total_cost_eur": 15}


_COORDS = {"Prague Castle": (50.0900, 14.4000), "Charles Bridge": (50.0865, 14.4114)}


def test_measured_travel_time_replaces_the_guess_and_shifts_the_day(patched_agent):
    """A 10-minute guess that is really 35 makes every later time in the day wrong."""
    from unittest.mock import patch
    from agent import agent as mod
    with patch.object(mod, "route_matrix", lambda pts, mode="walk": [35]):
        fixed, changed = mod._apply_travel_times(_routing_day(), {"details": []}, _COORDS)
    items = fixed["days"][0]["items"]
    assert changed == 1
    assert items[1]["duration_min"] == 35
    assert items[2]["time"] == "11:35"            # the day moved with it
    assert "35 min on foot" in items[1]["note"]
    assert "matrix" not in items[1]["note"].lower()   # no tool names in traveller text


def test_routing_failure_leaves_the_plan_untouched(patched_agent):
    """An itinerary with estimated walking times is fine; one that never arrives is not."""
    from unittest.mock import patch
    from agent import agent as mod

    def boom(*a, **k):
        raise mod.ToolError("network", "down")

    with patch.object(mod, "route_matrix", boom):
        same, changed = mod._apply_travel_times(_routing_day(), {"details": []}, _COORDS)
    assert changed == 0
    assert same["days"][0]["items"][1]["duration_min"] == 10


def test_routing_is_skipped_without_known_coordinates(patched_agent):
    """Only venues a lookup already returned are routed — nothing is geocoded to fill gaps."""
    from agent import agent as mod
    _, changed = mod._apply_travel_times(_routing_day(), {"details": []}, {})
    assert changed == 0


def test_a_rental_car_switches_the_routing_mode(patched_agent):
    from unittest.mock import patch
    from agent import agent as mod
    seen = {}

    def fake(points, mode="walk"):
        seen["mode"] = mode
        return [20]

    with patch.object(mod, "route_matrix", fake):
        mod._apply_travel_times(_routing_day(), {"details": ["we have a rental car"]}, _COORDS)
    assert seen["mode"] == "drive"


def test_route_matrix_sends_one_request_for_a_whole_day(monkeypatch):
    """One call per day, not one per leg — this shares the run's time budget."""
    from agent import tools
    monkeypatch.setattr(tools, "GEOAPIFY_KEY", "k")
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"sources_to_targets": [[{"time": 0}, {"time": 600}, {"time": 900}],
                                           [{"time": 600}, {"time": 0}, {"time": 300}],
                                           [{"time": 900}, {"time": 300}, {"time": 0}]]}

    monkeypatch.setattr(tools.requests, "post",
                        lambda url, **kw: (calls.append(url), Resp())[1])
    legs = tools.route_matrix([(50.09, 14.40), (50.08, 14.41), (50.07, 14.42)])
    assert legs == [10, 5]                        # consecutive legs, in minutes
    assert len(calls) == 1


# --- Time budget -------------------------------------------------------------------------
def test_calls_are_truncated_to_the_remaining_wall_time():
    """The wall lets MAX_RUN_SECONDS rise without risking a platform timeout: a call started
    late can no longer overshoot, because it is given only the time that is left."""
    import time as _t
    from agent import llm
    try:
        llm.set_wall(_t.monotonic() + 25)
        assert llm._call_timeout() == 24 or llm._call_timeout() == 25
        llm.set_wall(_t.monotonic() + 500)
        assert llm._call_timeout() == llm._TIMEOUT_S     # never more than the ceiling
        llm.set_wall(_t.monotonic() - 10)                # already past it
        assert llm._call_timeout() == 5                  # a floor, not a negative timeout
    finally:
        llm.set_wall(None)


def test_the_run_budget_fits_inside_the_platform_limit():
    from agent.agent import MAX_RUN_SECONDS, HARD_WALL_SECONDS
    assert MAX_RUN_SECONDS < HARD_WALL_SECONDS
    assert HARD_WALL_SECONDS <= 290, "must leave room to serialise the response before 300s"
    assert MAX_RUN_SECONDS >= 200, "the old 180 left ~90s of the platform budget unused"


def test_open_ended_blocks_get_no_map_link(patched_agent):
    from agent.agent import _with_map_links
    items = _with_map_links({"days": [{"day": 1, "items": [
        {"name": "Free time / shopping", "venue": "Wenceslas Square"},
        {"name": "Prague Castle", "venue": "Prague Castle"},
    ]}]}, "Prague")["days"][0]["items"]
    assert "map_url" not in items[0]
    assert "map_url" in items[1]


# --- The critic's findings must actually get fixed ---------------------------------------
def test_the_planner_gets_a_chance_to_fix_what_the_critic_found():
    """One cycle meant defects were reported to the traveller but never repaired."""
    from agent.agent import MAX_REFLECT_CYCLES
    assert MAX_REFLECT_CYCLES >= 2


def test_computed_defects_are_rechecked_after_the_fix(patched_agent):
    """Only defects that survived the re-plan should reach the traveller."""
    from agent.agent import _is_computed_defect
    assert _is_computed_defect("Day 3: 'A' runs to 09:15 but 'B' starts at 09:10.")
    assert _is_computed_defect("Only 0 min between the last activity and the departure.")
    assert _is_computed_defect("The last day runs to 20:00, past the 18:30 departure.")
    # A judgement cannot be re-verified in code, so it is kept as written.
    assert not _is_computed_defect("Day 2 feels rushed for a family with young children.")


def test_bullets_are_spaced_even_if_the_model_runs_them_together(patched_agent):
    """Markdown needs a blank line between items or they render as one block of text."""
    from agent.agent import _space_bullets
    out = _space_bullets("## Day 1\n- **09:00** — [A](u)\n  - tip\n- **11:00** — [B](u)\n  - tip")
    assert "\n\n- **11:00**" in out
    assert "## Day 1\n- **09:00**" in out      # no blank line after a heading
    assert "  - tip\n- " not in out            # sub-lines stay with their bullet


def test_live_prices_are_fetched_once_and_handed_to_the_planner(patched_agent):
    """One search for the whole trip. Costing from model memory produces numbers years out
    of date; looking each item up would cost a reasoning cycle per item."""
    from unittest.mock import patch
    from agent import agent as mod
    calls, seen = [], {}

    def fake_tool(tool, tool_input):
        calls.append(tool)
        return {"ok": True, "answer": "Coffee EUR4-8, lunch EUR15-20, castle entry EUR18.",
                "snippets": [{"content": "Museum tickets are typically EUR12-18."}]}

    with patch.object(mod, "run_tool", fake_tool):
        prices = mod._prices_for({"destination": "Prague"})
    assert calls == ["search_tool"], "should be one call for the trip, not one per item"
    assert "castle entry" in prices["summary"]

    def fake_json(msgs, **kw):
        seen["user"] = msgs[-1]["content"]
        return {"thought": "done", "done": True,
                "draft_plan": {"days": [], "total_cost_eur": 0}}

    with patch.object(mod, "_chat_json", fake_json):
        mod._plan({"destination": "Prague", "days": 2, "group": "solo", "budget": "low"}, [],
                  run_id="t", prices=prices)
    assert "Current local prices" in seen["user"] and "castle entry" in seen["user"]


def test_a_failed_price_search_never_blocks_the_plan(patched_agent):
    """Prices are a bonus. A search that fails, times out or returns sample data must leave
    the planner working from its own estimates rather than stopping."""
    from unittest.mock import patch
    from agent import agent as mod

    def boom(tool, tool_input):
        raise RuntimeError("upstream down")

    with patch.object(mod, "run_tool", boom):
        assert mod._prices_for({"destination": "Prague"}) is None
    # Sample data is not real prices, so it is refused rather than passed off as live.
    with patch.object(mod, "run_tool", lambda t, i: {"ok": True, "fictive": True,
                                                     "answer": "x", "snippets": []}):
        assert mod._prices_for({"destination": "Prague"}) is None
    assert mod._prices_for({"destination": ""}) is None


def test_the_call_timeout_is_evaluated_per_call_not_per_client():
    """The client is cached and outlives a request on a warm container. A timeout fixed at
    construction is whatever the first request computed and never shrinks, so the wall stops
    being enforced and a late call can run past the platform limit."""
    import inspect
    import time as _t
    from agent import llm
    assert "timeout=" not in inspect.getsource(llm._get_client)
    src = inspect.getsource(llm.chat)
    assert 'timeout, retries = _call_budget()' in src
    assert 'base["timeout"] = timeout' in src
    # The SDK retries per attempt, so the retry budget has to shrink with the timeout or
    # the wall bounds one attempt while the call runs for three.
    assert "with_options(max_retries=retries)" in src
    try:
        llm.set_wall(_t.monotonic() + 20)
        assert llm._call_timeout() <= 20
        llm.set_wall(_t.monotonic() + 1000)
        assert llm._call_timeout() == llm._TIMEOUT_S     # never beyond the ceiling
        llm.set_wall(_t.monotonic() - 5)                 # already past it
        assert llm._call_timeout() == 5                  # a floor, not a negative timeout
    finally:
        llm.set_wall(None)


def test_the_pipeline_cannot_run_past_the_platform_limit(patched_agent):
    """Vercel kills the function at 300s. Worst path — the planner always wants another tool
    and the critic always fails — with the clock faked and the wall genuinely applied."""
    from unittest.mock import patch
    from agent import agent as mod
    from agent import llm

    done = {"thought": "done", "done": True, "draft_plan": {
        "days": [{"day": 1, "title": "D", "items": [
            {"time": "09:00", "name": "Castle", "venue": "Prague Castle",
             "duration_min": 120, "cost_eur": 10}]}], "total_cost_eur": 10}}

    for call_s, tool_s in ((25, 4), (45, 10), (90, 15)):
        clock, calls = {"t": 0.0}, {"n": 0}

        def spend():
            """A call takes what it wants, or what the wall allows — whichever is less."""
            clock["t"] += min(call_s, max(5, llm._call_timeout()))

        def fake_json(msgs, **kw):
            calls["n"] += 1
            spend()
            body = msgs[-1]["content"]
            if "Draft:" in body or "VERIFIED" in body:
                return {"verdict": "FAIL", "must_fix": ["Day 1: invented"],
                        "be_aware": [], "fixes": []}
            if "Stop now" in body or "budget for this run" in body:
                return done
            return {"thought": "x", "tool": "maps_tool",
                    "tool_input": {"query": f"q{calls['n']}"}}

        with patch.object(mod, "_chat_json", fake_json), \
             patch.object(mod, "run_tool", lambda t, i: (clock.__setitem__(
                 "t", clock["t"] + tool_s), {"ok": True, "results": []})[1]), \
             patch.object(mod, "chat", lambda m, **k: (spend(), "# Itinerary")[1]), \
             patch.object(mod.time, "monotonic", lambda: clock["t"]), \
             patch.object(llm.time, "monotonic", lambda: clock["t"]), \
             patch.object(mod, "_safe_geocode", lambda x: {"lat": 50.0, "lon": 14.4}), \
             patch.object(mod, "_forecast_for", lambda *a: None), \
             patch.object(mod, "_prices_for", lambda *a: None), \
             patch.object(mod, "_apply_travel_times", lambda p, pr, c, r=None: (p, 0)), \
             patch.object(mod, "_profile", lambda *a, **k: {
                 "profile": {"destination": "Prague", "days": 4, "group": "2 friends",
                             "budget": "mid-range"},
                 "missing": [], "confirmed": True, "question": "", "intent": "replace"}):
            mod._run_turn("User: x\nAgent: Does this look right?\nUser: yes", [])

        assert clock["t"] <= mod.HARD_WALL_SECONDS + 5, (
            f"{call_s}s/call ran to {clock['t']:.0f}s — past the wall")



def test_repeated_lookup_failures_stop_the_research(patched_agent):
    """Each retry is a whole LLM turn, so a tool that will not answer can eat a run. After a
    couple of failures the planner is told to write the plan from what it has."""
    from unittest.mock import patch
    from agent import agent as mod
    seen = []

    def fake_tool(tool, tool_input):
        return {"ok": False, "error_type": "http_400", "note": "Upstream returned an error"}

    def fake_json(msgs, **kw):
        body = msgs[-1]["content"]
        seen.append(body)
        if "stop researching" in body or "Stop now" in body:
            return {"thought": "done", "done": True, "draft_plan": {
                "days": [{"day": 1, "title": "D", "items": [
                    {"time": "09:00", "name": "Castle", "venue": "Prague Castle",
                     "duration_min": 120, "cost_eur": 10}]}], "total_cost_eur": 10}}
        return {"thought": "look", "tool": "maps_tool",
                "tool_input": {"query": f"attempt {len(seen)}"}}

    with patch.object(mod, "_chat_json", fake_json), patch.object(mod, "run_tool", fake_tool):
        plan = mod._plan({"destination": "Prague", "days": 2, "group": "solo",
                          "budget": "low"}, [], run_id="t")

    assert any("stop researching" in b for b in seen), "never told to give up"
    assert not plan.get("degraded"), "a real plan should still come back"
    assert len(seen) <= 5, f"burned {len(seen)} turns on a failing tool"


def test_place_lookups_default_to_the_destination(patched_agent):
    """A bare "Colosseum" resolves to a hamlet in Australia. The planner knows the city and
    omits it as obvious, so supplying it here saves a turn spent noticing and retrying."""
    from agent.agent import _localise
    prof = {"destination": "Rome, Italy"}
    assert _localise("maps_tool", {"query": "Colosseum"}, prof)["near"] == "Rome, Italy"
    # An explicit near wins: the planner may be looking somewhere else on purpose.
    assert _localise("maps_tool", {"query": "X", "near": "Vatican"}, prof)["near"] == "Vatican"
    # Other tools are untouched.
    assert _localise("weather_tool", {"location": "Rome"}, prof) == {"location": "Rome"}
    assert _localise("maps_tool", {"query": "X"}, {"destination": ""}) == {"query": "X"}


def test_a_timed_out_finalize_still_returns_the_places_it_found(patched_agent):
    """The run had the Colosseum's hours and the mall's address when the finalize call timed
    out. Those must reach the traveller rather than being replaced by "self-guided"."""
    from unittest.mock import patch
    from agent import agent as mod
    lookups = [{"thought": "a", "tool": "maps_tool", "tool_input": {"query": "Colosseum"}},
               {"thought": "b", "tool": "maps_tool", "tool_input": {"query": "Pantheon"}}]
    index = {"n": 0}

    def fake_json(msgs, **kw):
        if index["n"] < len(lookups):
            step = lookups[index["n"]]
            index["n"] += 1
            return step
        raise TimeoutError("finalize timed out")

    def fake_tool(tool, tool_input):
        return {"ok": True, "results": [{"name": tool_input["query"],
                                         "lat": 41.9, "lon": 12.5, "open_hours": None}]}

    with patch.object(mod, "_chat_json", fake_json), \
         patch.object(mod, "run_tool", fake_tool), \
         patch.object(mod, "is_timeout", lambda e: True):
        plan = mod._plan({"destination": "Rome", "days": 2, "group": "2 friends",
                          "budget": "mid-range"}, [], run_id="t")

    assert plan["timed_out"] is True          # still honest about what happened
    names = [i["name"] for d in plan["days"] for i in d["items"]]
    assert "Colosseum" in names and "Pantheon" in names
    assert not any("self-guided" in n for n in names)


def test_rendering_details_are_not_reported_as_caveats(patched_agent):
    """Whether the prose came from the model or the deterministic renderer is an internal
    detail. The itinerary is the same either way, and listing it under "not fully validated"
    makes a sound plan look defective."""
    import inspect
    from agent import agent as mod
    src = inspect.getsource(mod)
    assert "The formatter returned nothing" not in src
    assert "Ran out of time before the final polish" not in src



def test_travel_times_never_overwrite_a_visit(patched_agent):
    """Regression: any item with venue_from had its duration replaced by a routing time, so a
    120-minute Colosseum visit became "1 min" and a metro ride became a 275-minute walk. The
    corrupted plan then broke the trip window — day 2 ran past midnight."""
    from unittest.mock import patch
    from agent import agent as mod
    items = [{"time": "10:15", "name": "Travel to RomaEst", "venue_from": "Piazza Navona",
              "venue": "Primark", "duration_min": 75, "cost_eur": 0},
             {"time": "12:00", "name": "Shopping at Primark", "venue_from": "Primark",
              "venue": "Primark", "duration_min": 120, "cost_eur": 0},
             {"time": "16:00", "name": "Colosseum", "venue_from": "Colosseum",
              "venue": "Colosseum", "duration_min": 120, "cost_eur": 22}]
    coords = {"Piazza Navona": (41.899, 12.473), "Primark": (41.913, 12.660),
              "Colosseum": (41.890, 12.492)}

    with patch.object(mod, "route_matrix", lambda pts, mode="walk": [275, 240]):
        out, changed = mod._apply_travel_times({"days": [{"day": 1, "items": items}]},
                                               {"details": []}, coords)
    got = [i["duration_min"] for i in out["days"][0]["items"]]
    assert got == [75, 120, 120], "a visit duration was overwritten by a routing time"
    assert changed == 0, "an implausible cross-city walking time was accepted"


def test_a_plausible_walking_leg_is_still_corrected(patched_agent):
    """The feature still works where it should: a real leg with a believable measured time."""
    from unittest.mock import patch
    from agent import agent as mod
    items = [{"time": "09:00", "name": "Prague Castle", "venue": "Prague Castle",
              "duration_min": 90, "cost_eur": 10},
             {"time": "10:30", "name": "Walk to Charles Bridge", "venue_from": "Prague Castle",
              "venue": "Charles Bridge", "duration_min": 10, "cost_eur": 0}]
    coords = {"Prague Castle": (50.09, 14.40), "Charles Bridge": (50.086, 14.411)}
    with patch.object(mod, "route_matrix", lambda pts, mode="walk": [22]):
        out, changed = mod._apply_travel_times({"days": [{"day": 1, "items": items}]},
                                               {"details": []}, coords)
    assert changed == 1
    assert out["days"][0]["items"][1]["duration_min"] == 22
    assert out["days"][0]["items"][0]["duration_min"] == 90     # the visit is untouched


def test_a_fix_runs_whenever_there_is_time_left(patched_agent):
    """The fix cycle was gated on having room for a whole fresh plan plus another review —
    most of the budget — so it never ran even with 100s spare. A fix patches a draft against
    a list of defects; it is a small call, and the wall truncates it anyway."""
    import inspect
    from agent import agent as mod
    src = inspect.getsource(mod)
    loop = src[src.index("for c in range(MAX_REFLECT_CYCLES)"):]
    guard = loop[:loop.index("draft = _plan(")]
    assert "PLANNER_FINALIZE_RESERVE_S" not in guard, (
        "sizing a fix as a fresh plan is what stopped it running")
    assert "_expired(deadline)" in guard      # the clock still stops it


# --- One budget, one wall ------------------------------------------------------------------
def test_there_is_a_single_time_budget_and_a_wall():
    """Every stage runs until MAX_RUN_SECONDS is gone; the wall then truncates whatever call
    is in flight. Per-stage reserves were removed: sizing a small fix call as if it were a
    fresh plan meant the guard demanded most of the budget be unspent, so the fix, the
    review and the formatter were all skipped while time remained."""
    from agent import agent as mod
    assert mod.MAX_RUN_SECONDS < mod.HARD_WALL_SECONDS
    assert mod.HARD_WALL_SECONDS <= 290, "leave room to serialise the reply before 300s"
    for gone in ("PLANNER_FINALIZE_RESERVE_S", "REVIEW_MIN_SECONDS",
                 "FORMAT_MIN_SECONDS", "DELIVER_RESERVE_S", "_low_on_time"):
        assert not hasattr(mod, gone), f"{gone} should have been removed"


def test_the_finalize_call_runs_even_once_research_time_is_spent(patched_agent):
    """The loop ending on the budget is the normal case, and this call is what turns the
    research into an itinerary. Aborting instead produced placeholder days."""
    import inspect
    from agent import agent as mod
    src = inspect.getsource(mod._plan)
    tail = src[src.index("Safety net: force a finalize"):]
    forced = tail[:tail.index("msgs.append")]
    assert "_abort_plan" not in forced, "the forced finalize is still gated behind an abort"


def test_a_failing_review_does_not_lose_the_plan(patched_agent):
    """The draft is already valid by the time the critic runs. Losing the whole turn because
    the review or the fix timed out is the worst possible trade — the request errors out and
    the traveller gets nothing at all."""
    from unittest.mock import patch
    from agent import agent as mod

    draft = {"days": [{"day": 1, "title": "D", "items": [
        {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
         "duration_min": 120, "cost_eur": 22}]}], "total_cost_eur": 22}
    profile = {"destination": "Rome", "days": 1, "group": "2 friends", "budget": "mid-range"}

    def boom(*a, **k):
        raise RuntimeError("review timed out")

    with patch.object(mod, "_plan", lambda *a, **k: dict(draft)), \
         patch.object(mod, "_reflect", boom), \
         patch.object(mod, "_safe_geocode", lambda x: None), \
         patch.object(mod, "_forecast_for", lambda *a: None), \
         patch.object(mod, "_prices_for", lambda *a: None), \
         patch.object(mod, "_apply_travel_times", lambda p, pr, c, r=None: (p, 0)), \
         patch.object(mod, "_format", lambda *a, **k: "ITINERARY"), \
         patch.object(mod, "_profile", lambda *a, **k: {
             "profile": profile, "missing": [], "confirmed": True,
             "question": "", "intent": "replace"}):
        out = mod._run_turn("User: x\nAgent: Does this look right?\nUser: yes", [])

    assert "ITINERARY" in out["response"], "the plan was lost when the review failed"


def test_the_research_loop_leaves_room_for_the_rest_of_the_pipeline():
    """Eight lookups spent the whole budget on research, so the finalize, review and fix all
    ran against a wall that was nearly gone — and the request errored out."""
    from agent.agent import MAX_PLANNER_STEPS
    assert MAX_PLANNER_STEPS <= 6


def test_a_fix_does_not_restart_the_research(patched_agent):
    """The fix cycle used to call _plan with no draft, so the planner rebuilt the itinerary
    from scratch — running a whole second ReAct loop and re-looking-up places it had already
    checked. That doubled the run and pushed it past the time limit."""
    from unittest.mock import patch
    from agent import agent as mod
    calls, tools = [], []

    def fake_json(msgs, **kw):
        calls.extend(m["content"] for m in msgs)
        if len(calls) >= 3:
            return {"thought": "done", "done": True, "draft_plan": {
                "days": [{"day": 1, "title": "D", "items": [
                    {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
                     "duration_min": 90, "cost_eur": 22}]}], "total_cost_eur": 22}}
        return {"thought": "look", "tool": "maps_tool", "tool_input": {"query": "x"}}

    draft = {"days": [{"day": 1, "title": "D", "items": [
        {"time": "23:30", "name": "Dinner", "venue": "Trastevere",
         "duration_min": 60, "cost_eur": 50}]}], "total_cost_eur": 50}

    with patch.object(mod, "_chat_json", fake_json), \
         patch.object(mod, "run_tool", lambda t, i: (tools.append(t), {"ok": True})[1]):
        mod._plan({"destination": "Rome", "days": 1, "group": "2 friends",
                   "budget": "mid-range"}, [], run_id="t", draft=draft,
                  feedback={"issues": ["Day 1 ends at 00:30"], "fixes": []})

    from agent.agent import MAX_FIX_STEPS, MAX_PLANNER_STEPS
    assert MAX_FIX_STEPS < MAX_PLANNER_STEPS, "a fix must not get a full research budget"
    assert len(tools) <= MAX_FIX_STEPS, f"a fix ran {len(tools)} lookups"
    assert any("Trastevere" in c for c in calls), "the fix was not given the plan to correct"


def test_running_out_of_call_budget_does_not_lose_the_itinerary(patched_agent):
    """Observed: "CallLimitExceeded: per-run LLM call ceiling reached (12/12)" raised inside the
    formatter and the whole request failed, discarding a finished plan. Exhausting the call
    budget is an expected end to a long run — the deterministic renderer needs no call."""
    from unittest.mock import patch
    from agent import agent as mod
    from agent.usage import CallLimitExceeded

    plan = {"days": [{"day": 1, "title": "D", "items": [
        {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
         "duration_min": 90, "cost_eur": 22}]}], "total_cost_eur": 22}

    def boom(*a, **k):
        raise CallLimitExceeded("per-run LLM call ceiling reached (12/12)")

    with patch.object(mod, "chat", boom):
        out = mod._format({"destination": "Rome", "days": 1}, plan, [])
    assert "Colosseum" in out, "the finished plan was lost when the call budget ran out"


def test_the_call_ceiling_fits_the_pipeline_with_headroom():
    """MAX_CALLS_PER_RUN is the runaway guard on a fixed $9 allowance, so it must not be
    raised casually — but it also must not sit exactly on the worst case, or the fix cycle
    can never finish and the formatter crashes on the call that would have written the plan.
    """
    from agent.agent import MAX_PLANNER_STEPS, MAX_FIX_STEPS, MAX_REFLECT_CYCLES
    from agent.usage import MAX_CALLS_PER_RUN

    worst_case = (1                      # Conversational Intake
                  + MAX_PLANNER_STEPS + 1  # research turns + the forced finalize
                  + MAX_REFLECT_CYCLES     # critic pass(es)
                  + MAX_FIX_STEPS + 1      # a fix: lookups + the rewrite
                  + 1)                     # Output Formatter
    assert worst_case <= MAX_CALLS_PER_RUN, (
        f"a full turn needs {worst_case} calls but the ceiling is {MAX_CALLS_PER_RUN}")


def test_no_formatter_failure_can_lose_a_finished_plan(patched_agent):
    """The plan is complete by the time the formatter runs, so nothing that happens there is
    worth losing it over. Timeouts and the call budget were handled; an upstream 500 or a
    malformed response still killed the turn and the traveller saw an error page."""
    from unittest.mock import patch
    from agent import agent as mod
    from agent.usage import CallLimitExceeded

    plan = {"days": [{"day": 1, "title": "D", "items": [
        {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
         "duration_min": 90, "cost_eur": 18}]}], "total_cost_eur": 18}

    for failure in (TimeoutError("slow"), CallLimitExceeded("12/12"),
                    RuntimeError("upstream 500"), ValueError("malformed response")):
        def boom(*a, **k):
            raise failure

        with patch.object(mod, "chat", boom):
            out = mod._format({"destination": "Rome", "days": 1}, plan, [])
        assert "Colosseum" in out, f"the plan was lost when the formatter hit {failure!r}"


def test_run_agent_never_raises(patched_agent):
    """The API layer turns any exception into "The agent failed to complete this request",
    which is the least useful thing a traveller can be shown. Every stage already degrades on
    its own, so this is the last line of defence: an unforeseen failure still returns a
    reply, and a stored itinerary is shown again rather than lost."""
    from unittest.mock import patch
    from agent import agent as mod

    plan = {"days": [{"day": 1, "title": "D", "items": [
        {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
         "duration_min": 90, "cost_eur": 18}]}], "total_cost_eur": 18}
    stored = {"profile": {"destination": "Rome", "days": 1}, "plan": plan}

    for failure in (RuntimeError("upstream 500"), KeyError("boom"),
                    ValueError("bad json"), TimeoutError("slow")):
        def boom(*a, **k):
            raise failure

        with patch.object(mod, "_run_turn", boom):
            with_plan = mod.run_agent("User: x", state=stored)
            without = mod.run_agent("User: x", state=None)

        assert "Colosseum" in with_plan["response"], "a stored itinerary was thrown away"
        assert with_plan["branch"] == "error"          # honest about what happened
        assert "went wrong" in without["response"]     # and says so plainly
        assert without["steps"] == []


def test_a_provider_outage_does_not_discard_the_research(patched_agent):
    """Observed: a 502 from the provider on the finalize call threw away five completed
    steps. Only timeouts degraded; every other failure propagated and lost the turn."""
    from unittest.mock import patch
    from agent import agent as mod

    lookups = [{"thought": "a", "tool": "maps_tool", "tool_input": {"query": "Colosseum"}},
               {"thought": "b", "tool": "maps_tool", "tool_input": {"query": "Pantheon"}}]
    index = {"n": 0}

    def fake_json(msgs, **kw):
        if index["n"] < len(lookups):
            step = lookups[index["n"]]
            index["n"] += 1
            return step
        raise RuntimeError("upstream error")          # a 502 from the provider

    def fake_tool(tool, tool_input):
        return {"ok": True, "results": [{"name": tool_input["query"],
                                         "lat": 41.9, "lon": 12.5, "open_hours": None}]}

    with patch.object(mod, "_chat_json", fake_json), \
         patch.object(mod, "run_tool", fake_tool):
        plan = mod._plan({"destination": "Rome", "days": 2, "group": "2 friends",
                          "budget": "mid-range"}, [], run_id="t")

    names = [i["name"] for d in plan["days"] for i in d["items"]]
    assert "Colosseum" in names and "Pantheon" in names
    assert not any("self-guided" in n for n in names)


def test_configuration_and_budget_errors_still_fail_loudly(patched_agent):
    """Neither is transient: a missing key needs an operator, and the spend guard exists to
    stop the run. Degrading either into a vague itinerary would hide a real problem."""
    from unittest.mock import patch
    import pytest
    from agent import agent as mod
    from agent.llm import ConfigError
    from agent.usage import CallLimitExceeded

    for failure in (ConfigError("no API key"), CallLimitExceeded("15/15")):
        def boom(*a, **k):
            raise failure

        with patch.object(mod, "_chat_json", boom):
            with pytest.raises(type(failure)):
                mod._plan({"destination": "Rome", "days": 2, "group": "2",
                           "budget": "low"}, [], run_id="t")


# ---- Cost plausibility reaches the critic, and only as a question -------------------
def test_a_free_meal_is_put_to_the_critic_as_a_question(patched_agent, scripted_chat):
    """Live prices are injected but nothing proved the planner used them. A EUR0 dinner
    now reaches the review — as something to judge, not as a verified defect, because a
    plan can legitimately contain one."""
    free_dinner = json.dumps({
        "thought": "done", "done": True,
        "draft_plan": {"days": [{"day": 1, "title": "Day 1", "items": [
            {"time": "17:30", "name": "Temple", "duration_min": 90, "cost_eur": 8,
             "note": "quiet at dusk"},
            {"time": "19:00", "name": "Dinner in Pontocho", "duration_min": 90,
             "cost_eur": 0, "note": "Small izakayas line the alley."}]}],
            "total_cost_eur": 8},
    })
    seen = {}

    def _chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        sysmsg = messages[0]["content"] if messages else ""
        if "Reflection Layer" in sysmsg:
            seen["prompt"] = messages[1]["content"]
        return scripted_chat(planner_reply=free_dinner)(
            messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)

    patched_agent.install(_chat)
    out = patched_agent.module.run_agent("User: 1 day in Kyoto\nUser: yes")

    # Nothing was computed about this plan, so the only finding is the cost one — and it
    # arrives under the heading the critic is told to judge rather than to repeat.
    assert "Already VERIFIED defects" not in seen["prompt"]
    questions = seen["prompt"].split("Possible problems")[1]
    assert "Dinner in Pontocho" in questions and "EUR0" in questions
    # The critic passed it, so the traveller is not warned about an unconfirmed suspicion.
    assert "caveats" not in out["response"].lower()


def test_the_retry_budget_shrinks_with_the_wall():
    """Regression: a real run was killed by the platform at 300s.

    The SDK's timeout is per ATTEMPT, so max_retries=2 let a call run for three times what
    the wall had budgeted. A formatter call starting 26s before the wall was given 26s,
    retried twice and ran for 78 — past the platform cap, which kills the function and
    costs the whole reply instead of degrading it.
    """
    import time as _t
    from agent import llm

    try:
        llm.set_wall(_t.monotonic() + 26)          # close to the wall
        timeout, retries = llm._call_budget()
        assert retries == 0, "no room for a retry this late"
        assert timeout * (1 + retries) <= 26

        llm.set_wall(_t.monotonic() + 250)         # start of a run
        timeout, retries = llm._call_budget()
        assert timeout == llm._TIMEOUT_S, "an early call still gets the full ceiling"
        assert timeout * (1 + retries) <= 250, "attempts must fit inside the wall"
    finally:
        llm.set_wall(None)


def test_the_wall_leaves_room_for_the_platform_cap():
    """230/250 against a 300s maxDuration: ~50s for the cold start, both Supabase round
    trips and serialising the reply. 240/265 was tried and a real run was killed."""
    from agent.agent import MAX_RUN_SECONDS, HARD_WALL_SECONDS
    assert MAX_RUN_SECONDS < HARD_WALL_SECONDS
    assert HARD_WALL_SECONDS <= 250, "a real run past this was killed by the platform"



def test_a_clean_second_review_is_not_reported_as_a_failed_one(patched_agent):
    """Regression from a real recorded run. The first review found two defects, the fix pass
    corrected both, and the second review returned nothing — which is what a clean plan looks
    like. The code read "said nothing" as "could not answer" and delivered a correct
    itinerary under a banner saying it had not been validated.
    """
    from unittest.mock import patch
    from agent import agent as mod

    prof = {"destination": "Rome", "days": 1, "group": "2 friends", "budget": "mid-range"}
    draft = {"days": [{"day": 1, "title": "D1", "items": [
        {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
         "duration_min": 120, "cost_eur": 20}]}], "total_cost_eur": 20}

    # The critic replies, and has nothing to report.
    with patch.object(mod, "_chat_json", lambda *a, **k: {"verdict": "PASS",
                                                          "must_fix": [], "be_aware": []}):
        v = mod._reflect(prof, draft, [], run_id="t")
    assert v["answered"] is True

    # The critic does not reply at all (a timeout degrades raw to None).
    with patch.object(mod, "_chat_json", lambda *a, **k: None):
        v = mod._reflect(prof, draft, [], run_id="t")
    assert v["answered"] is False
    assert v["verdict"] == "FAIL", "an unreadable critic must never green-light a plan"


def test_a_bare_follow_up_uses_the_stored_profile(patched_agent):
    """The README promises conversation_id retrieves the stored profile AND plan, but the
    clarify gate ran against the profile intake re-extracted from the transcript alone. So
    "what does day 2 cost?" sent with its conversation_id was answered with "where would you
    like to go?" while the server was holding that traveller's itinerary.
    """
    from unittest.mock import patch
    from agent import agent as mod

    profile = {"destination": "Rome", "days": 3, "group": "2 friends",
               "budget": "mid-range", "walking": "high"}
    plan = {"days": [{"day": 2, "title": "D2", "items": [
        {"time": "09:00", "name": "Colosseum", "venue": "Colosseum",
         "duration_min": 120, "cost_eur": 20}]}], "total_cost_eur": 20}

    # Intake sees only the question, so it extracts nothing and reports everything missing.
    empty = {"profile": {}, "missing": ["destination", "days", "budget", "group"],
             "confirmed": False, "question": "Where would you like to go?", "intent": "question"}
    with patch.object(mod, "_profile", lambda *a, **k: dict(empty)), \
         patch.object(mod, "_answer_question", lambda *a, **k: "Day 2 costs EUR20 per person."):
        out = mod._run_turn("what does day 2 cost per person?", [],
                            state={"profile": profile, "plan": plan})

    assert out["branch"] == "question", f"routed to {out['branch']} instead of answering"
    assert "EUR20" in out["response"]
