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
    assert "yes" in out["response"].lower() and "profile" in out["response"].lower()
    assert [s["module"] for s in out["steps"]] == ["Conversational Intake"]   # still no planning


def test_branch_C_confirmed_runs_planner(patched_agent, scripted_chat):
    patched_agent.install(scripted_chat())   # default intake is complete + confirmed
    out = patched_agent.module.run_agent(
        "User: 2 days in Kyoto, couple, mid-range, temples\nAgent: ...confirm?\nUser: yes")
    mods = [s["module"] for s in out["steps"]]
    assert "Preference Profiler" in mods and "ReAct Planner" in mods and "Output Formatter" in mods


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
    assert set(out.keys()) == {"response", "steps"}
    assert isinstance(out["response"], str) and out["response"]
    assert isinstance(out["steps"], list) and out["steps"]


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
    assert set(out.keys()) == {"response", "steps"}
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


def test_map_links_added_to_plan():
    from agent.agent import _with_map_links
    plan = {"days": [{"day": 1, "items": [{"name": "Kinkaku-ji"}]}]}
    out = _with_map_links(plan, "Kyoto")
    url = out["days"][0]["items"][0]["map_url"]
    assert url.startswith("https://www.google.com/maps/search/") and "Kinkaku-ji" in url
