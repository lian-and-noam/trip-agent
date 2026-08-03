"""Revision path: Branch D (question) and Branch E (revise), plus the patch merge.

These encode the two properties the revision path exists for: an edit must be surgical
(days nobody asked about come through untouched) and it must be cheap (a follow-up costs
a fraction of a full re-plan).
"""
import json

from agent import agent as A


def _counting(scripted):
    """Wrap a scripted chat so a test can assert how many LLM calls a branch actually made."""
    calls = []

    def _chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        calls.append((messages[0]["content"] if messages else "")[:40])
        return scripted(messages, temperature=temperature, json_mode=json_mode,
                        max_tokens=max_tokens)
    return _chat, calls


# ---- _apply_patch: the deterministic merge ----------------------------------------
def test_patch_replaces_only_the_named_day(two_day_plan):
    before = two_day_plan()
    patch = {"days": [{"day": 2, "title": "Lighter", "items": [
        {"time": "10:00", "name": "Slow morning", "duration_min": 120, "cost_eur": 3}]}]}
    after, changed = A._apply_patch(before, patch)

    assert changed == [2]
    assert after["days"][0] == before["days"][0]          # day 1 byte-identical
    assert after["days"][1]["title"] == "Lighter"
    assert len(after["days"][1]["items"]) == 1
    assert after["total_cost_eur"] == 3                    # recomputed, not trusted


def test_patch_without_a_day_number_is_ignored(two_day_plan):
    """A patch day with no number cannot be placed, and must not clobber day 1."""
    before = two_day_plan()
    after, changed = A._apply_patch(before, {"days": [{"items": [
        {"name": "Mystery", "cost_eur": 99}]}]})
    assert changed == []
    assert after["days"] == before["days"]


def test_patch_with_no_usable_items_leaves_the_day_alone(two_day_plan):
    before = two_day_plan()
    after, changed = A._apply_patch(before, {"days": [{"day": 1, "items": []}]})
    assert changed == []
    assert after["days"][0] == before["days"][0]


def test_patch_survives_junk(two_day_plan):
    before = two_day_plan()
    for junk in (None, "nope", [1, 2], {"days": "not-a-list"}, {}):
        after, changed = A._apply_patch(before, junk)
        assert changed == [] and len(after["days"]) == 2


# ---- Branch E: revise --------------------------------------------------------------
def test_revise_edits_the_stored_plan_without_replanning(patched_agent, scripted_chat, prior_state):
    chat, calls = _counting(scripted_chat())
    patched_agent.install(chat)
    out = patched_agent.module.run_agent(
        "User: 2 days in Kyoto\nAgent: [itinerary]\nUser: make day 2 lighter", state=prior_state)

    assert out["branch"] == "revise"
    mods = [s["module"] for s in out["steps"]]
    assert "Plan Editor" in mods
    assert "ReAct Planner" not in mods            # the expensive loop never ran
    # Day 1 was never mentioned, so it survives the edit untouched.
    assert out["state"]["plan"]["days"][0] == prior_state["plan"]["days"][0]
    assert out["state"]["plan"]["days"][1]["title"] == "Day 2 (lighter)"


def test_revise_is_cheaper_than_planning(patched_agent, scripted_chat, prior_state):
    """The whole point of Branch E: a follow-up must not cost a full re-plan."""
    chat, revise_calls = _counting(scripted_chat())
    patched_agent.install(chat)
    patched_agent.module.run_agent("User: make day 2 lighter", state=prior_state)

    chat2, plan_calls = _counting(scripted_chat())
    patched_agent.install(chat2)
    patched_agent.module.run_agent("User: 2 days in Kyoto\nUser: yes")   # no prior state

    assert len(revise_calls) < len(plan_calls)
    assert len(revise_calls) == 3        # intake + editor + formatter


def test_revise_reports_when_the_edit_could_not_be_applied(patched_agent, scripted_chat,
                                                           prior_state):
    empty = json.dumps({"thought": "no", "days": [], "summary": "that day does not exist"})
    patched_agent.install(scripted_chat(editor_reply=empty))
    out = patched_agent.module.run_agent("User: change day 9", state=prior_state)

    assert "caveats" in out["response"].lower()
    assert out["state"]["plan"]["days"] == prior_state["plan"]["days"]   # unchanged


def test_malformed_editor_output_never_crashes(patched_agent, scripted_chat, prior_state):
    patched_agent.install(scripted_chat(editor_reply="this is not json at all"))
    out = patched_agent.module.run_agent("User: make day 2 lighter", state=prior_state)
    assert out["branch"] == "revise"
    assert out["state"]["plan"]["days"] == prior_state["plan"]["days"]


# ---- Branch D: question ------------------------------------------------------------
def test_question_answers_without_touching_the_plan(patched_agent, scripted_chat, prior_state):
    intake = dict(_confirmed(), intent="question")
    chat, calls = _counting(scripted_chat(intake=intake, qa_reply="Day 2 costs €32."))
    patched_agent.install(chat)
    out = patched_agent.module.run_agent("User: what does day 2 cost?", state=prior_state)

    assert out["branch"] == "question"
    assert "€32" in out["response"]
    assert [s["module"] for s in out["steps"]] == ["Conversational Intake", "Itinerary Q&A"]
    assert len(calls) == 2                                        # cheapest branch with a plan
    assert out["state"]["plan"] == prior_state["plan"]            # nothing mutated


# ---- Profile change routes to a full re-plan, not an edit --------------------------
def test_changing_a_required_field_forces_reconfirmation(patched_agent, scripted_chat,
                                                         prior_state):
    """Product rule: a new destination is a new trip. It must not be applied as an edit."""
    intake = dict(_confirmed(), intent="revise")          # model says "revise"...
    intake["profile"] = dict(intake["profile"], destination="Lisbon")   # ...but the trip changed
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent("User: actually make it Lisbon", state=prior_state)

    assert out["branch"] == "confirm"                     # deterministic override won
    assert "re-plan" in out["response"].lower() or "different trip" in out["response"].lower()
    mods = [s["module"] for s in out["steps"]]
    assert "Plan Editor" not in mods and "ReAct Planner" not in mods
    # The existing itinerary is retained until the replacement is confirmed.
    assert out["state"]["plan"]["days"] == prior_state["plan"]["days"]


def test_optional_field_change_is_still_an_edit(patched_agent, scripted_chat, prior_state):
    """Changing something optional (dietary) is a revision, not a new trip."""
    intake = dict(_confirmed(), intent="revise")
    intake["profile"] = dict(intake["profile"], dietary=["vegetarian"])
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent("User: we're vegetarian now", state=prior_state)
    assert out["branch"] == "revise"


# ---- Branch F: "continue" carries on rather than starting over ---------------------
def _with_verdict(scripted, verdict):
    """A scripted chat whose Reflection Layer returns a chosen verdict."""
    base = scripted()

    def _chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        if "Reflection Layer" in (messages[0]["content"] if messages else ""):
            return json.dumps(verdict)
        return base(messages, temperature=temperature, json_mode=json_mode,
                    max_tokens=max_tokens)
    return _chat


def _state_with_open_issues(prior_state, issues):
    plan = dict(prior_state["plan"], open_issues=list(issues))
    return dict(prior_state, plan=plan)


def test_continue_is_never_read_as_a_different_trip(patched_agent, scripted_chat, prior_state):
    """The reported bug: a bare "continue" sent the traveller back to a confirmation
    prompt for the itinerary they already had."""
    intake = dict(_confirmed(), intent="replace")
    intake["profile"] = dict(intake["profile"], destination="Kyoto, Japan")  # intake drift
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent(
        "User: 2 days in Kyoto\nAgent: [itinerary]\nUser: continue", state=prior_state)

    assert out["branch"] != "confirm"
    assert "Does this look right?" not in out["response"]


def test_continue_with_outstanding_defects_re_enters_the_fix_cycle(
        patched_agent, scripted_chat, prior_state):
    """A plan delivered with unresolved must_fix items must be worked on, not described."""
    state = _state_with_open_issues(prior_state, ["Day 1 has no dinner."])
    patched_agent.install(_with_verdict(scripted_chat, {"verdict": "PASS", "must_fix": [],
                                                        "be_aware": [], "fixes": []}))
    out = patched_agent.module.run_agent("User: continue", state=state)

    assert out["branch"] == "resume"
    mods = [s["module"] for s in out["steps"]]
    assert "ReAct Planner" in mods and "Reflection Layer" in mods
    assert "Itinerary Q&A" not in mods                     # it acted instead of describing
    # The critic passed the corrected plan, so nothing is left outstanding.
    assert "open_issues" not in out["state"]["plan"]


def test_validate_acts_on_outstanding_defects(patched_agent, scripted_chat, prior_state):
    """"validate" reaches the fix cycle even when intake called it a question."""
    state = _state_with_open_issues(prior_state, ["Day 1 has no dinner."])
    intake = dict(_confirmed(), intent="question")
    chat = _with_verdict(lambda: scripted_chat(intake=intake),
                         {"verdict": "PASS", "must_fix": [], "be_aware": [], "fixes": []})
    patched_agent.install(chat)
    out = patched_agent.module.run_agent("User: validate", state=state)

    assert out["branch"] == "resume"
    assert "Itinerary Q&A" not in [s["module"] for s in out["steps"]]


def test_defects_the_fix_cycle_cannot_clear_stay_on_record(patched_agent, scripted_chat,
                                                           prior_state):
    """Asking again must make progress, so what is still wrong is carried forward."""
    state = _state_with_open_issues(prior_state, ["Day 1 has no dinner."])
    patched_agent.install(_with_verdict(scripted_chat,
                                        {"verdict": "FAIL", "must_fix": ["Day 1 has no dinner."],
                                         "be_aware": [], "fixes": []}))
    out = patched_agent.module.run_agent("User: carry on", state=state)

    assert out["state"]["plan"]["open_issues"] == ["Day 1 has no dinner."]
    assert "caveats" in out["response"].lower()


def test_continue_carrying_an_instruction_is_left_to_the_model(patched_agent, scripted_chat,
                                                               prior_state):
    """Only bare carry-on phrasings are routed deterministically. Anything with an
    instruction in it still goes through intake, including its replace override."""
    intake = dict(_confirmed(), intent="revise")
    intake["profile"] = dict(intake["profile"], destination="Lisbon")
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent(
        "User: continue, but move it to Lisbon", state=prior_state)

    assert out["branch"] == "confirm"


def test_unresolved_defects_are_recorded_on_a_newly_planned_itinerary(patched_agent,
                                                                      scripted_chat):
    """The plan branch must leave behind the record that makes "continue" work."""
    patched_agent.install(_with_verdict(scripted_chat,
                                        {"verdict": "FAIL", "must_fix": ["Day 1 has no dinner."],
                                         "be_aware": [], "fixes": []}))
    out = patched_agent.module.run_agent("User: 2 days in Kyoto\nUser: yes")

    assert out["branch"] == "plan"
    assert out["state"]["plan"]["open_issues"] == ["Day 1 has no dinner."]


def test_an_edit_elsewhere_does_not_erase_the_record(two_day_plan):
    """A day-scoped patch resolves nothing on the other days."""
    before = dict(two_day_plan(), open_issues=["Day 1 has no dinner."])
    after, _ = A._apply_patch(before, {"days": [{"day": 2, "title": "Lighter", "items": [
        {"time": "10:00", "name": "Slow morning", "duration_min": 120, "cost_eur": 3}]}]})
    assert after["open_issues"] == ["Day 1 has no dinner."]


def _confirmed():
    return {"profile": {"days": 2, "destination": "Kyoto", "budget": "mid-range",
                        "group": "couple", "style": "temples"},
            "confirmed": True, "question": ""}


# ---- A question can never replace the plan it is asking about ----------------------
def test_a_question_is_never_treated_as_a_new_trip(patched_agent, scripted_chat, prior_state):
    """The reported bug: asking "what is the most busy day?" came back as a confirmation
    prompt. Intake re-reads the whole transcript each turn, so it can rephrase a required
    field on a turn that asked for no change at all — and answering a question mutates
    nothing, so there is no silent replacement for the override to prevent."""
    intake = dict(_confirmed(), intent="question")
    intake["profile"] = dict(intake["profile"], group="two friends")   # was "couple"
    patched_agent.install(scripted_chat(intake=intake, qa_reply="Day 2 is the busiest."))
    out = patched_agent.module.run_agent(
        "User: 2 days in Kyoto\nAgent: [itinerary]\nUser: what is the most busy day?",
        state=prior_state)

    assert out["branch"] == "question"
    assert "different trip" not in out["response"]
    assert "Does this look right?" not in out["response"]
    assert out["state"]["plan"] == prior_state["plan"]        # nothing mutated
    # The drift is discarded too: the answer and the stored state both come from the
    # profile the traveller confirmed, not from this turn's re-extraction of it.
    assert out["state"]["profile"]["group"] == prior_state["profile"]["group"]


def test_a_real_change_still_forces_reconfirmation(patched_agent, scripted_chat, prior_state):
    """The exemption is scoped to questions; an edit that changes the trip is unaffected."""
    intake = dict(_confirmed(), intent="revise")
    intake["profile"] = dict(intake["profile"], destination="Lisbon")
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent("User: make it Lisbon instead", state=prior_state)
    assert out["branch"] == "confirm"


def test_the_trace_names_the_field_that_forced_a_replacement(patched_agent, scripted_chat,
                                                             prior_state):
    """Without this the trace shows only that a turn was re-confirmed, not why, and the
    stored profile has to be read out of the database to find out."""
    intake = dict(_confirmed(), intent="revise")
    intake["profile"] = dict(intake["profile"], destination="Lisbon")
    patched_agent.install(scripted_chat(intake=intake))
    out = patched_agent.module.run_agent("User: make it Lisbon instead", state=prior_state)

    confirm = [s for s in out["steps"] if s["response"].get("stage") == "confirm"][0]
    assert confirm["response"]["changed_fields"] == ["destination"]
    assert confirm["response"]["replacing"] is True


def test_required_changed_names_every_differing_field():
    before = {"destination": "Kyoto", "days": 2, "budget": "mid-range", "group": "couple"}
    assert A._required_changed(before, before) == []
    assert A._required_changed(before, dict(before, destination=" kyoto ")) == []  # rephrasing
    assert A._required_changed(before, dict(before, days=3, group="4 friends")) == \
        ["days", "group"]
