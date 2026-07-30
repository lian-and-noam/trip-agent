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


def _confirmed():
    return {"profile": {"days": 2, "destination": "Kyoto", "budget": "mid-range",
                        "group": "couple", "style": "temples"},
            "confirmed": True, "question": ""}
