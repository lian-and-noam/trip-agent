"""Shared test setup: put the project root on sys.path and provide LLM fakes so no
test ever makes a real network/LLM call."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Every module name the orchestrator is allowed to log (must match the architecture diagram).
# Each of these is a real LLM call; the deterministic validation layer is not a step.
DIAGRAM_MODULES = {"Conversational Intake", "ReAct Planner", "Plan Editor",
                   "Reflection Layer", "Output Formatter", "Itinerary Q&A"}

# A complete, already-confirmed intake result — makes the default fake go straight to Branch C.
_CONFIRMED_INTAKE = {
    "profile": {"days": 2, "destination": "Kyoto", "budget": "mid-range",
                "group": "couple", "style": "temples", "priorities": ["temples"]},
    "confirmed": True, "question": "",
}


def _default_scripted_chat(planner_reply=None, formatter_reply="# Itinerary\n- 09:00 Visit (€5)",
                           intake=None, editor_reply=None, qa_reply="Day 3 costs €40."):
    """Return a fake `chat` that answers by which module's system prompt it sees.

    `intake` overrides the Conversational Intake JSON (to drive branches A-E);
    `planner_reply` / `editor_reply` override a module's raw string (to inject malformed
    output); `qa_reply` is what the Itinerary Q&A module returns.
    """
    intake_obj = intake if intake is not None else _CONFIRMED_INTAKE
    done_plan = json.dumps({
        "thought": "done", "done": True,
        "draft_plan": {"days": [{"day": 1, "title": "Day 1", "items": [
            {"time": "09:00", "name": "Visit", "duration_min": 90, "cost_eur": 5, "note": "tip"}]}],
            "total_cost_eur": 5},
    })
    # A patch that rewrites day 2 only — days the patch omits must survive untouched.
    default_patch = json.dumps({
        "thought": "lighten day 2", "summary": "Removed one stop from day 2.",
        "days": [{"day": 2, "title": "Day 2 (lighter)", "items": [
            {"time": "10:00", "name": "Slow morning", "duration_min": 120,
             "cost_eur": 3, "note": "relaxed"}]}],
    })

    def _chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        sysmsg = messages[0]["content"] if messages else ""
        if "Conversational Intake" in sysmsg:
            return json.dumps(intake_obj)
        if "ReAct Planner" in sysmsg:
            return planner_reply if planner_reply is not None else done_plan
        if "Plan Editor" in sysmsg:
            return editor_reply if editor_reply is not None else default_patch
        if "answer questions about a trip itinerary" in sysmsg:
            return qa_reply
        if "Reflection Layer" in sysmsg:
            return json.dumps({"verdict": "PASS", "issues": [], "fixes": []})
        if "Output Formatter" in sysmsg:
            return formatter_reply
        return "{}"
    return _chat


def _two_day_plan():
    """A stored plan to revise, with two distinct days so a patch can be shown to be surgical."""
    return {"days": [
        {"day": 1, "title": "Day 1", "items": [
            {"time": "09:00", "name": "Fushimi Inari", "duration_min": 120,
             "cost_eur": 0, "note": "go early"}]},
        {"day": 2, "title": "Day 2", "items": [
            {"time": "09:00", "name": "Museum", "duration_min": 90, "cost_eur": 12, "note": "x"},
            {"time": "14:00", "name": "Market", "duration_min": 60, "cost_eur": 20, "note": "y"}]},
    ], "total_cost_eur": 32}


@pytest.fixture
def two_day_plan():
    return _two_day_plan


@pytest.fixture
def prior_state():
    """The {profile, plan} a previous turn would have persisted to Supabase."""
    return {
        "profile": {"days": 2, "destination": "Kyoto", "budget": "mid-range",
                    "group": "couple", "style": "temples"},
        "plan": _two_day_plan(),
    }


@pytest.fixture
def scripted_chat():
    return _default_scripted_chat


@pytest.fixture
def diagram_modules():
    return DIAGRAM_MODULES


@pytest.fixture
def patched_agent(monkeypatch):
    """Patch the orchestrator's LLM + geocode boundaries. Returns the `agent` module.
    Call `patched_agent.install(chat_fn)` inside a test to swap the fake chat."""
    from agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "geocode_place",
                        lambda name: {"lat": 35.0, "lon": 135.0, "name": name, "country": "JP"})

    class _Harness:
        module = agent_mod

        def install(self, chat_fn):
            monkeypatch.setattr(agent_mod, "chat", chat_fn)

    return _Harness()
