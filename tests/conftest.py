"""Shared test setup: put the project root on sys.path and provide LLM fakes so no
test ever makes a real network/LLM call."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Every module name the orchestrator is allowed to log (must match the architecture diagram).
DIAGRAM_MODULES = {"Conversational Intake", "Preference Profiler",
                   "ReAct Planner", "Reflection Layer", "Output Formatter"}

# A complete, already-confirmed intake result — makes the default fake go straight to Branch C.
_CONFIRMED_INTAKE = {
    "profile": {"days": 2, "destination": "Kyoto", "budget": "mid-range",
                "group": "couple", "style": "temples", "priorities": ["temples"]},
    "confirmed": True, "question": "",
}


def _default_scripted_chat(planner_reply=None, formatter_reply="# Itinerary\n- 09:00 Visit (€5)", intake=None):
    """Return a fake `chat` that answers by which module's system prompt it sees.
    `intake` overrides the Conversational Intake JSON (to drive Branch A/B/C);
    `planner_reply` overrides the planner's raw string (to inject malformed output)."""
    intake_obj = intake if intake is not None else _CONFIRMED_INTAKE
    done_plan = json.dumps({
        "thought": "done", "done": True,
        "draft_plan": {"days": [{"day": 1, "title": "Day 1", "items": [
            {"time": "09:00", "name": "Visit", "duration_min": 90, "cost_eur": 5, "note": "tip"}]}],
            "total_cost_eur": 5},
    })

    def _chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        sysmsg = messages[0]["content"] if messages else ""
        if "Conversational Intake" in sysmsg:
            return json.dumps(intake_obj)
        if "ReAct Planner" in sysmsg:
            return planner_reply if planner_reply is not None else done_plan
        if "Reflection Layer" in sysmsg:
            return json.dumps({"verdict": "PASS", "issues": [], "fixes": []})
        if "Output Formatter" in sysmsg:
            return formatter_reply
        return "{}"
    return _chat


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
