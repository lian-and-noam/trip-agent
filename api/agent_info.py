"""GET /api/agent_info — agent description, purpose, prompt template, and worked examples."""
import json
import os
from http.server import BaseHTTPRequestHandler

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_PATH = os.path.join(_HERE, "example_run.json")


def _load_examples():
    """Worked examples, recorded from real runs by scripts/record_example.py.

    Held in a data file rather than inline so regenerating them after a change to the
    prompts or the pipeline is a re-run, not an edit — which is how the previous placeholder
    was able to sit here going stale. A missing or unreadable file degrades to an empty list
    so the endpoint still answers with valid, complete metadata.
    """
    try:
        with open(_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


INFO = {
    "description":
        "An autonomous trip-planning agent. One Conversational Intake call reads the whole "
        "dialogue and routes the turn: it asks for anything missing, confirms a typed traveller "
        "profile, then plans an itinerary with a ReAct loop over travel tools (live weather, plus "
        "places and opening hours via Geoapify/OpenStreetMap, web search and reviews via Tavily; "
        "flight-search and booking are fictive by design and buy nothing), self-critiques the "
        "draft, and returns a costed day-by-day plan with real map links. Once a plan exists the "
        "agent can revise a specific day, answer a question about it, or carry on and clear "
        "defects the critic left open — none of which re-plans the trip.",
    "purpose":
        "Replace hours of fragmented trip research with one autonomous pass that produces a "
        "personalized, budget-aware, geographically sane itinerary — and then behaves like a "
        "planner you can talk to, editing the plan you already have instead of starting over.",
    "prompt_template": {
        "template":
            "Plan a {days}-day trip to {destination} for a {group} who likes {style}. "
            "Budget: {budget}. Travelling {start_time} to {end_time}, starting at "
            "{start_point} and ending at {end_point}. Staying at {lodging}. "
            "Each day should run roughly {day_start} to {day_end}. "
            "Must-see: {priorities}. Avoid: {avoid}. Also: {details}. "
            "Accessibility needs: {accessibility}.",
    },
    "prompt_examples": _load_examples(),
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Re-read per request so a regenerated example file is picked up without a redeploy.
        # The file is small and this endpoint is not hot.
        self._respond(200, dict(INFO, prompt_examples=_load_examples()))

    def do_OPTIONS(self):
        self._respond(204, None)

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode())
