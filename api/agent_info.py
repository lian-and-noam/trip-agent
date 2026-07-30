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
        "structured maps/search/reviews mocks and fictive flights/booking), self-critiques the "
        "draft, and returns a costed day-by-day plan with real map links. Once a plan exists the "
        "agent can revise a specific day or answer a question about it without re-planning the "
        "trip.",
    "purpose":
        "Replace hours of fragmented trip research with one autonomous pass that produces a "
        "personalized, budget-aware, geographically sane itinerary — and then behaves like a "
        "planner you can talk to, editing the plan you already have instead of starting over.",
    "prompt_template": {
        "template":
            "Plan a {days}-day trip to {destination} for a {group} who likes {style}. "
            "Budget: {budget}. Must-see: {priorities}. Avoid: {avoid}. "
            "Accessibility needs: {accessibility}.",
        "required_fields": ["days", "destination", "group", "style", "budget"],
        "optional_fields": ["when", "origin", "dietary", "walking", "accessibility",
                            "priorities", "avoid"],
        "notes":
            "The agent is conversational and the backend is stateless. Send the running "
            "transcript in `prompt` as 'User:' / 'Agent:' lines. Anything missing is asked for "
            "in a single message; the agent then shows the typed profile and waits for "
            "confirmation before spending tokens on planning. After a plan exists, "
            "'make day 2 lighter' edits it and 'what does day 3 cost?' answers from it. "
            "Changing a required field is treated as a new trip and re-confirmed.",
        "optional_request_fields":
            "`conversation_id` and `device_id` (UUIDs) may be sent alongside `prompt` to enable "
            "the revision path. A request carrying only `prompt` behaves identically.",
    },
    "modules": ["Conversational Intake", "ReAct Planner", "Plan Editor",
                "Reflection Layer", "Output Formatter", "Itinerary Q&A"],
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
