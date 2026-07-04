"""GET /api/agent_info — agent description, purpose, prompt template, and a worked example."""
import json
from http.server import BaseHTTPRequestHandler

# TODO: replace with real agent example and response
_EXAMPLE_STEPS = [
    {
        "module": "Preference Profiler",
        "prompt": {},
        "response": {},
    },
    {
        "module": "ReAct Planner",
        "prompt": {},
        "response": {},
    },
    {
        "module": "ReAct Planner",
        "prompt": {},
        "response": {},
    },
    {
        "module": "Reflection Layer",
        "prompt": {},
        "response": {},
    },
    {
        "module": "Output Formatter",
        "prompt": {},
        "response": {},
    },
]

_EXAMPLE_RESPONSE = ()

INFO = {
    "description":
        "An autonomous Trip Planning agent. From a single natural-language request it profiles "
        "the traveller, runs a ReAct loop over travel tools (live weather plus maps/search/reviews, "
        "and fictive flights/booking), self-critiques the draft, and returns a costed day-by-day "
        "itinerary.",
    "purpose":
        "Replace hours of fragmented trip research with one autonomous pass that produces a "
        "personalized, budget-aware, geographically sane itinerary.",
    "prompt_template": {
        "template":
            "Plan a {days}-day trip to {destination} for a {group} who likes {style}. "
            "Budget: {budget}. Must-see: {priorities}. Avoid: {avoid}. "
            "Accessibility needs: {accessibility}.",
    },
    "prompt_examples": [{
        "prompt": "TODO: Replace with real prompt example ",
        "full_response": _EXAMPLE_RESPONSE,
        "steps": _EXAMPLE_STEPS,
    }],
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._respond(200, INFO)

    def do_OPTIONS(self):
        self._respond(204, None)

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode())
