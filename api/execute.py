"""POST /api/execute — main entry point.

Body: {"prompt": "..."}  ->  {"status", "error", "response", "steps"}

The response always uses the same envelope (HTTP 200 for both ok and error) because the
caller parses the JSON body. Internal error detail is logged, never returned to the client.
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler

# Make the local `agent` package importable when Vercel runs this file (project root on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import run_agent      # noqa: E402
from agent.llm import ConfigError      # noqa: E402
from agent import obs                  # noqa: E402

# `prompt` carries the entire conversation transcript (stateless multi-turn intake),
# so the caps are larger than a single message but still bounded.
MAX_BODY_BYTES = 64 * 1024   # reject oversized bodies before reading them into memory
MAX_PROMPT_CHARS = 16000     # roughly 4k tokens of conversation history


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = self._content_length()
            if length > MAX_BODY_BYTES:
                return self._envelope("error", "Request body too large.")

            try:
                payload = json.loads(self.rfile.read(length) if length > 0 else b"{}")
            except Exception:
                return self._envelope("error", "Request body must be valid JSON.")
            if not isinstance(payload, dict):
                return self._envelope("error", "Request body must be a JSON object.")

            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self._envelope("error", "Missing or invalid 'prompt'.")
            prompt = prompt.strip()
            if len(prompt) > MAX_PROMPT_CHARS:
                return self._envelope("error", f"'prompt' too long (max {MAX_PROMPT_CHARS} characters).")

            out = run_agent(prompt)
            self._envelope("ok", None, response=out["response"], steps=out["steps"])

        except ConfigError as e:
            obs.log("execute_config_error", detail=str(e))
            self._envelope("error", "Server is not configured correctly (missing LLM credentials).")
        except Exception as e:
            # Log the real detail server-side; return a generic, safe message to the client.
            obs.log("execute_error", error=type(e).__name__, detail=str(e))
            self._envelope("error", "The agent failed to complete this request. Please try again.")

    def do_OPTIONS(self):
        self._respond(204, None)

    def _content_length(self):
        try:
            return max(0, int(self.headers.get("Content-Length", 0)))
        except (TypeError, ValueError):
            return 0

    def _envelope(self, status, error, response=None, steps=None):
        """Emit the required contract: {status, error, response, steps}."""
        self._respond(200, {"status": status, "error": error,
                            "response": response, "steps": steps or []})

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode())
