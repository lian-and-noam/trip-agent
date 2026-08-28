"""POST /api/execute — main entry point.

Body: {"prompt": "..."}                                  -> {"status", "error", "response", "steps"}
      {"prompt": "...", "conversation_id": "..."}        (optional, enables the revision path)

The response envelope is fixed by the project brief and never gains fields. Conversation
state travels the other way: the client owns an anonymous conversation id, the server looks
up the stored profile/plan for it, and the agent edits that plan instead of re-reading a
rendered itinerary out of the transcript. A request with only `prompt` behaves exactly as
it always did, so the documented contract keeps working untouched.

HTTP is always 200 for both ok and error because the caller parses the JSON body. Internal
error detail is logged, never returned to the client.
"""
import os
import sys
import json
import uuid
from http.server import BaseHTTPRequestHandler

# Make the local `agent` package importable when Vercel runs this file (project root on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import run_agent, AgentError   # noqa: E402
from agent.llm import ConfigError               # noqa: E402
from agent.usage import CallLimitExceeded          # noqa: E402
from agent import obs, store                    # noqa: E402

# `prompt` carries the conversation transcript. With the itinerary now held server-side it
# no longer grows with every revision, but the cap stays as a backstop.
MAX_BODY_BYTES = 64 * 1024   # reject oversized bodies before reading them into memory
MAX_PROMPT_CHARS = 16000     # roughly 4k tokens of conversation history


def _as_uuid(value):
    """Return a canonical UUID string, or None.

    Strict on purpose: these ids are interpolated into PostgREST filters, so anything that
    is not a well-formed UUID is dropped rather than passed through.
    """
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def _title_for(profile):
    """Short human label for the conversation list, e.g. '7 days in Kyoto'."""
    p = profile if isinstance(profile, dict) else {}
    dest, days = p.get("destination"), p.get("days")
    if dest and days:
        return f"{days} days in {dest}"[:120]
    return (dest or "Trip")[:120]


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        conversation_id = None
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

            # The conversation id IS the capability: a v4 UUID minted by the client and
            # known only to it. There is no second identifier to check it against, and
            # adding one would be an authentication guard, which the brief forbids.
            conversation_id = _as_uuid(payload.get("conversation_id"))

            # Prior state is best-effort: if Supabase is unconfigured or slow, the agent
            # simply plans fresh instead of revising. It never blocks the turn.
            prior = None
            if conversation_id:
                row = store.get_conversation(conversation_id)
                if row:
                    prior = {"profile": row.get("profile"), "plan": row.get("plan")}

            out = run_agent(prompt, state=prior)

            state = out.get("state") or {}
            if conversation_id:
                store.save_conversation(conversation_id,
                                        profile=state.get("profile"), plan=state.get("plan"),
                                        title=_title_for(state.get("profile")))
            self._envelope("ok", None, response=out["response"], steps=out["steps"])

        except CallLimitExceeded as e:
            # A ceiling was hit rather than something breaking. Say so plainly: the user
            # can retry, and the operator needs to know the budget guard fired.
            obs.log("execute_budget_stop", detail=str(e))
            self._envelope("error", "This request hit the agent's cost safety limit and was "
                                    "stopped before completing. Try a shorter trip.")
        except ConfigError as e:
            obs.log("execute_config_error", detail=str(e))
            self._envelope("error", "Server is not configured correctly (missing LLM credentials).")
        except AgentError as e:
            # The turn failed part-way: log the underlying cause and return the steps taken
            # so far, so the trace panel shows where it stopped instead of going blank.
            cause = e.__cause__ or e
            obs.log("execute_error", error=type(cause).__name__, detail=str(cause),
                    steps=len(e.steps))
            self._envelope("error", "The agent failed to complete this request. Please try again.",
                           steps=e.steps)
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
