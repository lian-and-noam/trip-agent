"""Supabase persistence: conversation state, traveller preferences, and run accounting.

Talks to PostgREST directly over `requests` rather than pulling in `supabase-py`. The
serverless bundle stays small, cold starts stay fast, and there is no new dependency to
pin — `requests` is already required by the tool layer.

Every function here is total. A missing config, a timeout, a 500, or malformed JSON all
degrade to None/False and are logged; the agent then runs exactly as it did before
Supabase existed. Persistence is an enhancement, never a dependency — the same discipline
`_safe_geocode` and `run_tool` already follow.

Three tables (see db/schema.sql):
  conversations    the current profile + plan for a conversation. This is what makes the
                   revision path possible without replaying the itinerary through the
                   prompt, and what survives a browser refresh.
  runs             one row per /api/execute turn with token counts and cost, which is how
                   the $9 project budget is actually tracked rather than estimated.
"""
import os

import requests

from . import obs

URL = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
# Service key preferred: writes happen server-side only and must never reach the browser.
KEY = (os.environ.get("SUPABASE_SERVICE_KEY")
       or os.environ.get("SUPABASE_ANON_KEY") or "").strip()

_TIMEOUT = float(os.environ.get("SUPABASE_TIMEOUT_S") or 2.5)


def enabled():
    """True when Supabase is configured. Everything below no-ops when this is False."""
    return bool(URL and KEY)


def _headers(extra=None):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}",
         "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def _request(method, path, **kw):
    """One guarded PostgREST call. Returns parsed JSON, or None on any failure."""
    if not enabled():
        return None
    try:
        r = requests.request(method, f"{URL}/rest/v1/{path}",
                             headers=_headers(kw.pop("extra_headers", None)),
                             timeout=_TIMEOUT, **kw)
        r.raise_for_status()
        if not r.content:
            return []
        return r.json()
    except requests.exceptions.Timeout:
        obs.log("store_error", op=path, error="timeout")
    except requests.exceptions.HTTPError as e:
        obs.log("store_error", op=path, error="http",
                status=getattr(e.response, "status_code", 0))
    except requests.exceptions.RequestException:
        obs.log("store_error", op=path, error="network")
    except ValueError:
        obs.log("store_error", op=path, error="bad_json")
    return None


# ---- conversations -----------------------------------------------------------------
def get_conversation(conversation_id):
    """Return {profile, plan, title, device_id} for a conversation, or None."""
    if not conversation_id:
        return None
    rows = _request("GET", "conversations",
                    params={"id": f"eq.{conversation_id}",
                            "select": "id,device_id,profile,plan,title", "limit": 1})
    return (rows or [None])[0] if isinstance(rows, list) else None


def save_conversation(conversation_id, device_id=None, profile=None, plan=None, title=None):
    """Upsert the current state of a conversation. Returns True on success."""
    if not conversation_id:
        return False
    # `updated_at` is deliberately absent: the database owns it via the touch trigger.
    # PostgREST would pass a client-supplied "now()" through as a string literal, and
    # Postgres rejects 'now()' as timestamptz input, failing every write silently.
    row = {"id": conversation_id}
    for k, v in (("device_id", device_id), ("profile", profile),
                 ("plan", plan), ("title", title)):
        if v is not None:
            row[k] = v
    out = _request("POST", "conversations", json=row,
                   extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
    return out is not None


# ---- traveller preferences ---------------------------------------------------------
def record_run(run_id, conversation_id=None, snapshot=None, ms=None, branch=None):
    """Persist one turn's token/cost accounting. This is the $9 budget ledger."""
    snap = snapshot or {}
    out = _request("POST", "runs", json={
        "run_id": run_id,
        "conversation_id": conversation_id,
        "branch": branch,
        "llm_calls": snap.get("calls", 0),
        "prompt_tokens": snap.get("prompt_tokens", 0),
        "completion_tokens": snap.get("completion_tokens", 0),
        "cost_usd": snap.get("cost_usd", 0),
        "ms": ms,
    }, extra_headers={"Prefer": "return=minimal"})
    return out is not None


def spend_to_date():
    """Total USD spent across every recorded run, or None when unavailable.

    Backed by a SQL function so the sum happens in Postgres instead of pulling every row
    into the function. This is the number to check against the $9 ceiling.
    """
    out = _request("POST", "rpc/total_spend_usd", json={})
    if isinstance(out, (int, float)):
        return float(out)
    if isinstance(out, list) and out and isinstance(out[0], (int, float)):
        return float(out[0])
    return None
