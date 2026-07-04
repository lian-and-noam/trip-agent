"""Record/replay for LLM calls, so iterating on the agent does not cost money.

A single confirmed turn makes up to ~9 model calls. Re-running one scenario twenty times
while debugging the revision path would burn a meaningful slice of a $9 budget on
responses we have already seen.

With `LLM_CASSETTE_MODE=auto` the first run records each distinct request to disk and
every later run replays it for free. The cache key covers everything that can change a
response (model, messages, temperature, token cap, JSON mode), so a prompt edit correctly
misses the cache and re-records only what actually changed.

Modes:
  off     - disabled (the default, and what production runs)
  replay  - never call the provider; a miss is an error, so CI cannot silently spend
  record  - always call the provider and overwrite the stored response
  auto    - replay when present, otherwise call and record

This is a development tool. It is inert unless the env var is set.
"""
import hashlib
import json
import os

MODE = (os.environ.get("LLM_CASSETTE_MODE") or "off").strip().lower()
DIR = os.environ.get("LLM_CASSETTE_DIR") or ".cassettes"


class CassetteMiss(RuntimeError):
    """Replay was demanded but nothing was recorded for this request."""


def enabled():
    return MODE in ("replay", "record", "auto")


def key_for(model, messages, temperature, max_tokens, json_mode):
    """Stable hash of everything that can change the response."""
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "json_mode": bool(json_mode)},
        sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _path(key):
    return os.path.join(DIR, key + ".json")


def load(key):
    """Return a recorded {content, usage} dict, or None when nothing is stored."""
    try:
        with open(_path(key), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save(key, content, usage, request_preview=None):
    """Store one response. Best-effort: a read-only filesystem must not break a run."""
    try:
        os.makedirs(DIR, exist_ok=True)
        with open(_path(key), "w", encoding="utf-8") as f:
            json.dump({"content": content, "usage": usage,
                       "request": request_preview}, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def should_call_provider(key):
    """Decide whether this request has to hit the network.

    Returns (call_provider: bool, recorded: dict|None). Raises CassetteMiss in strict
    replay mode so a missing recording fails loudly instead of quietly spending.
    """
    if MODE == "off":
        return True, None
    if MODE == "record":
        return True, None
    recorded = load(key)
    if recorded is not None:
        return False, recorded
    if MODE == "replay":
        raise CassetteMiss(
            f"no recording for request {key} and LLM_CASSETTE_MODE=replay. "
            f"Re-run once with LLM_CASSETTE_MODE=auto to record it.")
    return True, None
