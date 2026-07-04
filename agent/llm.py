"""LLMod.ai client built on the OpenAI SDK.
- Configuration is validated once, up front, so a missing key surfaces as a clear
  error instead of a KeyError deep inside a request.
- `chat()` uses an explicit timeout and remembers whether the endpoint supports JSON
  mode, so a single logical call is never billed twice.
- `parse_json()` never raises: it returns a parsed value or None, using a balanced-brace
  scan rather than a greedy regex.

LLMod fronts litellm. Reasoning models (gpt-5*, o-series) accept only temperature=1 and
expect `max_completion_tokens` rather than `max_tokens`, so request parameters are built
per-model instead of assumed.
"""
import os
import time
import re
import json
from openai import OpenAI, APITimeoutError

from . import cassette, obs, usage

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MODEL = os.environ.get("LLMOD_MODEL", "NBUECSE-gpt-5-mini")

# Per-call timeout and retry budget. Reasoning models think before emitting any tokens,
# so a planner call can legitimately run far longer than a chat-model call.
#
# A ceiling, not a budget: set_wall() already caps each call at the time actually left, so
# this only decides when a slow call is given up on while budget remains. 110s because the
# planner finalize measures 51-67s and the formatter 97s once it is emitting markdown rather
# than reasoning; smaller values cut both off mid-call.
_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "110"))
# The SDK retries 5xx and connection errors, which fail fast and usually succeed on the
# next attempt — a 502 from the provider should not cost a run that has already done its
# research. Timeouts are NOT retried by this: a slow call retried is simply slow twice, and
# the per-call timeout below already bounds it.
_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))

# Absolute wall-clock limit for the whole request, set once per run by the agent. Without it
# each call needs a pessimistic fixed timeout and the run must reserve that much headroom
# against the platform's limit. With a wall, a call takes whatever time is actually left.
_WALL = [None]


def set_wall(monotonic_deadline):
    """Set the hard stop for this request. Calls are truncated to fit inside it."""
    _WALL[0] = monotonic_deadline


def wall_exceeded():
    """True once the hard stop for this request has passed.

    Callers use this to skip optional work rather than begin a call that can only be
    truncated. It matters because `_call_timeout` never returns less than 5 seconds: the
    wall bounds when a call may START, not how long one already started may run, so every
    call begun at or after it pushes the request that much further past its budget.
    """
    return _WALL[0] is not None and time.monotonic() >= _WALL[0]


def _call_budget():
    """(timeout, max_retries) that together cannot outlast the wall.

    The SDK's `timeout` is per ATTEMPT, not per call, so `max_retries=2` lets a call run
    for three times what the wall budgeted for it. Truncating only the timeout therefore
    does not bound the call: a formatter call starting 26s before the wall was given 26s,
    retried twice on a provider 5xx and ran for 78 — past the platform's own limit, which
    kills the function and costs the whole reply rather than degrading it.

    Retries are still worth having, so they are budgeted rather than removed: allow only
    as many attempts as fit in the time actually left. Early in a run that is the full
    ceiling with a retry to spare; near the wall it is one attempt and no retry.
    """
    if _WALL[0] is None:
        return _TIMEOUT_S, _MAX_RETRIES
    left = max(5, int(_WALL[0] - time.monotonic()))
    timeout = min(_TIMEOUT_S, left)
    attempts = left // max(1, timeout)
    return timeout, max(0, min(_MAX_RETRIES, attempts - 1))


def _call_timeout():
    """The per-attempt timeout alone, without the retry budget. See `_call_budget`."""
    return _call_budget()[0]

# Reasoning models spend part of the completion budget on hidden reasoning tokens, so the
# visible answer needs headroom on top of the caller's request or `content` comes back empty.
#
# 4000, not 2000: measured on a 7-day plan, the Output Formatter consumed its entire
# 3400-token cap on reasoning and returned an EMPTY string — the user saw a caveats banner
# with no itinerary under it. The Reflection Layer came within 291 tokens of the same.
# Reasoning cost scales with input complexity, and 2000 was empirically too small for the
# heavier modules. This is a ceiling, not a target.
_REASONING_HEADROOM = 4000

_client = None
_json_mode_supported = None  # None until probed, then True/False


class ConfigError(RuntimeError):
    """Raised when required LLM configuration is missing or blank. The API layer
    catches this and returns a clean error envelope instead of a stack trace."""


def _require_config():
    """Validate required env vars once. Returns (api_key, base_url)."""
    api_key = (os.environ.get("LLMOD_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError("LLMOD_API_KEY is not set")
    # The SDK appends /chat/completions, so the base URL must carry the /v1 segment.
    # Without it the default pointed at https://api.llmod.ai/chat/completions.
    base_url = (os.environ.get("LLMOD_BASE_URL") or "https://api.llmod.ai/v1").strip()
    return api_key, base_url


def _get_client():
    """The shared client. Its timeout is deliberately NOT set here.

    The client is cached and outlives a single request on a warm container, so a timeout
    fixed at construction is whichever value the first request happened to compute — and it
    never shrinks as a later run approaches its deadline. The result is a wall clock that
    nothing enforces. The timeout is passed per call instead; see chat().
    """
    global _client
    if _client is None:
        api_key, base_url = _require_config()
        _client = OpenAI(api_key=api_key, base_url=base_url, max_retries=_MAX_RETRIES)
    return _client


def _is_reasoning_model(name):
    """True for models that reject `temperature` and use `max_completion_tokens`."""
    n = (name or "").lower()
    return "gpt-5" in n or bool(re.search(r"(^|[-/])o[134]\b", n))


def _completion_params(messages, temperature, max_tokens):
    """Build create() kwargs appropriate to the configured model."""
    params = {"model": MODEL, "messages": messages}
    if _is_reasoning_model(MODEL):
        params["max_completion_tokens"] = max_tokens + _REASONING_HEADROOM
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens
    return params


def _looks_like_unsupported_json_mode(err):
    """True only when the endpoint explicitly rejected `response_format`.

    Nothing else qualifies. The previous version treated *any* 400 as an unsupported JSON
    mode, so a context-length error ("maximum context length is 8192 tokens") re-issued the
    request without JSON mode — billing it twice and failing again — then latched
    `_json_mode_supported` off for the lifetime of the warm instance, silently degrading
    every later request it served. Everything else now propagates to the caller.
    """
    msg = str(err).lower()
    return "response_format" in msg or "json_object" in msg


def is_timeout(err):
    """True when a call failed because it ran out of time, not because it was rejected.

    Lives here so the orchestrator never has to import SDK exception types. Also matches the
    underlying httpx/socket timeouts the SDK wraps, by class name, since those can surface
    directly when a connection is interrupted rather than cleanly wrapped.
    """
    if isinstance(err, APITimeoutError):
        return True
    return "timeout" in type(err).__name__.lower()


def _unpack(completion):
    """Pull (content, usage) out of a provider response without trusting its shape."""
    try:
        content = completion.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        content = ""
    u = getattr(completion, "usage", None)
    return content, {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "completion_tokens": getattr(u, "completion_tokens", 0) or 0}


def _meter(u):
    """Fold reported usage into the active run meter, if a run is being metered."""
    m = usage.current()
    if m is not None:
        m.record(u.get("prompt_tokens"), u.get("completion_tokens"))


def chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
    """Run one chat completion and return the content string ("" if none).

    Every call is metered before it is made: `before_call()` raises CallLimitExceeded rather
    than spending past the per-run call ceiling or the cumulative budget. When a cassette
    is active the response may be replayed from disk, which costs nothing but still
    exercises the same guards so development matches production control flow.

    When `json_mode` is requested we ask for a strict JSON object. Only if the endpoint
    rejects that parameter do we fall back once and remember it, so the same request is
    not billed twice. Transient errors (auth, rate limit, timeout) propagate to the caller.
    """
    global _json_mode_supported

    m = usage.current()
    if m is not None:
        m.before_call()

    ck = (cassette.key_for(MODEL, messages, temperature, max_tokens, json_mode)
          if cassette.enabled() else None)
    if ck:
        call_provider, recorded = cassette.should_call_provider(ck)
        if not call_provider:
            _meter(recorded.get("usage") or {})
            obs.log("llm_replay", key=ck)
            return recorded.get("content") or ""

    client = _get_client()
    base = _completion_params(messages, temperature, max_tokens)
    # Evaluated per call, so both shrink as the run nears its deadline and a late call —
    # including everything the SDK retries on its behalf — cannot run past the wall.
    timeout, retries = _call_budget()
    base["timeout"] = timeout
    client = client.with_options(max_retries=retries)

    completion = None
    if json_mode and _json_mode_supported is not False:
        try:
            completion = client.chat.completions.create(
                **base, response_format={"type": "json_object"})
            _json_mode_supported = True
        except Exception as e:
            if not _looks_like_unsupported_json_mode(e):
                raise
            _json_mode_supported = False

    if completion is None:
        completion = client.chat.completions.create(**base)

    content, u = _unpack(completion)
    _meter(u)
    if ck:
        cassette.save(ck, content, u, request_preview=str(messages[0].get("content", ""))[:120])
    return content


def _first_json_object(text):
    """Return the first balanced {...} substring, respecting string literals and escapes."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json(text):
    """Best-effort JSON extraction from an LLM reply. Returns the parsed value (usually
    a dict) or None. Tolerant of ```json fences and surrounding prose."""
    cleaned = re.sub(r"```json|```", "", str(text)).strip()
    for candidate in (cleaned, _first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None
