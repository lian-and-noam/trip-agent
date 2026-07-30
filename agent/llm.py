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
# so a planner call can legitimately run far longer than a chat-model call. Retries are
# disabled: a timeout here means the model is slow, and retrying only doubles the wait.
#
# This value is the overshoot past agent.MAX_RUN_SECONDS: a call started one tick before
# the deadline still runs this long. Keep MAX_RUN_SECONDS + _TIMEOUT_S below vercel.json's
# maxDuration (180 + 110 = 290 < 300) so the handler always gets to write its response.
#
# 110, not 40: measured on a 7-day plan, the planner finalize takes 51-67s and the Output
# Formatter 97s once it is actually emitting markdown instead of spending its whole budget
# on reasoning. At 40 both were cut off mid-call; at 75 the formatter still was. The budget
# split deliberately favours this per-call ceiling over MAX_RUN_SECONDS, because the
# formatter runs last with ~140s of completed work behind it — losing it discards the turn.
_TIMEOUT_S = int(os.environ.get("LLM_TIMEOUT_S", "110"))
_MAX_RETRIES = 0

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
    global _client
    if _client is None:
        api_key, base_url = _require_config()
        _client = OpenAI(api_key=api_key, base_url=base_url,
                         timeout=_TIMEOUT_S, max_retries=_MAX_RETRIES)
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


def _meter(u, billed):
    """Fold reported usage into the active run meter, if a run is being metered."""
    m = usage.current()
    if m is not None:
        m.record(u.get("prompt_tokens"), u.get("completion_tokens"), billed=billed)


def chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
    """Run one chat completion and return the content string ("" if none).

    Every call is metered before it is made: `before_call()` raises BudgetExceeded rather
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
            _meter(recorded.get("usage") or {}, billed=False)
            obs.log("llm_replay", key=ck)
            return recorded.get("content") or ""

    client = _get_client()
    base = _completion_params(messages, temperature, max_tokens)

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
    _meter(u, billed=True)
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
