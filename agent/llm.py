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
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MODEL = os.environ.get("LLMOD_MODEL", "NBUECSE-gpt-5-mini")

# Per-call timeout and retry budget. Reasoning models think before emitting any tokens,
# so a planner call can legitimately run far longer than a chat-model call. Retries are
# disabled: a timeout here means the model is slow, and retrying only doubles the wait
# (worst case per call is now _TIMEOUT_S, keeping MAX_RUN_SECONDS under Vercel's limit).
_TIMEOUT_S = 60
_MAX_RETRIES = 0

# Reasoning models spend part of the completion budget on hidden reasoning tokens, so the
# visible answer needs headroom on top of the caller's request or `content` comes back empty.
_REASONING_HEADROOM = 2000

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
    base_url = (os.environ.get("LLMOD_BASE_URL") or "https://api.llmod.ai").strip()
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
    """True when the error is a 400 rejecting `response_format` — the only case where
    retrying without JSON mode is the right response. A 400 that names a different
    parameter (temperature, token limits) must not disable JSON mode."""
    msg = str(err).lower()
    if "response_format" in msg or "json_object" in msg:
        return True
    if any(k in msg for k in ("temperature", "max_tokens", "max_completion_tokens")):
        return False
    return getattr(err, "status_code", None) == 400 or err.__class__.__name__ == "BadRequestError"


def chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
    """Run one chat completion and return the content string ("" if none).

    When `json_mode` is requested we ask for a strict JSON object. Only if the endpoint
    rejects that parameter we fall back once and remember it, so the same request is
    not billed twice. Transient errors (auth, rate limit, timeout) propagate to the caller.
    """
    global _json_mode_supported
    client = _get_client()
    base = _completion_params(messages, temperature, max_tokens)

    if json_mode and _json_mode_supported is not False:
        try:
            c = client.chat.completions.create(**base, response_format={"type": "json_object"})
            _json_mode_supported = True
            return c.choices[0].message.content or ""
        except Exception as e:
            if not _looks_like_unsupported_json_mode(e):
                raise
            _json_mode_supported = False

    c = client.chat.completions.create(**base)
    return c.choices[0].message.content or ""


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
