"""Validation and coercion for every LLM output in the pipeline.

Intentionally stdlib-only (no pydantic) to keep the serverless function small and
its cold starts fast. Every function here is total: it never raises on bad input, it
coerces or clamps toward a safe, schema-valid value. This is what lets the orchestrator
turn a malformed model reply into a recoverable state instead of a 500.
"""

# --- Deterministic bounds (independent of the LLM critic) ---------------------------
DAYS_MIN, DAYS_MAX = 1, 30
BUDGET_LEVELS = ("low", "mid-range", "luxury")
WALKING_LEVELS = ("light", "moderate", "high", "unlimited")
# Rough daily distance each level implies (km); kept for reference and prompt wording.
WALKING_KM = {"light": 3, "moderate": 9, "high": 18, "unlimited": 30}
# Rough per-person, per-day spend ceilings by tier (EUR), used for a deterministic
# feasibility check that does not depend on the LLM critic.
_PER_DAY_CEILING = {"low": 120, "mid-range": 260, "luxury": 650}


def is_obj(x):
    """True only for a JSON object (dict). Guards every `.get()` in the pipeline."""
    return isinstance(x, dict)


def as_obj(x):
    """Coerce to a dict. A non-dict reply (list, str, number, None) becomes {}."""
    return x if isinstance(x, dict) else {}


def _as_str(x, default=""):
    if isinstance(x, str):
        return x
    if x is None:
        return default
    return str(x)


def _as_int(x, default):
    try:
        if isinstance(x, bool):  # bool is an int subclass; reject it explicitly
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _as_list(x):
    if isinstance(x, list):
        return x
    if x is None or x == "":
        return []
    return [x]


def _clamp(n, lo, hi):
    return max(lo, min(hi, n))


def _as_walking(x, default="moderate"):
    """Coerce a walking-tolerance value to a category in WALKING_LEVELS.
    Accepts a category name, a common synonym, or a number of km per day."""
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        n = float(x)
        return "light" if n <= 5 else "moderate" if n <= 13 else "high" if n <= 25 else "unlimited"
    s = str(x or "").strip().lower()
    if s in WALKING_LEVELS:
        return s
    if any(w in s for w in ("unlimited", "no limit", "as much", "any amount")):
        return "unlimited"
    if any(w in s for w in ("high", "a lot", "lots", "far", "long")):
        return "high"
    if any(w in s for w in ("light", "little", "minimal", "low", "short", "small")):
        return "light"
    return default


# --- Profile ------------------------------------------------------------------------
def validate_profile(obj):
    """Coerce the Preference Profiler output into a typed, bounded profile dict.

    Always returns a complete dict with every expected key, so no downstream module can
    raise a KeyError. `days` is clamped to a sane range and `budget` is snapped to a
    known tier; any such correction is appended to `assumptions` so it is visible in the
    trace and to the user.
    """
    o = as_obj(obj)
    assumptions = [_as_str(a) for a in _as_list(o.get("assumptions"))]

    raw_days = _as_int(o.get("days"), 3)
    days = _clamp(raw_days, DAYS_MIN, DAYS_MAX)
    if days != raw_days:
        assumptions.append(f"Requested {raw_days} days; clamped to {days} for a feasible plan.")

    budget = _as_str(o.get("budget"), "mid-range").strip().lower()
    if budget not in BUDGET_LEVELS:
        assumptions.append(f'Unrecognized budget "{o.get("budget")}"; defaulted to "mid-range".')
        budget = "mid-range"

    return {
        "days": days,
        "destination": _as_str(o.get("destination")).strip(),
        "style": _as_str(o.get("style")),
        "group": _as_str(o.get("group")),
        "budget": budget,
        # Optional fields — enrich the plan when provided, harmless when empty.
        "when": _as_str(o.get("when")).strip(),           # travel dates or season
        "origin": _as_str(o.get("origin")).strip(),       # departure city (used for flights)
        "dietary": [_as_str(d) for d in _as_list(o.get("dietary"))],
        "walking": _as_walking(o.get("walking")),         # light | moderate | high | unlimited
        "accessibility": bool(o.get("accessibility")),
        "priorities": [_as_str(p) for p in _as_list(o.get("priorities"))],
        "avoid": [_as_str(a) for a in _as_list(o.get("avoid"))],
        "assumptions": assumptions,
    }


def compact_profile(profile):
    """Return the profile with empty/default optional fields dropped.

    The required fields are always kept; optional fields are included only when the user
    actually supplied them. Sending this smaller object to the planner, critic and
    formatter reduces prompt size without changing the plan.
    """
    o = as_obj(profile)
    out = {k: o[k] for k in ("days", "destination", "style", "group", "budget") if k in o}
    for k in ("when", "origin"):
        if o.get(k):
            out[k] = o[k]
    for k in ("dietary", "priorities", "avoid"):
        if o.get(k):
            out[k] = o[k]
    if o.get("accessibility"):
        out["accessibility"] = True
    if o.get("walking") and o.get("walking") != "moderate":  # skip the default to save tokens
        out["walking"] = o["walking"]
    return out


def budget_ceiling_eur(profile):
    """Deterministic total-cost ceiling for the trip: per-day tier ceiling x days."""
    per_day = _PER_DAY_CEILING.get(profile.get("budget"), 260)
    return per_day * max(1, _as_int(profile.get("days"), 1))


# Fields the traveller must supply before planning begins (drives Conversational Intake).
REQUIRED_FIELDS = ("destination", "days", "budget", "group", "style")


def missing_required(profile):
    """Return the required fields the user has not supplied yet. `days` must be a positive
    int; the rest must be non-empty strings. Deterministic (no LLM), so the
    clarify/confirm/plan decision is cheap and predictable."""
    o = as_obj(profile)
    missing = []
    for f in REQUIRED_FIELDS:
        v = o.get(f)
        if f == "days":
            if _as_int(v, 0) < 1:
                missing.append(f)
        elif not isinstance(v, str) or not v.strip():
            missing.append(f)
    return missing


# --- Planner turn -------------------------------------------------------------------
def classify_turn(obj):
    """Classify one ReAct turn: ("done", draft_plan) | ("tool", name, input) |
    ("invalid", None). Never raises; a non-dict turn is "invalid"."""
    if not is_obj(obj):
        return ("invalid", None)
    if obj.get("done") and is_obj(obj.get("draft_plan")):
        return ("done", obj["draft_plan"])
    tool = obj.get("tool")
    if isinstance(tool, str) and tool:
        ti = obj.get("tool_input")
        return ("tool", tool, ti if is_obj(ti) else {})
    return ("invalid", None)


# --- Draft plan ---------------------------------------------------------------------
def validate_draft_plan(obj):
    """Coerce a draft plan into a clean, self-consistent structure, or return None.

    `total_cost_eur` is recomputed from the items so the number is always internally
    consistent (the model's own total is not trusted). Returns None only when there is
    not a single usable day/item to salvage; the caller then falls back to `minimal_plan`.
    """
    o = as_obj(obj)
    days_out = []
    for i, day in enumerate(_as_list(o.get("days"))):
        d = as_obj(day)
        items_out = []
        for it in _as_list(d.get("items")):
            io = as_obj(it)
            items_out.append({
                "time": _as_str(io.get("time")),
                "name": _as_str(io.get("name")),
                "duration_min": _clamp(_as_int(io.get("duration_min"), 60), 0, 24 * 60),
                "cost_eur": max(0, _as_int(io.get("cost_eur"), 0)),
                "note": _as_str(io.get("note")),
            })
        days_out.append({
            "day": _as_int(d.get("day"), i + 1),
            "title": _as_str(d.get("title")),
            "items": items_out,
        })

    if not any(day["items"] for day in days_out):
        return None
    total = sum(it["cost_eur"] for day in days_out for it in day["items"])
    return {"days": days_out, "total_cost_eur": total}


def minimal_plan(profile):
    """A valid, honest fallback plan so finalize is never None.

    Marked `degraded` so the Output Formatter and UI can tell the user this is a skeleton
    the planner could not fully build out."""
    days = _clamp(_as_int(profile.get("days"), 1), DAYS_MIN, DAYS_MAX)
    dest = profile.get("destination") or "your destination"
    return {
        "degraded": True,
        "days": [
            {"day": n + 1, "title": f"Day {n + 1} in {dest}",
             "items": [{"time": "09:00", "name": f"Explore {dest} (self-guided)",
                        "duration_min": 240, "cost_eur": 0,
                        "note": "Planner could not complete a detailed plan; this is a placeholder day."}]}
            for n in range(days)
        ],
        "total_cost_eur": 0,
    }


# --- Verdict ------------------------------------------------------------------------
def validate_verdict(obj):
    """Coerce the Reflection Layer output. An unknown or garbled verdict is treated as
    FAIL, so an unreadable critic can never green-light a bad plan."""
    o = as_obj(obj)
    verdict = _as_str(o.get("verdict")).strip().upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "issues": [_as_str(x) for x in _as_list(o.get("issues"))],
        "fixes": [_as_str(x) for x in _as_list(o.get("fixes"))],
    }
