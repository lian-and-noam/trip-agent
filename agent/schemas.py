"""Validation and coercion for every LLM output in the pipeline.

Intentionally stdlib-only (no pydantic) to keep the serverless function small and
its cold starts fast. Every function here is total: it never raises on bad input, it
coerces or clamps toward a safe, schema-valid value. This is what lets the orchestrator
turn a malformed model reply into a recoverable state instead of a 500.
"""
import re

# --- Deterministic bounds (independent of the LLM critic) ---------------------------
DAYS_MIN, DAYS_MAX = 1, 30
BUDGET_LEVELS = ("low", "mid-range", "luxury")
WALKING_LEVELS = ("light", "moderate", "high", "unlimited")
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


# Models write numbers the way people do — "€25", "40 EUR", "$1,200.50", "about 90 min".
# Grab the first number-shaped run of characters, then normalise the separators.
_NUMBER_RE = re.compile(r"-?\d[\d ,_]*(?:\.\d+)?")
_DECIMAL_COMMA_RE = re.compile(r"-?\d+,\d{2}")


def _to_number(x):
    """Best-effort numeric extraction. Returns a float, or None when there is none."""
    if isinstance(x, bool):          # bool is an int subclass; reject it explicitly
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return None
    m = _NUMBER_RE.search(x)
    if not m:
        return None
    tok = m.group(0).replace(" ", "").replace("_", "")
    if "." in tok:
        tok = tok.replace(",", "")                 # 1,200.50 -> comma is a thousands mark
    elif _DECIMAL_COMMA_RE.fullmatch(tok):
        tok = tok.replace(",", ".")                # 12,90    -> European decimal comma
    else:
        tok = tok.replace(",", "")                 # 1,200    -> thousands mark
    try:
        return float(tok)
    except ValueError:
        return None


def _as_int(x, default):
    """Coerce to int, rounding to nearest instead of truncating.

    Tolerant of the currency and unit formats a model actually emits. The old `int(x)`
    turned "€25" and "40 EUR" into `default`, silently zeroing item costs and letting an
    over-budget trip slip past the deterministic budget guard, and it truncated 12.90 to 12
    so every fractional cost lost its remainder.
    """
    n = _to_number(x)
    if n is None:
        return default
    return int(n + 0.5) if n >= 0 else int(n - 0.5)   # half away from zero


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
        # A concrete amount if the traveller named one ("2000 euro"). The tier stays as a
        # coarse style signal, but this is what the budget guard actually measures against —
        # "mid-range" is a guess at their wallet, 2000 EUR is a fact about it.
        "budget_amount_eur": max(0, _as_int(o.get("budget_amount_eur"), 0)),
        # Whether that amount covers the whole party or one person. Costs are quoted per
        # person throughout, so a party total has to be divided before it can be compared.
        "budget_basis": ("total" if _as_str(o.get("budget_basis")).strip().lower() == "total"
                         else "per person"),
        # Optional fields — enrich the plan when provided, harmless when empty.
        "when": _as_str(o.get("when")).strip(),           # travel dates or season
        # Where the trip physically begins and ends, and when. A traveller who says "from
        # 17/8 15:00 at Prague Bank Hotel to a 20/8 18:30 flight" is stating hard anchors:
        # day 1 cannot start at 09:00, and the last day has to end at the airport in time.
        # Without these the planner has no anchor for day one or the journey home.
        "start_point": _as_str(o.get("start_point")).strip(),   # hotel, station, address
        "end_point": _as_str(o.get("end_point")).strip(),       # airport, station, hotel
        "start_time": _as_str(o.get("start_time")).strip(),     # e.g. "17/8 15:00"
        "end_time": _as_str(o.get("end_time")).strip(),         # e.g. "20/8 18:30"
        "lodging": _as_str(o.get("lodging")).strip(),           # named hotel, if given
        # Anything else concrete the traveller said that no typed field covers: "we have a
        # rental car", "my sister joins on day 3", "no early mornings", "celebrating an
        # anniversary". The typed fields can never anticipate all of it, and a detail the
        # traveller bothered to state should reach the planner rather than being discarded
        # for not fitting the schema.
        "details": [_as_str(d).strip() for d in _as_list(o.get("details")) if _as_str(d).strip()],
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
    if o.get("budget_amount_eur"):
        out["budget_amount_eur"] = o["budget_amount_eur"]
        out["budget_basis"] = o.get("budget_basis", "per person")
    for k in ("when", "start_point", "end_point",
              "start_time", "end_time", "lodging"):
        if o.get(k):
            out[k] = o[k]
    for k in ("dietary", "priorities", "avoid", "details"):
        if o.get(k):
            out[k] = o[k]
    if o.get("accessibility"):
        out["accessibility"] = True
    if o.get("walking") and o.get("walking") != "moderate":  # skip the default to save tokens
        out["walking"] = o["walking"]
    return out


def budget_ceiling_eur(profile):
    """Deterministic per-person total-cost ceiling for the trip.

    A stated amount wins over the tier: if the traveller said "2000 euro" there is no reason
    to measure them against a generic mid-range allowance. A party total is divided by the
    group size first, because every cost in the plan is quoted per person.
    """
    stated = _as_int(profile.get("budget_amount_eur"), 0)
    if stated > 0:
        if profile.get("budget_basis") == "total":
            stated = stated // max(1, group_size(profile))
        return stated
    per_day = _PER_DAY_CEILING.get(profile.get("budget"), 260)
    return per_day * max(1, _as_int(profile.get("days"), 1))


_GROUP_NUMBERS = {"solo": 1, "single": 1, "couple": 2, "pair": 2, "two": 2, "three": 3,
                  "four": 4, "five": 5, "six": 6}


def group_size(profile):
    """Best-effort headcount from the free-text group field ("2 friends" -> 2).

    Defaults to 1 rather than guessing high: dividing a party total by too large a number
    would silently shrink the budget and make the plan poorer than the traveller asked for.
    """
    text = _as_str(profile.get("group")).lower()
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        return max(1, min(20, int(m.group(1))))
    for word, n in _GROUP_NUMBERS.items():
        if word in text:
            return n
    return 1


# Fields the traveller must supply before planning begins (drives Conversational Intake).
# `style` (interests) is deliberately NOT required: it shapes the itinerary but blocking on
# it cost an extra clarify turn on almost every conversation. It is still collected when
# offered, and the confirmation card shows it as unspecified so it is easy to add.
REQUIRED_FIELDS = ("destination", "days", "budget", "group")


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
# A "1 minute" museum visit is not a judgement call, it is a broken number — the model
# occasionally emits duration in hours, or drops a digit. Floors are per kind of item, and
# only ever raise: a genuinely quick stop is 15 minutes, not 1.
_MIN_DURATION_MIN = 15
_MIN_TRANSFER_MIN = 5
_TRANSFER_WORDS = ("transfer", "walk", "tram", "metro", "train", "bus", "taxi", "drive",
                   "ride", "transit", "depart", "arrival")


def _sane_duration(value, name=""):
    """Coerce duration to minutes, rejecting values too small to be real.

    Legs can legitimately be short, so they get a lower floor than visits. Everything is
    still capped at a day. Deterministic on purpose: asking the critic to notice "1 minute"
    worked only sometimes, and a plan with a 1-minute castle visit is visibly broken.
    """
    minutes = _clamp(_as_int(value, 60), 0, 24 * 60)
    low = (name or "").lower()
    floor = _MIN_TRANSFER_MIN if any(w in low for w in _TRANSFER_WORDS) else _MIN_DURATION_MIN
    return max(minutes, floor) if minutes < floor else minutes


# Control characters and lone surrogates reach us when a model emits broken \u escapes:
# "Na P\u000159\u0000edkop\u00011b" instead of "Na Příkopě". They render as boxes or
# vanish, and they end up inside URLs. Stripped at validation so nothing downstream has to
# cope with them.
_BAD_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")


# A model occasionally writes the text of an escape sequence instead of the character it
# stands for, so "Václav" arrives as "Vu00e1clav" or "Ve1clav" and an en dash as "u2013".
# Restricted to the ranges these mistakes actually land in — accented Latin and punctuation —
# so ordinary text containing a "u" followed by digits is left alone.
# Restricted to the ranges these mistakes land in — accented Latin and punctuation — and
# never after a lowercase letter, so an ordinary word is not chopped into an escape.
_LITERAL_ESCAPE = re.compile(r"(?<![a-z])u(00[a-fA-F][0-9a-fA-F]|20[0-3][0-9a-fA-F])")


def _decode_literal_escapes(text):
    def sub(m):
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return m.group(0)
    return _LITERAL_ESCAPE.sub(sub, text)


def clean_text(value):
    """A display-safe string: no control characters, no lone surrogates, tidy whitespace."""
    return " ".join(_BAD_CHARS.sub("", _decode_literal_escapes(_as_str(value))).split())


# A venue is fed to a map search, so it has to be an address-like name. Trailing notes such
# as "Prague (flight)" or "the hotel (check-in)" describe the activity, and searching for
# them lands nowhere useful.
_VENUE_ASIDE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]\s*$")


def clean_venue(value):
    """A venue name fit for a map query: no parenthetical asides, no leading articles."""
    v = _VENUE_ASIDE.sub("", clean_text(value)).strip(" ,-–—")
    return re.sub(r"^(the|a)\s+", "", v, flags=re.I).strip()


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
            item = {
                "time": _as_str(io.get("time")),
                # Every user-visible string is cleaned: these end up in the itinerary text
                # and inside generated map URLs, where a stray control character is a broken
                # link rather than just an odd glyph.
                "name": clean_text(io.get("name")),
                # The searchable proper name of the place, kept apart from `name` (which is a
                # human label like "Popular spot — stroll and photos"). Feeding that label to
                # Maps produced a keyword search rather than a pin on anywhere in particular.
                # Empty for items that are not a place at all.
                "venue": clean_venue(io.get("venue")),
                # For a leg, where it starts. The item then links to directions
                # venue_from -> venue rather than to a pin on the destination.
                "venue_from": clean_venue(io.get("venue_from")),
                "duration_min": _sane_duration(io.get("duration_min"), io.get("name")),
                "note": clean_text(io.get("note")),
            }
            # A cost the planner never supplied used to become 0 here, which reads
            # downstream as "this is free" — indistinguishable from a genuinely free
            # item, so nothing could tell a skipped price from a real one. The number
            # still defaults to 0 because everything that sums costs needs an int, but
            # the fact that it was never given is kept alongside it.
            amount = _to_number(io.get("cost_eur"))
            item["cost_eur"] = max(0, _as_int(io.get("cost_eur"), 0)) if amount is not None else 0
            if amount is None:
                item["cost_missing"] = True
            items_out.append(item)
        days_out.append({
            "day": _as_int(d.get("day"), i + 1),
            "title": clean_text(d.get("title")),
            "items": items_out,
        })

    if not any(day["items"] for day in days_out):
        return None
    total = sum(it["cost_eur"] for day in days_out for it in day["items"])
    out = {"days": days_out, "total_cost_eur": total}
    # Defects the critic raised that no fix pass resolved before the plan was delivered.
    # They are part of the plan's state, not of the message that announced them: without
    # them here the caveat exists only as prose in a reply nobody can act on, and the next
    # turn has no way to know work was left unfinished.
    unresolved = [t for t in (clean_text(x) for x in _as_list(o.get("open_issues"))) if t]
    if unresolved:
        out["open_issues"] = unresolved
    return out


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
# Advisories are the "good to know" list under an itinerary. Length and count are asked for
# in the critic's prompt rather than enforced here: truncating mid-thought produced worse
# text than a slightly long line, and a hard cap silently dropped real advice. What is left
# here is only tidying — stripping filler that adds nothing at any length.
_ADVISORY_PREAMBLE = re.compile(
    r"^(?:please\s+)?(?:note|be\s+aware|advisory|reminder|tip)\s*[:\-–—]\s*", re.I)
# The critic often appends its own excuse for not verifying something. That is about the
# agent, not the trip, and the caveats banner already says the plan was not fully validated.
_ADVISORY_HEDGE = re.compile(
    r"\s*[\(\[][^\)\]]*(?:cannot|can't|could not|unable to|not)\s+"
    r"(?:verify|confirm|check)[^\)\]]*[\)\]]\s*$", re.I)


def _tidy_advisory(text):
    """Tidy one line of traveller-facing advice. Never truncates.

    Drops a leading "Note:"/"Be aware:" label and any trailing parenthetical about the
    agent's own inability to verify something — that is about the agent, not the trip, and
    the caveats banner already says the plan was not fully validated.
    """
    t = " ".join(_as_str(text).split())
    t = _ADVISORY_HEDGE.sub("", t)
    t = _ADVISORY_PREAMBLE.sub("", t).strip()
    return t[:1].upper() + t[1:] if t else t


def validate_verdict(obj):
    """Coerce the Reflection Layer output. An unknown or garbled verdict is treated as
    FAIL, so an unreadable critic can never green-light a bad plan.

    The critic separates concrete defects (`must_fix`) from advisory travel notes
    (`be_aware`). Merging them made a competent plan look broken: nine equally-alarming
    bullets read as "this itinerary is wrong" when most were "book ahead" and "churches
    close for services". `issues` is retained as the union of both, for logging and for any
    caller still expecting the old flat shape.
    """
    o = as_obj(obj)
    verdict = _as_str(o.get("verdict")).strip().upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    must_fix = [_as_str(x) for x in _as_list(o.get("must_fix"))]
    be_aware = [_as_str(x) for x in _as_list(o.get("be_aware"))]
    if not must_fix and not be_aware:
        # A critic replying in the older flat shape still works: issues are defects.
        must_fix = [_as_str(x) for x in _as_list(o.get("issues"))]
    must_fix = [t for t in (clean_text(x) for x in must_fix) if t]
    be_aware = [t for t in (_tidy_advisory(clean_text(x)) for x in be_aware) if t]
    return {
        "verdict": verdict,
        "must_fix": must_fix,
        "be_aware": be_aware,
        "issues": must_fix + be_aware,
        "fixes": [_as_str(x) for x in _as_list(o.get("fixes"))],
    }
