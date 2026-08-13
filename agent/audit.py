"""Deterministic checks over a draft itinerary.

Everything here is arithmetic or table lookup — no LLM call, no network. The Reflection
Layer is a model asked to judge a plan, and models check clock arithmetic unreliably: a day
whose items overlap by 20 minutes, or that ends at 01:40, is something a critic notices
only sometimes. Computing those defects here and handing the critic a concrete list is both
more reliable and cheaper, because the critic then spends its attention on judgement — is
this day too packed, does it match their interests — rather than re-deriving sums.

Nothing here rewrites the plan except the food-cost floor, which corrects a systematic model
bias rather than a one-off mistake. Everything else reports and lets the critic decide.
"""
import re

# --- Meal venues -----------------------------------------------------------------------
# Meals point at an AREA rather than a named restaurant. A restaurant recalled from model
# memory cannot be verified — it may have closed or been renamed — and a map pin presents it
# as fact. An area is real by construction; the character of the place goes in the note.
_MEAL_WORDS = ("breakfast", "brunch", "lunch", "dinner", "supper", "coffee", "cafe",
               "café", "drinks", "snack", "meal")

# Words that mark a venue as a specific business rather than a district.
_BUSINESS_MARKERS = ("restaurant", "bistro", "trattoria", "osteria", "brasserie", "grill",
                     "bar ", "pub", "hospoda", "kavárna", "cafe", "café", "steakhouse",
                     "pizzeria", "bakery", "deli", "&", "'s")


def is_meal(name):
    n = (name or "").lower()
    return any(w in n for w in _MEAL_WORDS)


# Open-ended blocks: the traveller chooses what to do, so there is no single place to pin.
_FLEXIBLE_WORDS = ("free time", "free afternoon", "free morning", "free evening", "flexible",
                   "downtime", "at leisure", "leisure time", "rest", "buffer", "your choice",
                   "optional", "shopping", "wander", "explore as you like", "unstructured")


def is_flexible(name):
    n = (name or "").lower()
    return any(w in n for w in _FLEXIBLE_WORDS)


def looks_like_a_business(venue):
    """Heuristic: does this venue name an establishment rather than an area?"""
    v = (venue or "").strip().lower()
    if not v:
        return False
    if any(m in v for m in _BUSINESS_MARKERS):
        return True
    return bool(re.match(r"^(u |the )\w", v))


# --- Food costs ------------------------------------------------------------------------
# Models systematically lowball restaurant prices: they recall averages skewed by cheap
# eateries and omit drinks, service and tourist-area markup. These are per-person floors,
# not estimates — a plan may cost more, but a mid-range dinner is not €12.
_FOOD_FLOOR_EUR = {
    "low":       {"breakfast": 6,  "lunch": 10, "dinner": 14, "coffee": 3, "drinks": 6},
    "mid-range": {"breakfast": 10, "lunch": 18, "dinner": 26, "coffee": 5, "drinks": 10},
    "luxury":    {"breakfast": 20, "lunch": 38, "dinner": 65, "coffee": 8, "drinks": 18},
}


def _meal_kind(name):
    n = (name or "").lower()
    for kind in ("breakfast", "lunch", "dinner", "coffee", "drinks"):
        if kind in n:
            return kind
    if "brunch" in n:
        return "lunch"
    if "supper" in n:
        return "dinner"
    if "café" in n or "cafe" in n:
        return "coffee"
    return "lunch"


def apply_food_floors(plan, budget_tier):
    """Raise implausibly cheap meal costs to a realistic per-person floor.

    Returns (plan, corrections) so the change is visible rather than silent. Only ever
    raises: a planner that says dinner costs €40 is making a choice, and that is left alone.
    """
    floors = _FOOD_FLOOR_EUR.get(budget_tier, _FOOD_FLOOR_EUR["mid-range"])
    corrections, days = [], []
    for day in plan.get("days", []):
        items = []
        for it in day.get("items", []):
            new = dict(it)
            if is_meal(it.get("name")):
                floor = floors[_meal_kind(it.get("name"))]
                if int(new.get("cost_eur") or 0) < floor:
                    corrections.append(f"Day {day.get('day')}: {it.get('name')} "
                                       f"EUR{new.get('cost_eur', 0)} -> EUR{floor}")
                    new["cost_eur"] = floor
            items.append(new)
        days.append({**day, "items": items})
    out = {**plan, "days": days}
    out["total_cost_eur"] = sum(int(i.get("cost_eur") or 0)
                                for d in days for i in d.get("items", []))
    return out, corrections


# --- Schedule arithmetic ----------------------------------------------------------------
_TIME = re.compile(r"^\s*(\d{1,2})[:.](\d{2})\s*$")


def _minutes(time_str):
    """"09:30" -> 570. None when unparseable, so callers skip rather than guess."""
    m = _TIME.match(time_str or "")
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h * 60 + mi if 0 <= h < 24 and 0 <= mi < 60 else None


def _hhmm(total):
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def check_schedule(plan, profile=None):
    """Report concrete timing defects as short, traveller-readable strings.

    Deliberately conservative: an item with an unparseable time is skipped rather than
    guessed at. A false "these overlap" is worse than a missed one, because it sends the
    planner off to rewrite a day that was fine.
    """
    profile = profile or {}
    problems = []
    for day in plan.get("days", []):
        n = day.get("day")
        timed = []
        for it in day.get("items", []):
            start = _minutes(it.get("time"))
            if start is None:
                continue
            timed.append((start, start + max(0, int(it.get("duration_min") or 0)),
                          it.get("name") or "item"))

        for i in range(1, len(timed)):
            prev_start, prev_end, prev_name = timed[i - 1]
            start, _, name = timed[i]
            if start < prev_start:
                problems.append(f"Day {n}: '{name}' is scheduled before '{prev_name}' "
                                "but listed after it.")
            elif start < prev_end:
                problems.append(f"Day {n}: '{prev_name}' runs to {_hhmm(prev_end)} but "
                                f"'{name}' starts at {_hhmm(start)}.")

        if timed:
            last_end = max(e for _, e, _ in timed)
            first_start = min(s for s, _, _ in timed)
            if last_end > 24 * 60:
                problems.append(f"Day {n} ends at {_hhmm(last_end)} the next morning.")
            elif last_end > 23 * 60 + 30:
                problems.append(f"Day {n} ends very late ({_hhmm(last_end)}).")
            if last_end - first_start > 14 * 60:
                problems.append(f"Day {n} runs {(last_end - first_start) // 60}h end to end — "
                                "too long for one day.")

    problems.extend(_check_trip_anchors(plan, profile))
    return problems


def _clock_in(text):
    """Pull a clock time out of "20/8 18:30". None when there is no time of day."""
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", text or "")
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h * 60 + mi if 0 <= h < 24 and 0 <= mi < 60 else None


def _check_trip_anchors(plan, profile):
    """The stated start and end of the trip are hard constraints, not preferences."""
    out = []
    days = plan.get("days") or []
    if not days:
        return out

    start_clock = _clock_in(profile.get("start_time"))
    if start_clock is not None:
        starts = [m for m in (_minutes(i.get("time")) for i in days[0].get("items", []))
                  if m is not None]
        if starts and min(starts) < start_clock:
            out.append(f"Day 1 starts at {_hhmm(min(starts))} but the trip does not begin "
                       f"until {_hhmm(start_clock)}.")

    end_clock = _clock_in(profile.get("end_time"))
    if end_clock is not None:
        ends = [(_minutes(i.get("time")) or 0) + int(i.get("duration_min") or 0)
                for i in days[-1].get("items", []) if _minutes(i.get("time")) is not None]
        if ends and max(ends) > end_clock:
            out.append(f"The last day runs to {_hhmm(max(ends))}, past the "
                       f"{_hhmm(end_clock)} departure.")
        elif ends and profile.get("end_point") and end_clock - max(ends) < 120:
            out.append(f"Only {end_clock - max(ends)} min between the last activity and the "
                       f"{_hhmm(end_clock)} departure — too tight.")
    return out


# --- Opening hours ----------------------------------------------------------------------
def _parse_osm_hours(spec):
    """(open, close) minute spans for a simple OSM opening_hours string.

    Handles the common shapes ("Mo-Su 09:00-17:00", "09:00-17:00", "24/7") and gives up on
    anything more exotic. Giving up returns None, which callers treat as UNKNOWN — never as
    closed, because a wrong "this is shut" would move a stop that was perfectly fine.
    """
    s = (spec or "").strip().lower()
    if not s:
        return None
    if "24/7" in s:
        return [(0, 24 * 60)]
    spans = re.findall(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", s)
    if not spans:
        return None
    out = []
    for h1, m1, h2, m2 in spans:
        start, end = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
        if end <= start:
            end += 24 * 60                     # crosses midnight
        out.append((start, end))
    return out or None


def check_opening_hours(plan, hours_by_venue):
    """Flag items scheduled outside hours a real lookup actually returned.

    `hours_by_venue` holds only venues we have hours for. Anything absent is unknown and is
    not checked: the point is to use facts we have, not to invent the ones we lack.
    """
    problems = []
    lookup = {k.strip().lower(): v for k, v in (hours_by_venue or {}).items() if v}
    for day in plan.get("days", []):
        for it in day.get("items", []):
            spec = lookup.get((it.get("venue") or "").strip().lower())
            spans = _parse_osm_hours(spec) if spec else None
            start = _minutes(it.get("time"))
            if not spans or start is None:
                continue
            end = start + max(0, int(it.get("duration_min") or 0))
            if not any(o <= start and end <= c for o, c in spans):
                problems.append(f"Day {day.get('day')}: '{it.get('name')}' is "
                                f"{_hhmm(start)}-{_hhmm(end)} but {it.get('venue')} "
                                f"opens {spec}.")
    return problems


# --- Salvage ------------------------------------------------------------------------------
# Fallback times for a day assembled without the planner. Deliberately plain: this runs when
# the model could not finish, so the aim is a usable shape, not a clever one.
_SALVAGE_SLOTS = [("09:00", 120), ("11:30", 90), ("13:30", 60), ("15:00", 120),
                  ("17:30", 90), ("19:30", 90)]


def salvage_plan(profile, venues):
    """Build a real itinerary from places already looked up, when the planner cannot finish.

    The alternative is a page of "Explore <city> (self-guided)", which throws away everything
    the run learned — the venues it found, their addresses and opening hours. Spreading those
    across the days gives the traveller something they can actually use, and the caveat above
    it says plainly that it was assembled without the planner.

    Returns None when there is nothing real to place, so the caller keeps its own fallback.
    """
    named = [v for v in dict.fromkeys(venues or []) if v and v.strip()]
    if not named:
        return None

    days_count = max(1, int(profile.get("days") or 1))
    per_day = max(1, -(-len(named) // days_count))       # ceiling division
    days = []
    for index in range(days_count):
        chunk = named[index * per_day:(index + 1) * per_day]
        if not chunk:
            break
        items = []
        for slot, (time_str, minutes) in zip(chunk, _SALVAGE_SLOTS):
            items.append({"time": time_str, "name": slot, "venue": slot,
                          "duration_min": minutes, "cost_eur": 0,
                          "note": "Times are a suggestion — check opening hours before you go."})
        days.append({"day": index + 1,
                     "title": f"Day {index + 1} in {profile.get('destination') or 'town'}",
                     "items": items})
    if not days:
        return None
    return {"days": days, "total_cost_eur": 0, "salvaged": True}
