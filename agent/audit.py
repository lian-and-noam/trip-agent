"""Deterministic checks over a draft itinerary.

Everything here is arithmetic or table lookup — no LLM call, no network. The Reflection
Layer is a model asked to judge a plan, and models check clock arithmetic unreliably: a day
whose items overlap by 20 minutes, or that ends at 01:40, is something a critic notices
only sometimes. Computing those defects here and handing the critic a concrete list is both
more reliable and cheaper, because the critic then spends its attention on judgement — is
this day too packed, does it match their interests — rather than re-deriving sums.

Nothing here rewrites the plan. Every check reports and lets the critic decide — including
the cost checks, which used to impose per-person food-price floors and no longer do: the
floor overwrote costs that had been researched correctly.
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


# Sleeping is not an activity. A planner that writes "Lodging overnight, 22:00, 480 min"
# has described going to bed, but the schedule arithmetic below reads it as an eight-hour
# engagement — so the day "ends at 06:00 the next morning" and "runs 22h end to end", and
# every night of the trip becomes two defects the planner is then sent off to fix. It
# cannot fix them, because nothing is wrong. Their duration is dropped here instead: the
# item still anchors the end of the day, it just does not stretch it.
_OVERNIGHT_WORDS = ("overnight", "lodging", "night at", "check out", "check-out",
                    "sleep", "bed for the night")


def is_overnight(item):
    """True when an item is the traveller going to bed rather than doing something."""
    name = (item.get("name") or "").lower()
    if any(w in name for w in _OVERNIGHT_WORDS):
        return True
    # Unnamed but unmistakable: six hours or more starting in the evening. Six, not four —
    # a long dinner and a bar afterwards is a real late night and must still be reported.
    start, dur = _minutes(item.get("time")), int(item.get("duration_min") or 0)
    return start is not None and start >= 20 * 60 and dur >= 360


def _duration_of(item):
    """Scheduled minutes an item actually occupies. Overnight stays occupy none."""
    return 0 if is_overnight(item) else max(0, int(item.get("duration_min") or 0))


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
            timed.append((start, start + _duration_of(it), it.get("name") or "item"))

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

    problems.extend(_check_free_but_priced(plan))
    problems.extend(_check_gaps(plan))
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


# A note naming a price on an item that costs nothing is the plan contradicting itself:
# "taxi or Leonardo Express" at EUR0, or a coffee described as EUR1.50 inside a free walking
# block. Flagged rather than corrected — only the planner knows which option it meant, and
# inventing a number here would be the same guess in a different place.
_PRICE_IN_TEXT = re.compile(
    r"(?:EUR|€)\s?\d|\d+\s?(?:EUR|€)|\bcosts? depends\b|\bfare\b|\bticket price\b", re.I)


def _check_free_but_priced(plan):
    out = []
    for day in plan.get("days", []):
        for it in day.get("items", []):
            if int(it.get("cost_eur") or 0) > 0:
                continue
            if _PRICE_IN_TEXT.search(it.get("note") or ""):
                out.append(f"Day {day.get('day')}: '{it.get('name')}' is priced at EUR0 but its "
                           "note mentions a cost — choose one option and put its price in "
                           "cost_eur.")
    return out


# --- Cost plausibility --------------------------------------------------------------------
# Live local prices are fetched once per run and handed to the planner, but nothing
# downstream proves the planner used them: EUR0 is a legal cost, so a plan that ignored the
# price block entirely still validates. A dinner costed at EUR0 with a note that says
# nothing about a price is the shape that produces — the check above only fires when the
# note contradicts the number, so a silently free meal passes it.
#
# These are reported apart from check_schedule because they are SUSPICIONS, not defects.
# A EUR0 dinner is usually a missing price and occasionally a meal the hotel includes, and
# only the note and the plan around it distinguish the two — which is a judgement, so the
# critic makes it. An earlier version imposed deterministic price floors here instead; it
# overwrote costs the planner had researched correctly, and is deliberately not coming back.
_COST_EXPLAINED = re.compile(
    r"\b(included|complimentary|free|no charge|covered|prepaid|paid for|"
    r"byo|bring your own|self[- ]catered)\b", re.I)

# Below this many items a EUR0 total is a short plan rather than an uncosted one.
_MIN_ITEMS_FOR_TOTAL = 4


def check_costs(plan):
    """Report costs that look like the live price data never reached the itinerary.

    Returns traveller-readable strings for the critic to judge. Nothing here rewrites the
    plan, and nothing here is asserted as fact.
    """
    out = []
    items = priced = 0
    for day in plan.get("days", []):
        for it in day.get("items", []):
            items += 1
            if int(it.get("cost_eur") or 0) > 0:
                priced += 1
                continue
            name = it.get("name") or "item"
            note = it.get("note") or ""
            if it.get("cost_missing"):
                # Recorded at validation: the planner returned no usable number for this
                # item at all. That is a fact rather than a guess about one, so it needs
                # none of the name and note heuristics the case below relies on.
                out.append(f"'{name}' on day {day.get('day')} was never priced — the "
                           "planner returned no cost for it. Price it from the local "
                           "prices given, or state in the note why it costs nothing.")
                continue
            if not is_meal(name) or is_flexible(name):
                continue
            if _COST_EXPLAINED.search(note):
                continue      # the note accounts for the zero; there is nothing to judge
            if _PRICE_IN_TEXT.search(note):
                continue      # already reported as a verified defect by the check above
            # Phrased without a "Day N:" prefix on purpose: that prefix marks a defect this
            # module computed, and the orchestrator re-derives those before delivery to
            # confirm they are still present. This one cannot be re-derived that way.
            out.append(f"'{name}' on day {day.get('day')} is priced at EUR0, and its note "
                       "does not say what makes a meal there free. Confirm the price, or "
                       "say in the note why it costs nothing.")
    if items >= _MIN_ITEMS_FOR_TOTAL and priced == 0:
        out.append(f"Nothing in this itinerary is costed — all {items} items are EUR0. The "
                   "traveller cannot budget from this; price it from local prices.")
    return out


# A long silent gap reads as a planning hole rather than deliberate free time.
_MAX_GAP_MIN = 90


def _check_gaps(plan):
    out = []
    for day in plan.get("days", []):
        items = [i for i in day.get("items", []) if _minutes(i.get("time")) is not None]
        for previous, nxt in zip(items, items[1:]):
            end = _minutes(previous.get("time")) + _duration_of(previous)
            gap = _minutes(nxt.get("time")) - end
            if gap > _MAX_GAP_MIN:
                out.append(f"Day {day.get('day')}: {gap} min unaccounted between "
                           f"'{previous.get('name')}' and '{nxt.get('name')}' — fill it or "
                           "say it is free time.")
    return out
