"""Agent orchestrator with multi-turn Conversational Intake and a plan revision path.

One intake call reads the conversation and routes to one of five branches:
  A) required info still missing  -> ask ONE clarifying question, stop.        1 call
  B) complete but unconfirmed     -> show the typed profile, ask to confirm.   1 call
  C) confirmed, no plan yet       -> ReAct Planner (<-> Reflection) -> Format.  ~9 calls
  D) plan exists, user asks a
     question about it            -> Itinerary Q&A, no re-planning.            2 calls
  E) plan exists, user wants a
     change                       -> Plan Editor -> Output Formatter.          3 calls

Branches D and E are the point of the revision path. "Make day 2 lighter" used to re-run
the whole planner; it now patches the stored plan, which is roughly a 3x reduction on the
most common follow-up. A change to any REQUIRED profile field is deliberately NOT an edit
— it re-enters the confirm/plan flow, because a new destination means a new trip.

State lives in Supabase, keyed by conversation_id, so the itinerary never has to be
replayed through the prompt to be revisable. Without Supabase configured the agent still
works exactly as before: the caller simply passes no prior state and every turn plans
fresh. Persistence is an enhancement, never a dependency.

Every LLM call is logged as a step {module, prompt, response} with module names matching
the architecture diagram (Conversational Intake, ReAct Planner, Plan Editor, Reflection
Layer, Output Formatter, Itinerary Q&A). Steps contain ONLY real LLM calls — the
deterministic validation layer in schemas.py is not a model call and is not traced as one.
"""
import datetime
import json
import os
import re
import time
import urllib.parse

from .llm import chat, is_timeout, parse_json, set_wall as llm_set_wall
from .tools import run_tool, route_matrix, ToolError, TOOL_CATALOG, geocode_place
from . import audit, schemas, obs, usage

# The planner is bounded by both a step count and a clock, whichever binds first.
# Research past ~8 steps mostly re-checks known facts, while a run that spends its whole
# budget researching delivers an unfinished itinerary.
MAX_PLANNER_STEPS = int(os.environ.get("MAX_PLANNER_STEPS", "8"))
# Time held back so the planner can always finish writing its itinerary, sized for one
# call. Everything after the planner degrades on its own — the formatter falls back to a
# deterministic renderer — so only the planner's final output needs protecting.
# Must be at least a whole LLM timeout. The finalize call writes the entire itinerary — the
# largest response in the pipeline — and if it is started with less than a full call's worth
# of time it is cut short and the run degrades to placeholder days. Research stops early
# enough that this call always gets its full allowance.
PLANNER_FINALIZE_RESERVE_S = int(os.environ.get("PLANNER_FINALIZE_RESERVE_S", "130"))

# Ceiling on network time across one planner run. Tools make real HTTP calls, which share
# the run's budget with the model calls. Past this, tools are refused with a note telling
# the planner to work from what it has, rather than running the clock out.
MAX_TOOL_SECONDS = int(os.environ.get("MAX_TOOL_SECONDS", "35"))

# Consecutive failed lookups before the planner is told to stop researching. Each retry
# costs a full LLM turn, so a tool that will not answer can eat a run on its own.
MAX_TOOL_FAILURES = int(os.environ.get("MAX_TOOL_FAILURES", "2"))

# Below this, the critic is skipped so the formatter keeps its time.
REVIEW_MIN_SECONDS = int(os.environ.get("REVIEW_MIN_SECONDS", "75"))

# Below this the itinerary is rendered deterministically rather than by the model: a
# formatter call that cannot finish costs the time and returns nothing.
FORMAT_MIN_SECONDS = int(os.environ.get("FORMAT_MIN_SECONDS", "70"))
# Two passes: the critic reviews, the planner fixes what it found, and the fixed draft is
# checked again. One pass meant defects were reported to the traveller but never repaired.
MAX_REFLECT_CYCLES = int(os.environ.get("MAX_REFLECT_CYCLES", "2"))
MAX_OBS_CHARS = 1200      # trim tool observations fed back to the model to keep context small

# Wall-clock budget for one turn. Vercel terminates the function at vercel.json's
# maxDuration (300s). A call that starts just before the deadline still runs for up to
# llm._TIMEOUT_S afterwards. llm.set_wall() truncates each call to the time actually
# remaining, so no call can overshoot however late it starts. MAX_RUN_SECONDS is the work
# budget; HARD_WALL_SECONDS is the absolute stop, set below the platform's 300s limit with
# room to serialise and return the response.
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "260"))
HARD_WALL_SECONDS = int(os.environ.get("HARD_WALL_SECONDS", "270"))

# Held back for the Output Formatter, which runs after the review loop and is the slowest
# call in the pipeline. Without it the loop can spend the budget down to the deadline and
# leave the formatter nothing, which overruns the platform limit instead of degrading.
DELIVER_RESERVE_S = int(os.environ.get("DELIVER_RESERVE_S", "60"))


def _trace(steps, module, prompt, response):
    """Append one step in the required schema: {module, prompt, response}."""
    steps.append({"module": module, "prompt": prompt, "response": response})


def _expired(deadline):
    """Hard time gate: once this is True, no further LLM call may start.

    Checked immediately before every call in the pipeline, so a run degrades to whatever
    it has already built rather than being killed mid-flight by the platform.
    """
    return deadline is not None and time.monotonic() >= deadline


def _low_on_time(deadline, reserve_s):
    """True when less than `reserve_s` of the run budget remains.

    Distinct from _expired: this stops *optional* work (more tool calls) while there is
    still time to do the required work (writing the itinerary).
    """
    return deadline is not None and (deadline - time.monotonic()) < reserve_s


def _abort_plan(prof, steps, run_id, reason="the run deadline was reached", venues=None):
    """Return a valid plan without spending an LLM call, for when time has run out.

    Where the run already found real places, they are laid out across the days rather than
    thrown away: a page of "Explore <city> (self-guided)" discards everything the turn
    learned and helps nobody. Only when nothing was found does it fall back to the skeleton.
    """
    plan = audit.salvage_plan(prof, venues) or schemas.minimal_plan(prof)
    plan["timed_out"] = True
    _trace(steps, "ReAct Planner", {"thought": reason, "action": "abort"},
           {"draft_plan": plan})
    obs.log("planned", run_id=run_id, forced=True, timed_out=True,
            degraded=True, reason=reason, cost_eur=plan.get("total_cost_eur"))
    return plan


def _chat_json(messages, temperature, max_tokens, repairs=1):
    """One JSON LLM turn with a bounded repair loop. Returns a parsed value or None.

    If the model returns non-JSON, ask once more for a JSON-only reply before giving up.
    The repair is not logged as a separate step, so the trace stays at one step per
    logical turn, and the retry count is small so a misbehaving model can't run up cost.
    """
    raw = chat(messages, json_mode=True, temperature=temperature, max_tokens=max_tokens)
    obj = parse_json(raw)
    attempts = 0
    while obj is None and attempts < repairs:
        attempts += 1
        repair_msgs = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your previous reply was not valid JSON. "
                                        "Reply with ONLY one JSON object — no prose, no code fences."},
        ]
        raw = chat(repair_msgs, json_mode=True, temperature=0.0, max_tokens=max_tokens)
        obj = parse_json(raw)
    return obj


# ---------- Module: Conversational Intake ----------
INTENTS = ("revise", "replace", "question")


def _profile(conversation, has_plan=False):
    """One LLM call over the conversation. Extracts the typed profile, flags which required
    fields are still missing, detects confirmation, and — only when an itinerary already
    exists — classifies what the user wants done with it. Returns a decision dict; the
    branching happens in _run_turn().

    The intent block is appended only when it can matter, so an ordinary intake turn keeps
    the smallest prompt it can (project brief: minimize prompt/context size).
    """
    usage.set_module("Conversational Intake")
    sys = (
        "You are the Conversational Intake for a trip planner. You receive the FULL conversation "
        "so far as a transcript ('User:' lines are the traveller; 'Agent:' lines are your own "
        "earlier replies). Extract a typed trip profile from everything the user has said.\n"
        'Return ONLY JSON: {"profile":{...}, "confirmed":bool, "question":string' +
        (', "intent":string}' if has_plan else "}") + ".\n"
        "profile keys: days (int), destination (string), "
        "style (string: what the traveller ENJOYS — food, culture, nature, art, nightlife "
        "— NOT their pace or tempo), group (string), "
        'budget ("low"|"mid-range"|"luxury"), budget_amount_eur (int: a stated amount converted '
        'to EUR, 0 if none), budget_basis ("total"|"per person"), '
        'and optionally when (string: travel dates or '
        "season), "
        "start_point (string: where the trip physically begins — a named hotel, station or "
        "address, if the user gave one), end_point (string: where it must end — often an "
        "airport or station), start_time (string: date and clock time the trip begins), "
        "end_time (string: date and clock time it must end), lodging (string: the hotel name "
        "if named), details (string[]: any other concrete thing the traveller stated that no "
        "other field captures — a rental car, someone joining midway, no early mornings, an "
        "occasion being celebrated, a fixed appointment), dietary (string[]), "
        'walking ("light"|"moderate"|"high"|"unlimited" walking tolerance), '
        "accessibility (bool), priorities (string[]: specific places or experiences the user "
        "named as must-see), avoid (string[]).\n"
        "RULES:\n"
        "- For any REQUIRED field the user has NOT stated or clearly implied "
        "(days, destination, group, budget), set it to null. NEVER invent required fields.\n"
        "- Record group EXACTLY as the traveller phrased it (\"2 friends\" stays \"2 friends\"). "
        "Never add the traveller to the count, never restate it as a headcount, never append "
        "a parenthetical total. If the phrasing is ambiguous, leave it ambiguous.\n"
        "- An AMOUNT is a budget. \"2000 euro\", \"$1500\", \"about 800\" all count: set "
        "budget_amount_eur to the number (converting to EUR if needed), set budget_basis to "
        '"total" if it covers the whole party or "per person" if it is each, and ALSO pick the '
        "closest tier for budget. Never re-ask for a tier when an amount was given.\n"
        "- If the traveller describes how much walking they want (\"a lot of walking\", \"we "
        "walk everywhere\", \"easy days\", \"minimal walking\"), that is the walking field: map "
        'it to "light", "moderate", "high" or "unlimited". Do not drop it and do not confuse '
        "it with style.\n"
        "- style is OPTIONAL. Never ask for it and never treat its absence as incomplete: a "
        "profile with days, destination, group and budget IS complete. Capture style only if "
        "the user mentions interests.\n"
        '- If the user names interests anywhere ("love food and culture"), that IS style. '
        "Set style from them and never ask about pace.\n"
        "- Do not ask about optional fields; only capture them if the user mentions them. But DO "
        "capture every concrete detail the user gives. \"from 17/8 15:00 in Prague Bank Hotel to "
        "20/8 18:30 flight from Prague\" means start_time=\"17/8 15:00\", start_point=\"Prague Bank "
        "Hotel\", lodging=\"Prague Bank Hotel\", end_time=\"20/8 18:30\", end_point=\"Václav Havel "
        "Airport Prague\". Never discard a stated place or time.\n"
        "- start_point and end_point must be PLACES you could search on a map — an airport, "
        "station, hotel or address. \"18:30 flight from Prague\" means the airport, so write "
        "\"Václav Havel Airport Prague\", not \"Prague (flight)\" or \"flight from Prague\".\n"
        "- Anything concrete the traveller states that no typed field fits goes into details "
        "as its own short string, kept across turns. Nothing they took the trouble to tell you "
        "should be dropped merely because the schema has no column for it.\n"
        "- confirmed = true ONLY IF an earlier 'Agent:' turn already summarised the trip "
        "and asked the user to confirm, AND the user's LATEST message clearly agrees to proceed "
        "(e.g. 'yes', 'yep', 'looks good', 'correct', 'go ahead'). A summary that notes some "
        "optional detail is unset still counts as asking to confirm. Otherwise false.\n"
        "- question: if any required field is null, ask ONE friendly message that requests ALL the "
        'missing required fields together (not one at a time). If nothing is missing, set it to "". '
        "Format it to be read at a glance, not as one dense paragraph: a short lead-in line, then "
        "each thing you need on its own line starting with '- '. Put any examples in brackets on "
        "that same line. Never run two questions together in one sentence."
    )
    if has_plan:
        sys += (
            "\nAn itinerary has ALREADY been delivered in this conversation. Classify the user's "
            "LATEST message as intent:\n"
            '- "question": they are asking about the existing itinerary and want nothing changed '
            "(e.g. 'what does day 3 cost?', 'how far is that from the hotel?').\n"
            '- "revise": they want a change to the existing itinerary — a specific day, activity, '
            "pace, or spend (e.g. 'make day 2 lighter', 'swap the museum for a market').\n"
            '- "replace": they want a DIFFERENT trip — another destination, a different number of '
            "days, a different group or budget tier. Any change to a required field is a replace."
        )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": conversation}]
    obj = schemas.as_obj(_chat_json(msgs, temperature=0.2, max_tokens=700))
    profile = schemas.as_obj(obj.get("profile"))
    q = obj.get("question")
    intent = schemas._as_str(obj.get("intent")).strip().lower()
    missing = schemas.missing_required(profile)
    confirmed = bool(obj.get("confirmed"))
    # Deterministic backstop. The model can return confirmed=false after a plain "yes" if it
    # judges the profile incomplete over an unset optional field, leaving the traveller
    # re-confirming with no way forward. An unambiguous yes, after we asked and with nothing
    # required missing, is a confirmation whatever the model says.
    if not confirmed and not missing and _asked_to_confirm(conversation) \
            and _is_affirmative(_latest_user_message(conversation)):
        confirmed = True
    return {
        "profile": profile,
        "missing": missing,
        "confirmed": confirmed,
        "question": q if isinstance(q, str) else "",
        # Unrecognised or absent intent falls back to "revise": editing the existing plan is
        # the cheap, reversible option, where guessing "replace" would silently discard it.
        "intent": intent if intent in INTENTS else "revise",
    }


# Short, unambiguous agreements only. Anything longer or hedged ("yes but make it cheaper")
# is left to the model, which can tell agreement from an agreement-plus-edit.
_AFFIRMATIVES = {
    "y", "yes", "yes.", "yes!", "yep", "yup", "yeah", "ya", "sure", "ok", "okay", "k",
    "correct", "confirm", "confirmed", "looks good", "sounds good", "looks right",
    "go ahead", "go", "proceed", "start", "start planning", "plan it", "do it",
    "perfect", "great", "exactly", "that's right", "thats right", "לך על זה", "כן",
}


def _latest_user_message(conversation):
    """The text of the last 'User:' line in the transcript, or ''."""
    for line in reversed((conversation or "").splitlines()):
        if line.startswith("User:"):
            return line[len("User:"):].strip()
    return ""


def _is_affirmative(text):
    t = (text or "").strip().lower().strip(".!, ")
    return t in _AFFIRMATIVES


def _asked_to_confirm(conversation):
    """True if we already showed a trip summary and asked the user to confirm it."""
    return CONFIRM_MARKER.lower() in (conversation or "").lower()


# Fields whose change means a genuinely different trip. `style` is deliberately excluded:
# it is free text that the model rephrases every turn ("cultural and food-focused" ->
# "cultural/food-focused"), which can read as a new destination and force a spurious
# re-confirmation plus a full re-plan — 9 calls where 3 would do. A real change of interests
# still reaches the Plan Editor as an ordinary revision.
TRIP_IDENTITY_FIELDS = ("destination", "days", "budget", "group")


def _required_changed(before, after):
    """True when a field that defines the *identity of the trip* differs between profiles.

    A deterministic backstop to the model's intent call. The product rule is that changing
    where, when, how long, for whom, or at what budget means a new trip rather than an edit,
    so this forces the confirm/plan flow even if intake classified the turn as a revision.
    """
    for f in TRIP_IDENTITY_FIELDS:
        b, a = before.get(f), after.get(f)
        if isinstance(b, str) and isinstance(a, str):
            if b.strip().lower() != a.strip().lower():
                return True
        elif b != a:
            return True
    return False


def _fallback_question(missing):
    """Deterministic clarifying question if the model did not supply one.

    Laid out one item per line for the same reason the prompt asks the model to: several
    questions run together in a single paragraph are easy to half-answer.
    """
    labels = {"destination": "Where would you like to go?",
              "days": "How many days? (e.g. 4)",
              "budget": "What budget? (low, mid-range, luxury — or an amount, e.g. €2000 total)",
              "group": "Who's travelling? (solo, couple, family, friends)",
              "style": "What do you enjoy? (food, culture, nature, art, nightlife)"}
    parts = [labels.get(m, m) for m in missing]
    if len(parts) == 1:
        return parts[0]
    return "A couple of things and I can start planning:\n\n" + "\n".join(f"- {p}" for p in parts)


_DATE_NUMBERS = re.compile(r"\d+")


def _trip_window(prof):
    """The exact start/end times as one string, or "" when neither was given."""
    return " → ".join(t for t in (prof.get("start_time"), prof.get("end_time")) if t)


def _same_dates(when, window):
    """True when `when` adds nothing the trip window does not already show.

    Compares the numbers in each: "17/8 - 20/8" and "17/8 15:00 → 20/8 18:30" share every
    number in the vaguer one, so it is a duplicate. "August" has no numbers at all and is
    genuinely extra information, so it is kept.
    """
    when = (when or "").strip()
    if not when or not window:
        return False
    a, b = set(_DATE_NUMBERS.findall(when)), set(_DATE_NUMBERS.findall(window))
    return bool(a) and a.issubset(b)


CONFIRM_MARKER = "Does this look right?"


def _confirmation_message(prof):
    """Branch B reply: show the trip as a readable card and ask the user to confirm.

    This used to print the raw profile JSON in a fenced code block. It was precise and
    unreadable — braces, quotes and snake_case keys for something a traveller is being asked
    to check at a glance. The profile is still typed JSON everywhere else; only this one
    user-facing rendering is humanised.

    The summary lists only what the user actually gave. An earlier version added an
    "Interests: not specified" row, which read as a defect in the summary and made the model
    judge the profile incomplete — so a plain "yes" never registered as a confirmation. The
    invitation to add interests now sits after the question, where it is an offer rather than
    a missing field.
    """
    labels = (("destination", "Destination"), ("days", "Days"), ("group", "Travellers"),
              ("budget", "Budget"), ("style", "Interests"), ("when", "Dates"),
              ("walking", "Walking"), ("dietary", "Dietary"),
              ("priorities", "Must-see"), ("avoid", "Avoid"))

    def fmt(v):
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v if x)
        return str(v)

    rows = []
    for key, label in labels:
        value = prof.get(key)
        if not value:
            continue
        if key == "walking" and value == "moderate":
            continue          # the default; showing it implies the traveller chose it
        if key == "when" and _same_dates(value, _trip_window(prof)):
            continue          # the exact window below says this and more
        if key == "budget" and prof.get("budget_amount_eur"):
            basis = "total" if prof.get("budget_basis") == "total" else "per person"
            value = f"{value} (~€{prof['budget_amount_eur']:,} {basis})"
        rows.append(f"- **{label}:** {fmt(value)}")
    if prof.get("accessibility"):
        rows.append("- **Accessibility:** step-free / reduced mobility")

    # The trip anchors are folded into compact rows rather than getting one each: they matter
    # to the planner, but five separate rows doubled the length of a card meant to be glanced
    # at. Nothing the traveller supplied is hidden, though — if they said it, they see it.
    window = _trip_window(prof)
    if window:
        # Kept alongside `when` when they say different things ("August" plus exact times).
        # When they are the same dates twice, the Dates row above is the one dropped — it is
        # the vaguer copy, and losing the stated arrival hour is the worse outcome.
        rows.append(f"- **Trip window:** {window}")
    start = prof.get("start_point") or prof.get("lodging")
    if start:
        rows.append(f"- **Start point:** {start}")
    if prof.get("end_point"):
        rows.append(f"- **End point:** {prof['end_point']}")
    # Only when it adds something: as the start point it is already on the card above.
    lodging = prof.get("lodging")
    if lodging and lodging != start:
        rows.append(f"- **Staying at:** {lodging}")
    if prof.get("details"):
        rows.append(f"- **Also noted:** {'; '.join(prof['details'])}")

    note = ""
    if prof.get("assumptions"):
        note = "\n\n_Assumptions I made: " + "; ".join(prof["assumptions"]) + "_"

    nudge = ""
    if not prof.get("style"):
        nudge = ("\n\nIf you tell me what you enjoy — food, culture, nature, art, nightlife — "
                 "I'll tailor it to that. Otherwise I'll aim for a good mix.")
    return ("**Here's your trip so far**\n\n" + "\n".join(rows) + note +
            f"\n\n{CONFIRM_MARKER} Reply **yes** to start planning — or tell me what to change."
            + nudge)


# ---------- Module: ReAct Planner ----------
# Travel times are corrected once, AFTER the draft exists, rather than during the ReAct
# loop. Two reasons: the loop's tool budget is for research the planner asked for, and a
# whole day routes in a single Route Matrix call once the stops are known — during planning
# they are not. Bounded by its own clock and failure-silent: an itinerary with estimated
# walking times is fine, an itinerary that never arrives is not.
ROUTE_FIX_SECONDS = int(os.environ.get("ROUTE_FIX_SECONDS", "20"))


def _venue_points(items, coords_seen):
    """(index, (lat, lon)) for items whose venue we already have coordinates for."""
    out = []
    for i, it in enumerate(items):
        name = (it.get("venue") or "").strip()
        if name and name in coords_seen:
            out.append((i, coords_seen[name]))
    return out


def _apply_travel_times(plan, prof, coords_seen, run_id=None):
    """Replace guessed transfer durations with measured ones, and shift the day to match.

    Only touches legs (`is_leg`-shaped items with a venue_from) whose BOTH ends have known
    coordinates, so nothing is invented. Returns (plan, changed_count).
    """
    if not coords_seen:
        return plan, 0
    started = time.monotonic()
    mode = "drive" if "car" in " ".join(prof.get("details") or []).lower() else "walk"
    changed, days = 0, []

    for day in plan.get("days", []):
        items = list(day.get("items", []))
        points = _venue_points(items, coords_seen)
        if len(points) < 2 or time.monotonic() - started > ROUTE_FIX_SECONDS:
            days.append(day)
            continue
        try:
            legs = route_matrix([p for _, p in points], mode=mode)
        except ToolError:
            legs = None
        except Exception:                       # never let this break delivery
            legs = None
        if not legs:
            days.append(day)
            continue

        # legs[k] is the time from points[k] to points[k+1]; attribute it to a transfer item
        # sitting between them, if the planner made one.
        for k, minutes in enumerate(legs):
            start_i, end_i = points[k][0], points[k + 1][0]
            for idx in range(start_i + 1, end_i + 1):
                it = items[idx]
                if it.get("venue_from") and audit.is_meal(it.get("name")) is False:
                    if abs(int(it.get("duration_min") or 0) - minutes) >= 5:
                        items[idx] = {**it, "duration_min": minutes,
                                      "note": _with_travel_note(it.get("note"), minutes, mode)}
                        changed += 1
                    break
        days.append({**day, "items": _resequence(items)})

    if changed:
        obs.log("travel_times_applied", run_id=run_id, legs=changed)
    return {**plan, "days": days}, changed


def _with_travel_note(note, minutes, mode):
    """Say the time is measured, without naming the tool that measured it."""
    label = f"About {minutes} min on foot." if mode == "walk" else f"About {minutes} min by car."
    return f"{note.rstrip('.')}. {label}" if note else label


def _resequence(items):
    """Push later start times forward when a corrected duration overruns the next start.

    Only ever moves times forward, and only parseable ones. A day that admits it now runs
    later is more useful than one claiming the next stop starts before the transfer ends.
    """
    out, carry = [], 0
    starts = [audit._minutes(it.get("time")) for it in items]
    for i, it in enumerate(items):
        start = starts[i]
        out.append(it if start is None or carry == 0
                   else {**it, "time": audit._hhmm(start + carry)})
        if start is None:
            continue
        end = start + carry + max(0, int(it.get("duration_min") or 0))
        nxt = starts[i + 1] if i + 1 < len(items) else None
        if nxt is not None and end > nxt + carry:
            carry = end - nxt
    return out


# Defects the audit produces have a recognisable shape. Only those can be re-verified after
# a fix; the critic's own judgements ("day 2 feels rushed") cannot be re-checked in code, so
# they are kept as written.
_COMPUTED_DEFECT = re.compile(r"^(Day \d+[:.]|The last day |Only \d+ min )")


def _is_computed_defect(text):
    return bool(_COMPUTED_DEFECT.match((text or "").strip()))


def _localise(tool, tool_input, prof):
    """Fill in the destination on a place lookup that did not name one.

    A bare "Colosseum" resolves to a hamlet in Australia. The planner knows the destination
    and usually omits it as obvious, so supplying it here turns a wrong answer into a right
    one without costing a turn to notice and retry.
    """
    if tool != "maps_tool" or not isinstance(tool_input, dict):
        return tool_input
    destination = (prof.get("destination") or "").strip()
    if not destination or tool_input.get("near"):
        return tool_input
    return {**tool_input, "near": destination}


def _remember_hours(cache, observation, coords_cache=None):
    """Record what a maps_tool lookup returned, keyed by venue name.

    Only entries with actual hours are kept in `cache`: a venue OpenStreetMap has not tagged
    is unknown, and storing None would make the checker treat unknown as checked.
    Coordinates are kept separately and are free — they came back with the same response, so
    the routing pass later can reuse them instead of geocoding again.
    """
    if not isinstance(observation, dict):
        return
    for r in (observation.get("results") or []):
        if not isinstance(r, dict) or not r.get("name"):
            continue
        if cache is not None and r.get("open_hours"):
            cache.setdefault(r["name"], r["open_hours"])
        if coords_cache is not None and r.get("lat") is not None and r.get("lon") is not None:
            coords_cache.setdefault(r["name"], (r["lat"], r["lon"]))


def _prices_for(prof):
    """Live local prices for the destination, or None.

    One search for the whole trip, fetched here rather than left to the planner. Costing a
    trip from model memory produces numbers that are years out of date, and looking each item
    up would cost a Thought/Action/Observation cycle per item. This is a single HTTP call
    that adds no LLM turns at all.

    Not cached: a stale price is the problem this exists to solve.
    """
    destination = (prof.get("destination") or "").strip()
    if not destination:
        return None
    year = datetime.date.today().year
    query = (f"{destination} typical tourist prices {year} coffee lunch dinner "
             "museum entry ticket public transport ticket")
    try:
        data = run_tool("search_tool", {"query": query})
    except Exception:
        return None                      # prices are a bonus; never block the plan
    if not isinstance(data, dict) or not data.get("ok") or data.get("fictive"):
        return None
    snippets = [x.get("content") for x in (data.get("snippets") or [])
                if isinstance(x, dict) and x.get("content")]
    if not (data.get("answer") or snippets):
        return None
    return {"summary": data.get("answer"), "sources": snippets[:3]}


def _forecast_for(prof, coords):
    """Compact daily forecast for the destination, or None.

    Fetched here, once, off the geocode we already did — so weather reaches the planner
    without it spending a tool turn asking for it.
    """
    if not coords:
        return None
    try:
        data = run_tool("weather_tool", {"location": prof.get("destination", "")})
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    return [{"date": d.get("date"), "max_c": d.get("max_c"), "rain_pct": d.get("rain_pct")}
            for d in (data.get("daily") or [])[:8]]


def _plan(prof, steps, feedback=None, run_id=None, deadline=None,
          forecast=None, prices=None, hours_seen=None, coords_seen=None):
    usage.set_module("ReAct Planner")
    sys = (
        "You are the ReAct Planner for a trip. Work in a Thought -> Action -> Observation loop.\n"
        "Tools:\n" + TOOL_CATALOG + "\n\n"
        "On EACH turn return ONLY JSON, one of:\n"
        '  {"thought":"...","tool":"<tool_name>","tool_input":{...}}\n'
        '  {"thought":"...","done":true,"draft_plan":{"days":[{"day":1,"title":"...","items":['
        '{"time":"09:00","name":"...","venue":"...","venue_from":"","duration_min":90,'
        '"cost_eur":0,"note":"..."}]}],"total_cost_eur":0}}\n'
        'IMPORTANT — "venue" is the exact, searchable proper name of the place, and nothing '
        'else: "Charles Bridge", "Karlštejn Castle", "Lokál Dlouhá". No descriptions, no '
        'adjectives, no "or similar". If the item is not a specific place — a transfer, hotel '
        'check-in, rest, free time, or a generic suggestion with no venue chosen — set venue '
        'to "". A wrong venue is worse than an empty one, because it maps to the wrong place.\n'
        "Call a tool only when it adds real information. weather_tool returns LIVE data. "
        "booking_tool/flight_search_tool are fictive. Costs and the budget are PER PERSON for the whole "
        "trip; estimate cost_eur per person. You have about %d tool calls: spend them on facts "
        "that change the plan, then finalize. Do not re-check something you already know.\n\n"
        "PLAN RULES — the critic checks these, so build them in rather than leaving them out:\n"
        "- If the profile has start_time/start_point, DAY 1 BEGINS THERE AND THEN. Do not invent "
        "an earlier start. If it has end_time/end_point, the last day must arrive there with a "
        "sensible buffer (2-3h before a flight) and nothing may be scheduled after it.\n"
        "- If lodging is named, use that exact hotel for check-in/out and treat it as the base "
        "for each day's first and last legs.\n"
        "- Every entry in details is a real constraint or wish the traveller stated. Honour each "
        "one in the plan, or say in a note why it could not be met.\n"
        "- Keep the PER-PERSON total at or under €%d. If the profile names an amount, that is a "
        "figure the traveller gave you, not a suggestion.\n"
        "- Match the walking level: \"light\" means short distances and transit between stops; "
        "\"high\" or \"unlimited\" means walking routes are welcome and stops can be further "
        "apart. Do not schedule a walking-heavy day for someone who asked for light.\n"
        "- For MEALS, venue is a NEIGHBOURHOOD OR STREET, never a named restaurant: \"Malá "
        "Strana\", \"Nerudova\", \"Vinohrady\". Put the useful part in the note — cuisine, price "
        'level, and where to look: "traditional Czech, mid-range, plenty of options around '
        'Nerudova". The same goes for free time: name the area, say what is there in the note.\n'
        "- You MAY name specific places in the NOTE if reviews_tool actually returned them "
        "this run — \"reviewers rate Lokál Dlouhá and Kantýna near here\". Never name one from "
        "memory: an unchecked restaurant may have closed or been renamed, and putting it in "
        "the plan presents a guess as a fact.\n"
        "- When live prices are supplied above, cost every item from them. They are current and "
        "local; your own recollection of what things cost is neither.\n"
        "- Meal costs are PER PERSON and include a drink and service. Mid-range dinner in a "
        "European city is EUR25-30, lunch EUR15-20, coffee EUR4-6 — do not quote EUR10 for a "
        "sit-down dinner.\n"
        "- Use the forecast if one is given: put outdoor and walking-heavy days on dry ones, "
        "indoor alternatives on wet ones, and say why in the note.\n"
        '- A transfer is a JOURNEY, so give it both ends: "venue_from" is where it starts and '
        '"venue" is where it ends ("Prague Bank Hotel" -> "Old Town Square"). Never write the '
        'word "transfer" in either. Leave venue_from empty for anything that is not a journey.\n'
        "- EVERY item that happens somewhere needs a venue, including free time and flexible "
        'blocks. If an item offers alternatives ("Náplavka or shopping"), CHOOSE ONE as the '
        "venue and mention the alternative in the note — otherwise that stop is missing from "
        "the day's route and the traveller gets a broken chain.\n"
        "- Account for travel between locations. Either add a short transfer item, or start the "
        "next item late enough to absorb it and say so in the note. Never schedule two places "
        "back to back as though they were adjacent.\n"
        "- On any day with more than about 6 hours of activity, include at least one rest, coffee "
        "or downtime item. Queueing and standing count as effort.\n"
        "- If an item normally needs advance booking or a timed entry (major museums, popular "
        'restaurants, guided experiences), begin its note with "Book ahead:".\n'
        "- Keep days balanced. Do not stack several multi-hour, high-queue sites on one day while "
        "another day is nearly empty.\n"
        "- reviews_tool is real web search: \"best trattorias near Monti Rome\" comes back with "
        "summarised opinion and source links. One call per trip is usually enough to make the "
        "meal notes concrete.\n"
        "- Tool calls cost real time. Look things up only where the answer would CHANGE the "
        "plan, and never for a place you already know. Two or three lookups is normal for a "
        "trip; ten is not, and a run that spends its time researching returns no itinerary.\n"
        "- Verify opening hours with maps_tool for the few timed sites where it matters, and never "
        "schedule a site before it usually opens. If open_hours comes back null the hours are "
        "unknown: write the note from the TRAVELLER'S point of view — \"check opening hours and "
        "tickets before you go\". NEVER mention tools, lookups or what the agent did or did not "
        "find; the traveller has no idea what maps_tool is and does not need to."
        % (MAX_PLANNER_STEPS, schemas.budget_ceiling_eur(prof))
    )
    user = "Traveller profile:\n" + json.dumps(schemas.compact_profile(prof))
    if prices:
        user += ("\n\nCurrent local prices, from a live search. Cost the itinerary from these "
                 "rather than from memory:\n" + json.dumps(prices))
    if forecast:
        # Handed over rather than left for the planner to fetch: it is the same data, minus
        # an LLM round trip spent deciding to ask for it.
        user += ("\n\nForecast for the destination (use it — put outdoor days on the dry ones):\n"
                 + json.dumps(forecast))
    if feedback:
        user += "\n\nCritic feedback to fix:\n" + json.dumps(feedback)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    seen_calls = set()  # repetition guard: (tool, canonical tool_input)
    tool_seconds = [0.0]
    failures = [0]         # consecutive failed lookups; see MAX_TOOL_FAILURES
    venues_seen = []       # real places found this run, so an abort has something to salvage   # list so the loop body can mutate it; see MAX_TOOL_SECONDS

    for _ in range(MAX_PLANNER_STEPS):
        if _expired(deadline):
            # Hard gate. The previous `break` fell through to the forced finalize below,
            # which spent another LLM call after the time budget was already gone.
            return _abort_plan(prof, steps, run_id, venues=list(venues_seen))
        # Leave room to write the plan. Without this the loop could research right up to the
        # deadline and then have no budget left to produce an itinerary at all.
        if _low_on_time(deadline, PLANNER_FINALIZE_RESERVE_S):
            obs.log("planner_reserve_hit", run_id=run_id)
            break
        # 3000, not 1100. For a reasoning model this becomes a 3000 + llm._REASONING_HEADROOM
        # completion cap, and roughly 2000 of that goes on hidden reasoning. A 7-day plan is
        # ~2000 tokens of JSON, so a smaller budget truncates it mid-object on the finalize
        # attempt: unparseable -> repair retry -> ~36s burned -> repeat until the deadline.
        # This is a ceiling, not a target — tool-call turns still emit tiny JSON.
        try:
            turn = _chat_json(msgs, temperature=0.3, max_tokens=3000)
        except Exception as e:
            # A slow call must not discard the whole turn. Narrow on purpose: only a timeout
            # degrades — auth, config and rejected-parameter errors still fail loudly.
            if not is_timeout(e):
                raise
            obs.log("planner_timeout", run_id=run_id, where="loop")
            return _abort_plan(prof, steps, run_id, reason="a planner call timed out",
                               venues=list(venues_seen))
        kind = schemas.classify_turn(turn)

        if kind[0] == "done":
            plan = schemas.validate_draft_plan(kind[1]) or schemas.minimal_plan(prof)
            _trace(steps, "ReAct Planner",
                   {"thought": (turn or {}).get("thought"), "action": "finalize"},
                   {"draft_plan": plan})
            obs.log("planned", run_id=run_id, forced=False, cost_eur=plan.get("total_cost_eur"))
            return plan

        if kind[0] == "tool":
            _, tool, tool_input = kind
            if _expired(deadline):        # a slow tool may have consumed the rest of the run
                break
            key = (tool, json.dumps(tool_input, sort_keys=True, default=str))
            if key in seen_calls:
                observation = {"ok": False, "note": "Repeated identical call ignored. "
                                                     "Choose a different tool/input or finalize with a draft_plan."}
            elif tool_seconds[0] >= MAX_TOOL_SECONDS:
                # Out of network budget. Refusing here keeps the remaining time for writing
                # the itinerary; spending it on one more lookup buys a detail and loses the
                # whole plan.
                observation = {"ok": False, "note": "Research budget for this run is used up. "
                                                    "Finalize now with a draft_plan using what "
                                                    "you already know."}
                obs.log("tool_budget_exhausted", run_id=run_id, tool=tool)
            else:
                seen_calls.add(key)
                started = time.monotonic()
                observation = run_tool(tool, _localise(tool, tool_input, prof))
                tool_seconds[0] += time.monotonic() - started

            # Lookups that keep failing are a dead end, and every retry is a whole LLM turn.
            # After a couple, the plan is better served by writing it from what we know than
            # by spending the rest of the run on a tool that will not answer.
            if isinstance(observation, dict) and observation.get("ok") is False:
                failures[0] += 1
                if failures[0] >= MAX_TOOL_FAILURES:
                    observation = {**observation,
                                   "note": (observation.get("note", "") + " Several lookups "
                                            "have failed — stop researching and return a "
                                            "draft_plan built on what you already know.")}
            else:
                failures[0] = 0
                _remember_hours(hours_seen, observation, coords_seen)
                for result in (observation.get("results") or []
                               if isinstance(observation, dict) else []):
                    if isinstance(result, dict) and result.get("name"):
                        venues_seen.append(result["name"])
            _trace(steps, "ReAct Planner",
                   {"thought": (turn or {}).get("thought"), "tool": tool, "tool_input": tool_input},
                   {"observation": observation})
            obs.log("tool", run_id=run_id, tool=tool, ok=bool(observation.get("ok")))
        else:  # invalid turn — nudge without crashing; still bounded by the loop
            observation = {"ok": False, "note": "Your last message was not a valid action. Return a tool "
                                                'call or {"done":true,"draft_plan":{...}} as a JSON object.'}
            _trace(steps, "ReAct Planner", {"thought": None, "action": "invalid"}, {"observation": observation})

        # Keep context lean: assistant turn plus the trimmed observation only.
        obs_json = json.dumps(observation)[:MAX_OBS_CHARS]
        msgs.append({"role": "assistant", "content": json.dumps(turn) if turn is not None else "{}"})
        msgs.append({"role": "user", "content": "Observation: " + obs_json + "\nContinue."})

    # Safety net: force a finalize and always return a valid plan (never None).
    # Guarded too, because the loop can fall through here with the budget already spent.
    if _expired(deadline):
        return _abort_plan(prof, steps, run_id, venues=list(venues_seen))
    msgs.append({"role": "user",
                 "content": 'Stop now. Return ONLY {"thought":"...","done":true,"draft_plan":{...}}.'})
    try:
        turn = _chat_json(msgs, temperature=0.2, max_tokens=3000)   # see the loop call above
    except Exception as e:
        if not is_timeout(e):
            raise
        obs.log("planner_timeout", run_id=run_id, where="finalize")
        return _abort_plan(prof, steps, run_id, reason="the final planning call timed out",
                           venues=list(venues_seen))
    draft = (turn or {}).get("draft_plan")
    plan = schemas.validate_draft_plan(draft) or schemas.minimal_plan(prof)
    _trace(steps, "ReAct Planner",
           {"thought": (turn or {}).get("thought", "forced finalize"), "action": "finalize"},
           {"draft_plan": plan})
    obs.log("planned", run_id=run_id, forced=True, degraded=bool(plan.get("degraded")),
            cost_eur=plan.get("total_cost_eur"))
    return plan


# ---------- Module: Plan Editor ----------
def _apply_patch(plan, patch):
    """Merge day-scoped edits into a plan. Returns (new_plan, changed_day_numbers).

    Days the patch does not name come through byte-identical — the model never gets the
    chance to quietly reword an untouched day. `total_cost_eur` is recomputed from the
    merged items, so the total stays consistent with what is displayed.
    """
    base = schemas.validate_draft_plan(plan) or {"days": [], "total_cost_eur": 0}
    by_day = {d["day"]: d for d in base["days"]}
    changed = []
    for raw in schemas._as_list(schemas.as_obj(patch).get("days")):
        d = schemas.as_obj(raw)
        num = schemas._as_int(d.get("day"), 0)
        if num < 1:
            continue                      # a patch day with no number cannot be placed
        one = schemas.validate_draft_plan({"days": [d]})
        if not one:
            continue                      # no usable items; leave the original day alone
        merged = dict(one["days"][0])
        merged["day"] = num               # index-based numbering must not clobber the target
        by_day[num] = merged
        changed.append(num)
    days = [by_day[k] for k in sorted(by_day)]
    total = sum(i["cost_eur"] for d in days for i in d["items"])
    return {"days": days, "total_cost_eur": total}, sorted(changed)


def _edit_plan(prof, plan, conversation, steps, run_id=None):
    """Revise an existing itinerary in one call, instead of re-running the whole planner.

    The model returns ONLY the days it changes. That keeps the completion small, removes
    any chance of a 35-item plan being silently truncated on the way back, and guarantees
    untouched days are preserved exactly rather than approximately.
    """
    usage.set_module("Plan Editor")
    sys = (
        "You are the Plan Editor. You are given an existing trip itinerary as JSON and the "
        "traveller's requested change. Apply ONLY that change.\n"
        'Return ONLY JSON: {"thought":"...","days":[{"day":2,"title":"...","items":['
        '{"time":"09:00","name":"...","duration_min":90,"cost_eur":0,"note":"..."}]}],'
        '"summary":"one line describing what you changed"}\n'
        "RULES:\n"
        "- Include ONLY the days you are changing. Days you omit are kept exactly as they are.\n"
        "- Return each changed day COMPLETE, with all of its items — not just the edited item.\n"
        "- Keep the same day numbers. Do not renumber, add, or drop days.\n"
        "- Costs are per person in EUR for the whole trip.\n"
        "- If the request cannot be satisfied, return an empty days list and explain in summary."
    )
    user = ("Traveller profile:\n" + json.dumps(schemas.compact_profile(prof)) +
            "\n\nCurrent itinerary:\n" + json.dumps(plan) +
            "\n\nConversation (the requested change is the LAST user message):\n" + conversation)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    turn = schemas.as_obj(_chat_json(msgs, temperature=0.3, max_tokens=1100))
    new_plan, changed = _apply_patch(plan, turn)
    summary = schemas._as_str(turn.get("summary"))

    _trace(steps, "Plan Editor",
           {"thought": turn.get("thought"), "current_total_eur": plan.get("total_cost_eur"),
            "requested_change": "see conversation"},
           {"changed_days": changed, "summary": summary,
            "new_total_eur": new_plan.get("total_cost_eur")})
    obs.log("plan_edited", run_id=run_id, changed_days=changed,
            cost_eur=new_plan.get("total_cost_eur"))
    return new_plan, changed, summary


# ---------- Module: Itinerary Q&A ----------
def _answer_question(prof, plan, conversation, steps, run_id=None):
    """Answer a question about the delivered itinerary without touching it.

    One call, no planner, no formatter, and no risk of mutating a plan the user only asked
    about. Routing these here rather than through the editor is both cheaper and safer.
    """
    usage.set_module("Itinerary Q&A")
    sys = (
        "You answer questions about a trip itinerary that has already been produced. Use ONLY "
        "the itinerary and profile given — do not invent places, prices, or times. Costs are "
        "per person in EUR. Answer in one or two short sentences, in Markdown. If the answer "
        "is not in the itinerary, say so plainly and offer to revise the plan."
    )
    user = ("Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
            "\n\nItinerary:\n" + json.dumps(plan) +
            "\n\nConversation (the question is the LAST user message):\n" + conversation)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = chat(msgs, temperature=0.2, max_tokens=400) or ""
    _trace(steps, "Itinerary Q&A", {"plan_total_eur": plan.get("total_cost_eur")},
           {"answer": text})
    obs.log("question_answered", run_id=run_id, chars=len(text))
    return text


# ---------- Module: Reflection Layer ----------
def _reflect(prof, draft, steps, run_id=None, found=None):
    usage.set_module("Reflection Layer")
    sys = (
        "You are the Reflection Layer (critic). Check the draft itinerary against the profile for: "
        "geographic logic, time feasibility, budget, rest breaks, opening hours, and balance.\n"
        "CONTEXT you must assume:\n"
        "- Every cost_eur is PER PERSON for the whole trip, and total_cost_eur is the per-person "
        "sum. Never raise this as an ambiguity.\n"
        "- The traveller has given no exact travel dates unless the profile contains a `when` "
        "field. Without dates, anything date-dependent — weekly closing days, seasonal hours, "
        "religious services, festivals — is ADVISORY, not a defect.\n"
        "- The planner cannot verify live ticket prices or real-time opening hours. Price drift "
        "and booking availability are advisory too.\n"
        'Return ONLY JSON: {"verdict":"PASS"|"FAIL","must_fix":[...],"be_aware":[...],"fixes":[...]}\n'
        "- must_fix: concrete defects the planner could and should correct — impossible timing, a "
        "site scheduled before it opens, badly unbalanced days, geography that does not work, a "
        "total that breaks the stated budget.\n"
        "- be_aware: real but not fixable from here — booking advice, dress codes, accessibility, "
        "weather, price drift, date-dependent risks. Keep this list SHORT — only what genuinely "
        "changes what the traveller should do. Each one a single bullet of about 12 words, "
        'phrased as direct advice ("Book castle tours ahead — summer slots sell out"). No '
        "preamble, no restating the itinerary, no explaining why you cannot verify it.\n"
        '- verdict is "FAIL" only when must_fix is non-empty. be_aware items alone are a PASS.'
    )
    user = ("Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
            "\n\nDraft:\n" + json.dumps(draft))
    if found:
        # Arithmetic and looked-up facts, computed deterministically. Handing these over
        # means the critic does not have to re-derive clock sums it checks unreliably, and
        # can spend its attention on judgement instead.
        user += ("\n\nAlready VERIFIED defects — these are computed, not guesses. Include "
                 "every one in must_fix verbatim, then add anything else you find:\n"
                 + json.dumps(found))
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": user}]
    # 600 was too small for a reasoning model reviewing a whole plan: it spent the entire
    # completion budget (600 + llm._REASONING_HEADROOM) on hidden reasoning and returned
    # nothing parseable, so the repair retry fired on every run — doubling the cost and
    # adding ~50s, and sometimes still failing, which produced a FAIL with no issues.
    #
    # A timeout here must not discard a finished plan either. validate_verdict(None) yields
    # FAIL with no issues, which _run_turn already converts into an honest caveat.
    try:
        raw = _chat_json(msgs, temperature=0.2, max_tokens=1400)
    except Exception as e:
        if not is_timeout(e):
            raise
        obs.log("reflect_timeout", run_id=run_id)
        raw = None
    verdict = schemas.validate_verdict(raw)
    # The critic may drop or reword them; these are facts, so they are re-asserted here.
    for defect in (found or []):
        if defect not in verdict["must_fix"]:
            verdict["must_fix"].append(defect)
    if verdict["must_fix"]:
        verdict["verdict"] = "FAIL"
    obs.log("reflected", run_id=run_id, verdict=verdict["verdict"],
            must_fix=len(verdict["must_fix"]), be_aware=len(verdict["be_aware"]))
    _trace(steps, "Reflection Layer", {"profile": prof, "draft": draft}, verdict)
    return verdict


# ---------- Module: Output Formatter ----------
# Control characters have no place in a venue name, but models do emit them — a raw U+001F
# turned up where an em-dash belonged. Left in, quote_plus faithfully encodes it as %1F and
# the map link stops resolving.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _map_query(venue, city=""):
    """Build a Maps query from a venue name, qualified by the city.

    The city is appended only when the venue does not already name it — planners routinely
    return "Václav Havel Airport Prague", and blindly appending gave "... Prague Prague",
    which reads as a typo in a user-visible URL.
    """
    venue = " ".join(_CONTROL_CHARS.sub(" ", venue or "").split())
    city = " ".join(_CONTROL_CHARS.sub(" ", city or "").split())
    if city and city.casefold() not in venue.casefold():
        venue = f"{venue} {city}"
    return urllib.parse.quote_plus(venue)


# The Google Maps URL API accepts at most 9 waypoints between origin and destination.
_MAX_WAYPOINTS = 9


def _venue_of(item):
    """The item's proper place name, or "" when it is not a specific place.

    Links are built from `venue` alone, never from `name`. `name` is a human label
    ("Popular spot — stroll and photos"); putting that in a Maps URL yielded a keyword
    search that landed on no particular place. An item with no venue simply gets no link,
    which is more useful than a link to the wrong thing.
    """
    return _CONTROL_CHARS.sub(" ", (item.get("venue") or "").strip())


def _place_url(venue, destination):
    """A Maps link that resolves to one place.

    Uses the place search endpoint with the proper venue name plus the city, which Maps
    resolves to a single pin rather than a list of keyword matches.
    """
    return "https://www.google.com/maps/search/?api=1&query=" + _map_query(venue, destination)


def _leg_url(origin, destination, city):
    """Directions between two consecutive stops, for a transfer item.

    A transfer's own pin is useless — what the traveller needs is the way there from where
    they just were. Built only when both ends are known venues.
    """
    if not origin or not destination:
        return None
    return ("https://www.google.com/maps/dir/?api=1"
            f"&origin={_map_query(origin, city)}&destination={_map_query(destination, city)}")


# Items whose value is the journey rather than the place: they get a directions link from
# where the traveller was, instead of a pin on where they end up.
#
# Narrowed deliberately. "back to", "return to" and "depart" also describe ordinary stops
# ("head back to the square for dinner"), so they turned sightseeing into routes. What is
# left names a mode of travel or an explicit transfer.
_TRANSFER_HINTS = ("transfer", "travel to", "walk to", "tram to", "metro to", "train to",
                   "bus to", "taxi to", "drive to", "ride to", "transit to", "journey to",
                   "to the airport", "to airport", "to the station", "to the hotel")


def _is_transfer(name):
    n = (name or "").strip().lower()
    return any(h in n for h in _TRANSFER_HINTS)


def _with_map_links(plan, destination, coords=None):
    """Attach real, deterministically built links to the plan.

    Only two kinds now, both keyed off the planner's `venue` field:
      * map_url   — per item: a pin on the venue, or for a transfer, directions from the
                    previous stop to this one (`is_leg` marks which)
    A per-day route link used to be emitted too. It was dropped: every stop already links
    to itself and every transfer links to its own directions, so a third kind of link at day
    level restated what the bullets above it already said.

    An earlier version also emitted `site_url` for sights. It was a Google *search* URL
    labelled "official site", which is not an official site — so it is gone rather than
    mislabelled. Real official URLs need a Places API; guessing domains would invite
    hallucinated links.
    """
    out = {**plan, "days": []}
    for day in plan.get("days", []):
        items = []
        previous = ""          # the last venue the traveller was at, for transfer legs
        for it in day.get("items", []):
            new_it = {k: v for k, v in it.items() if k not in ("site_url", "route_url")}
            venue = _venue_of(it)
            # Backstop for the meal rule above. If the planner named a restaurant anyway, we
            # cannot verify it exists, so it gets no pin — a map link is an assertion that
            # the place is real and findable, and here we do not know that.
            # A map pin says "go here". It is wrong on two kinds of item: a meal named after
            # a restaurant we cannot verify exists, and an open block where the traveller
            # picks the destination themselves.
            name = it.get("name")
            if venue and audit.is_meal(name) and audit.looks_like_a_business(venue):
                venue = ""
            elif venue and audit.is_flexible(name):
                venue = ""
            if venue:
                # A journey links to directions, not to a pin on its destination. The start
                # is whatever the planner declared, falling back to where we actually were.
                start = _CONTROL_CHARS.sub(" ", (it.get("venue_from") or "").strip()) or previous
                # Both signals are required. The planner sets venue_from on ordinary stops
                # too, so on its own it would turn museum visits into driving routes.
                leg = (_leg_url(start, venue, destination)
                       if (it.get("venue_from") and _is_transfer(it.get("name"))) else None)
                new_it["map_url"] = leg or _place_url(venue, destination)
                new_it["is_leg"] = bool(leg)
                previous = venue
            items.append(new_it)
        new_day = {**day, "items": items}
        out["days"].append(new_day)
    return out


def _format_fallback(prof, plan, coords=None):
    """Render the plan as Markdown without an LLM.

    Used when the time budget is spent before the Output Formatter can run, so a turn
    always returns the itinerary it actually built instead of nothing. Deterministic,
    costs no tokens, and reuses the same real map links as the LLM path.
    """
    plan = _with_map_links(plan, prof.get("destination", ""), coords)
    dest = prof.get("destination") or "your destination"
    out = [f"# {prof.get('days', '')}-day itinerary — {dest}".strip(), ""]
    for day in plan.get("days", []):
        items = day.get("items", [])
        title = f" — {day['title']}" if day.get("title") else ""
        out.append(f"## Day {day.get('day', '')}{title}")
        for it in items:
            name = it.get("name") or "Activity"
            link = f"[{name}]({it['map_url']})" if it.get("map_url") else name
            time_part = f"**{it['time']}** — " if it.get("time") else ""
            out.append(f"- {time_part}{link}")
            if it.get("note"):
                out.append(f"  - {it['note']}")
            # The name already links to the directions when this is a leg; repeating the URL
            # on the detail line would be the same link twice on one bullet.
            out.append(f"  - {it.get('duration_min', 0)} min · €{it.get('cost_eur', 0)} per person")
            out.append("")          # blank line between items keeps the times scannable
        out += ["", f"_Day total: €{sum(i.get('cost_eur', 0) for i in items)}_"]
        out.append("")
    out.append(f"**Total: €{plan.get('total_cost_eur', 0)} per person**")
    return "\n".join(out)


# Markdown needs a blank line between list items for them to render as separate blocks. The
# formatter is asked for one, but a model that runs the bullets together produces a wall of
# text, so the spacing is guaranteed here instead of hoped for.
_TOP_BULLET = re.compile(r"(?m)^(?=[-*] )")


def _space_bullets(text):
    """Ensure one blank line before every top-level bullet, without touching sub-lines."""
    out, previous = [], ""
    for line in (text or "").splitlines():
        is_top = bool(_TOP_BULLET.match(line))
        # A sub-line under the previous bullet is indented; it stays attached to it.
        if is_top and previous.strip() and not previous.startswith("#"):
            out.append("")
        out.append(line)
        previous = line
    return "\n".join(out)


def _format(prof, plan, steps, run_id=None, warnings=None, coords=None, deadline=None):
    """Render the plan via the LLM, degrading to the deterministic renderer if it cannot.

    Two ways this call fails to produce text, and both were previously shipped straight to
    the user as an empty itinerary — a caveats banner with nothing underneath:

      * it times out (this is the last and slowest call in the pipeline), or
      * a reasoning model spends its entire completion budget thinking and returns "".

    Either way the plan itself is intact, so falling back to _format_fallback always beats
    returning nothing.
    """
    usage.set_module("Output Formatter")
    linked = _with_map_links(plan, prof.get("destination", ""), coords)
    sys = (
        "You are the Output Formatter. Turn the validated plan into a clear, friendly day-by-day "
        "itinerary in Markdown. For each item show time, name, a one-line tip, duration and cost. "
        "Links: use ONLY the URLs present in the plan and never invent one, and never write a "
        "bare URL as text. An item may have a map_url — render its name as a Markdown link to "
        "it; if it has no map_url, leave the name as plain text. ALWAYS use the item's own name "
        "as the link text, including when is_leg is true — \"Walk to Old Town Square\", never the "
        "word \"directions\". A link labelled by what it does instead of where it goes tells the "
        "reader nothing.\n"
        "LAYOUT — the itinerary is scanned, not read:\n"
        "- One bullet per item, and put a blank line between bullets so the times separate.\n"
        "- Start each bullet with the time in bold, then the name: **09:00** — [Name](url)\n"
        "- The tip, duration and cost go on indented sub-lines under that bullet. Write the tip "
        "as a plain sentence with NO \"Tip:\" label — the reader can see it is a tip.\n"
        "Costs are per person. End with a per-day and grand total cost. Be concise."
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
                                        "\n\nPlan:\n" + json.dumps(linked)}]
    # Writing a whole itinerary needs most of a full timeout. Starting the call with less
    # than that spends the remaining time and returns nothing, so below the threshold the
    # deterministic renderer is used directly.
    text = ""
    if _low_on_time(deadline, FORMAT_MIN_SECONDS):
        obs.log("format_skipped_low_time", run_id=run_id)
    else:
        try:
            text = chat(msgs, temperature=0.4, max_tokens=3000) or ""
        except Exception as e:
            if not is_timeout(e):
                raise
            obs.log("format_timeout", run_id=run_id)

    if not text.strip():
        text = _format_fallback(prof, plan, coords)
        _trace(steps, "Output Formatter", {"plan": linked},
               {"itinerary_markdown": text, "fallback": True})
        obs.log("format_empty_fallback", run_id=run_id)
        return text

    text = _space_bullets(text)
    _trace(steps, "Output Formatter", {"plan": linked}, {"itinerary_markdown": text})
    return text


def _safe_geocode(name):
    """Geocode gate: returns a hit or None; never raises (the network call is best-effort)."""
    try:
        return geocode_place(name)
    except Exception:
        return None


# ---------- Pipeline ----------
class AgentError(RuntimeError):
    """Wraps a failed turn together with the steps recorded before it failed, so the
    partial trace is not lost with the exception. Carries the usage snapshot too, so a
    failed turn's spend still reaches the budget ledger."""

    def __init__(self, message, steps, usage_snapshot=None, ms=None):
        super().__init__(message)
        self.steps = steps
        self.usage = usage_snapshot or {}
        self.ms = ms


def run_agent(user_prompt, state=None):
    """Run one turn, preserving the trace and the spend on failure.

    `state` is the prior {profile, plan} for this conversation, loaded from Supabase by the
    API layer. Passing None simply means no itinerary exists yet, which is exactly how the
    agent behaved before persistence existed.

    Returns {"response", "steps", "state", "branch", "usage", "ms"}. The step list is owned
    here rather than inside the turn, so when something raises mid-run the steps taken so
    far travel out with the error and the UI can show where it stopped.
    """
    steps = []
    meter = usage.start_run()
    t0 = time.monotonic()
    try:
        out = _run_turn(user_prompt, steps, state)
        out["usage"] = meter.snapshot()
        out["ms"] = int((time.monotonic() - t0) * 1000)
        obs.log("run_usage", branch=out.get("branch"), **meter.snapshot())
        return out
    except AgentError:
        raise
    except Exception as e:
        raise AgentError(str(e), steps, meter.snapshot(),
                         int((time.monotonic() - t0) * 1000)) from e
    finally:
        usage.end_run()


def _with_caveats(response, warnings, notes=None):
    """Prefix the user-visible banners.

    Two distinct things, deliberately shown separately: `warnings` are defects we could not
    fix or verify, and `notes` are advisory travel tips. Merging them made a competent plan
    look broken — a wall of equally-alarming bullets reads as "this itinerary is wrong" when
    most of them are "book ahead" and "cover your shoulders at St. Peter's".
    """
    warn = list(dict.fromkeys(w for w in warnings if w))
    note = list(dict.fromkeys(n for n in (notes or []) if n))
    head = ""
    if warn:
        head += ("> ⚠️ **Delivered with caveats** — this plan was not fully validated:\n"
                 + "\n".join(f"> - {w}" for w in warn) + "\n\n")
    if note:
        head += ("> ℹ️ **Good to know before you go**:\n"
                 + "\n".join(f"> - {n}" for n in note) + "\n\n")
    return head + response


def _deliver(prof, plan, steps, run_id, deadline, warnings, notes=None, coords=None):
    """Shared tail for the branches that hand back an itinerary.

    Runs the deterministic budget guard, then formats — via the LLM when there is time
    left, deterministically when there is not, so a turn always returns the plan it built.
    """
    ceiling = schemas.budget_ceiling_eur(prof)
    total = plan.get("total_cost_eur", 0)
    if total > ceiling:
        basis = ("your stated budget" if prof.get("budget_amount_eur")
                 else f"the guide for a {prof['budget']} {prof['days']}-day trip")
        warnings.append(f"Estimated cost €{total} per person exceeds ~€{ceiling} — {basis}.")
    if plan.get("timed_out"):
        warnings.append("The agent ran out of time before it could build a full itinerary; "
                        "this is a skeleton. Try a shorter trip, or send the request again.")
    elif plan.get("degraded"):
        warnings.append("The planner could not fully build this itinerary; some days are placeholders.")

    if _expired(deadline):
        response = _format_fallback(prof, plan, coords)
        _trace(steps, "Output Formatter", {"plan": plan},
               {"itinerary_markdown": response, "fallback": True})
        # Not surfaced to the traveller: the itinerary is the same itinerary either way, and
        # only the prose differs. Calling that a caveat makes a sound plan look defective.
        obs.log("format_fallback", run_id=run_id)
    else:
        response = _format(prof, plan, steps, run_id, warnings, coords, deadline)
    return _with_caveats(response, warnings, notes)


def _run_turn(user_prompt, steps, state=None):
    """Run one turn. `user_prompt` is the conversation transcript; `state` is the prior
    {profile, plan}. Returns {"response", "steps", "state", "branch"} for every branch."""
    run_id = obs.new_run_id()
    deadline = time.monotonic() + MAX_RUN_SECONDS
    # No LLM call may run past this, whenever it starts.
    llm_set_wall(time.monotonic() + HARD_WALL_SECONDS)
    conversation = user_prompt or ""

    prior = schemas.as_obj(state)
    prior_plan = schemas.validate_draft_plan(prior.get("plan"))
    prior_profile = schemas.validate_profile(prior.get("profile")) if prior.get("profile") else None

    obs.log("run_start", run_id=run_id, chars=len(conversation), has_plan=bool(prior_plan))

    decision = _profile(conversation, has_plan=bool(prior_plan))
    missing = decision["missing"]

    # ---- Branch A: required info still missing -> ask, stop (no planner, no token waste). ----
    if missing:
        question = decision["question"] or _fallback_question(missing)
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "clarify", "missing": missing, "reply": question})
        obs.log("intake_clarify", run_id=run_id, missing=missing)
        return {"response": question, "steps": steps, "state": prior, "branch": "clarify"}

    prof = schemas.validate_profile(decision["profile"])
    intent, confirmed = decision["intent"], decision["confirmed"]

    # Deterministic override of the model's intent: changing a required field is a new trip,
    # not an edit. Force re-confirmation so an existing itinerary is never silently replaced.
    replacing = bool(prior_plan and prior_profile and _required_changed(prior_profile, prof))
    if replacing:
        intent, confirmed = "replace", False

    # ---- Branch D: a question about the existing plan -> answer it, change nothing. ----
    if prior_plan and intent == "question":
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "question", "intent": intent, "profile": prof})
        base = prior_profile or prof
        answer = _answer_question(base, prior_plan, conversation, steps, run_id)
        obs.log("run_end", run_id=run_id, branch="question", steps=len(steps))
        return {"response": answer or "I could not answer that from the itinerary.",
                "steps": steps, "state": {"profile": base, "plan": prior_plan},
                "branch": "question"}

    # ---- Branch E: revise the existing plan -> patch it, do not re-plan from scratch. ----
    if prior_plan and intent == "revise":
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "revise", "intent": intent, "profile": prof})
        base = prior_profile or prof
        warnings = []
        if _expired(deadline):
            # No budget to edit; hand back what exists rather than nothing.
            return {"response": _with_caveats(
                        _format_fallback(base, prior_plan),
                        ["Ran out of time before the change could be applied."]),
                    "steps": steps, "state": {"profile": base, "plan": prior_plan},
                    "branch": "revise"}

        new_plan, changed, summary = _edit_plan(base, prior_plan, conversation, steps, run_id)
        if not changed:
            warnings.append("I could not apply that change"
                            + (f" — {summary}" if summary else ", so the plan is unchanged."))
            new_plan = prior_plan
        # No critic pass here: a revision is a bounded, user-requested edit and the
        # deterministic budget guard below still runs. That keeps this branch at 3 calls.
        response = _deliver(base, new_plan, steps, run_id, deadline, warnings,
                            coords=_safe_geocode(base.get('destination', '')))
        obs.log("run_end", run_id=run_id, branch="revise", changed_days=changed,
                steps=len(steps))
        return {"response": response, "steps": steps,
                "state": {"profile": base, "plan": new_plan}, "branch": "revise"}

    # ---- Branch B: complete but not confirmed -> show profile, ask to confirm, stop. ----
    # Still one LLM call so far; we do not plan against an unconfirmed profile.
    if not confirmed:
        reply = _confirmation_message(prof)
        if replacing:
            reply = ("That looks like a different trip from the one I already planned, so let me "
                     "check before I replace it.\n\n") + reply
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "confirm", "profile": prof, "reply": reply})
        obs.log("intake_confirm", run_id=run_id, replacing=replacing)
        # The existing plan is kept in state: if the user backs out, it is not lost.
        return {"response": reply, "steps": steps,
                "state": {"profile": prof, "plan": prior_plan}, "branch": "confirm"}

    # ---- Branch C: user confirmed -> only now spend tokens on the heavy loops. ----
    _trace(steps, "Conversational Intake", {"conversation": conversation},
           {"stage": "plan", "profile": prof})
    obs.log("intake_confirmed", run_id=run_id, destination=prof["destination"], days=prof["days"])

    warnings = []
    # Already needed to validate the destination; keeping the hit gives the itinerary a real
    # centred map link for free, with no extra network call.
    coords = _safe_geocode(prof["destination"])
    # Filled in by maps_tool observations during planning; consumed by the hours check and
    # the travel-time pass, so neither has to look up what we already fetched.
    hours_seen, coords_seen = {}, {}
    if coords is None:
        warnings.append(f'Could not locate "{prof["destination"]}" — it may be invalid or the '
                        "itinerary may be generic.")

    forecast, prices = _forecast_for(prof, coords), _prices_for(prof)
    draft = _plan(prof, steps, run_id=run_id, deadline=deadline,
                  forecast=forecast, prices=prices, hours_seen=hours_seen,
                  coords_seen=coords_seen)
    notes = []          # advisory travel tips; shown apart from the defect warnings
    for c in range(MAX_REFLECT_CYCLES):
        if _expired(deadline):
            break  # no budget left to review the draft; ship what we have
        # Costs nothing and never fails: arithmetic over the draft plus hours we already
        # looked up. Runs before the critic so its findings can be handed over as fact.
        if c == 0:
            # Once per turn, on the first draft: measured travel times before anything is
            # checked against them, so the schedule audit judges real durations.
            draft, legs_fixed = _apply_travel_times(draft, prof, coords_seen, run_id)
            if legs_fixed:
                notes.append(f"Travel times for {legs_fixed} legs are measured, not estimated.")
        draft, money_fixes = audit.apply_food_floors(draft, prof.get("budget"))
        if money_fixes:
            obs.log("food_floors_applied", run_id=run_id, count=len(money_fixes))
        found = audit.check_schedule(draft, prof) + \
            audit.check_opening_hours(draft, hours_seen)
        if _low_on_time(deadline, DELIVER_RESERVE_S):
            break   # only enough left to write up what we already have
        # The review is optional; writing the plan up is not. With little time left, skipping
        # it leaves the formatter enough to work with — a good plan reviewed by nobody beats
        # a reviewed plan nobody can read.
        if _low_on_time(deadline, REVIEW_MIN_SECONDS):
            break
        v = _reflect(prof, draft, steps, run_id, found=found)
        notes.extend(v["be_aware"])   # kept even on PASS — these are useful, not failures
        if v["verdict"] == "PASS":
            break
        # A fix cycle is a re-plan plus another review — two more calls. Do not start them
        # unless the formatter will still have its time afterwards.
        if (c == MAX_REFLECT_CYCLES - 1 or _expired(deadline)
                or _low_on_time(deadline, DELIVER_RESERVE_S + PLANNER_FINALIZE_RESERVE_S)):
            # The re-planned draft may have fixed some of these. Re-run the free checks so
            # the traveller is only warned about defects that are still there.
            still_wrong = set(audit.check_schedule(draft, prof)
                              + audit.check_opening_hours(draft, hours_seen))
            v["must_fix"] = [m for m in v["must_fix"]
                             if m in still_wrong or not _is_computed_defect(m)]
            # Out of cycles or time — deliver best effort, but say so. A FAIL carrying
            # nothing at all means the critic's own reply was unusable (validate_verdict
            # defaults to FAIL by design); shipping with no caveat would read to the user as
            # "fully validated".
            if v["must_fix"]:
                warnings.extend(v["must_fix"])
            elif not v["be_aware"]:
                warnings.append(
                    "The reviewer could not complete its check, so this itinerary has not "
                    "been validated. Times, opening hours and costs are worth confirming "
                    "yourself.")
            break
        draft = _plan(prof, steps, feedback={"issues": v["must_fix"], "fixes": v["fixes"]},
                      run_id=run_id, deadline=deadline,
                      forecast=forecast, prices=prices, hours_seen=hours_seen,
                      coords_seen=coords_seen)

    response = _deliver(prof, draft, steps, run_id, deadline, warnings, notes, coords)
    obs.log("run_end", run_id=run_id, branch="plan", steps=len(steps), warnings=len(warnings))
    return {"response": response, "steps": steps,
            "state": {"profile": prof, "plan": draft}, "branch": "plan"}
