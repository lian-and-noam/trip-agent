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

from .llm import (chat, is_timeout, parse_json, set_wall as llm_set_wall,
                  wall_exceeded, ConfigError)
from .tools import run_tool, route_matrix, ToolError, TOOL_CATALOG, geocode_place
from . import audit, schemas, obs, usage
from .usage import CallLimitExceeded

# The planner is bounded by both a step count and a clock, whichever binds first.
# Research past ~8 steps mostly re-checks known facts, while a run that spends its whole
# budget researching delivers an unfinished itinerary.
# Research turns before a forced finalize. Back to 5: eight lookups spent the whole budget
# on research, leaving the finalize, review and fix to run against a wall that was nearly
# gone. Beyond about five the planner mostly re-checks things it already knows.
MAX_PLANNER_STEPS = int(os.environ.get("MAX_PLANNER_STEPS", "6"))
# Tool turns allowed when correcting a draft. Small on purpose: the plan and the defects are
# already in the prompt, so a fix looks up a fact at most.
MAX_FIX_STEPS = int(os.environ.get("MAX_FIX_STEPS", "3"))
# Ceiling on network time across one planner run. Tools make real HTTP calls, which share
# the run's budget with the model calls. Past this, tools are refused with a note telling
# the planner to work from what it has, rather than running the clock out.
MAX_TOOL_SECONDS = int(os.environ.get("MAX_TOOL_SECONDS", "35"))

# Consecutive failed lookups before the planner is told to stop researching. Each retry
# costs a full LLM turn, so a tool that will not answer can eat a run on its own.
MAX_TOOL_FAILURES = int(os.environ.get("MAX_TOOL_FAILURES", "2"))

# Two passes: the critic reviews, the planner fixes what it found, and the fixed draft is
# checked again. One pass meant defects were reported to the traveller but never repaired.
MAX_REFLECT_CYCLES = int(os.environ.get("MAX_REFLECT_CYCLES", "2"))
MAX_OBS_CHARS = 1200      # trim tool observations fed back to the model to keep context small

# Time held back from the research loop for the forced finalize that writes the itinerary.
# Only one model call has to fit inside it, so it is sized to a slow turn rather than to the
# whole tail of the run; the reflection cycles after it have their own deadline checks.
PLAN_WRITE_RESERVE_SECONDS = int(os.environ.get("PLAN_WRITE_RESERVE_SECONDS", "60"))

# Wall-clock budget for one turn. Vercel terminates the function at vercel.json's
# maxDuration (300s) — a termination no code here can catch, because the interpreter is
# killed rather than raised in, so the traveller gets a 504 instead of the degraded reply
# every other failure path produces. Finishing early is the only defence.
#
# MAX_RUN_SECONDS is the work budget: no new stage begins after it. HARD_WALL_SECONDS is
# the absolute stop for LLM calls, and llm.set_wall() truncates each call to the time left.
# The two are separate because a stage checks the deadline before it starts and the call it
# then makes can run for up to llm._TIMEOUT_S; the wall is what actually bounds that.
#
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "230"))
HARD_WALL_SECONDS = int(os.environ.get("HARD_WALL_SECONDS", "250"))


def _trace(steps, module, prompt, response):
    """Append one step in the required schema: {module, prompt, response}."""
    steps.append({"module": module, "prompt": prompt, "response": response})


def _expired(deadline, reserve=0):
    """Hard time gate: once this is True, no further LLM call may start.

    Checked immediately before every call in the pipeline, so a run degrades to whatever
    it has already built rather than being killed mid-flight by the platform.

    `reserve` brings the gate forward by that many seconds, for a caller that must keep
    time back for work it still has to do after it stops.
    """
    return deadline is not None and time.monotonic() >= deadline - reserve


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
    while obj is None and attempts < repairs and not wall_exceeded():
        # A repair is a second call, and one begun past the wall still gets the minimum
        # per-call timeout — so retrying there buys a chance of valid JSON at the cost of
        # overshooting the budget the platform kills the function for exceeding. The
        # caller already degrades on a None, which is the cheaper failure.
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
            "days, a different group or budget tier. Any change to a required field is a replace.\n"
            "A question is a question about the itinerary, never a statement about the trip. "
            "Asking 'which day is busiest?' or 'how much is day 3?' adds NOTHING to the "
            "profile — not to details, priorities, when or style — and changes no field. "
            "Carry every field through from what the traveller actually said earlier, in the "
            "same words: re-stating a value you already extracted in different wording reads "
            "downstream as a change of plan and discards the itinerary."
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


# Short, unambiguous "carry on" phrasings, split by what the traveller is asking for.
# `_RESUME` means keep working on the trip we have; `_RECHECK` means look at it again and
# deal with what is outstanding. Neither names a destination, a length, a group or a
# budget, which is what makes them safe to route deterministically: a message with no trip
# information in it cannot be a request for a different trip.
_RESUME = {
    "continue", "continue please", "please continue", "continue planning", "carry on",
    "carry on then", "carry on please", "keep going", "keep planning", "go on", "resume",
    "pick up where you left off", "finish it", "finish the plan", "המשך", "תמשיך",
}
_RECHECK = {
    "validate", "validate it", "validate the plan", "check it", "check it again",
    "check the plan", "recheck", "re-check", "verify", "verify it", "fix it", "fix them",
    "fix the issues", "fix the problems", "בדוק", "תתקן",
}


def _resume_phrase(conversation):
    """Classify the latest message as "resume", "recheck", or "" for anything else.

    Exact match on short phrase sets, the same discipline `_is_affirmative` follows:
    "continue" is unambiguous, while "continue but drop the museum" carries an instruction
    and belongs to the model. Falls back to the whole text so a caller passing a bare
    message rather than a transcript is handled too; a real transcript can never equal one
    of these phrases, so the fallback cannot misfire.
    """
    text = (_latest_user_message(conversation) or conversation or "")
    text = text.strip().lower().strip(".!?, ")
    if text in _RESUME:
        return "resume"
    return "recheck" if text in _RECHECK else ""


# Fields whose change means a genuinely different trip. `style` is deliberately excluded:
# it is free text that the model rephrases every turn ("cultural and food-focused" ->
# "cultural/food-focused"), which can read as a new destination and force a spurious
# re-confirmation plus a full re-plan — 9 calls where 3 would do. A real change of interests
# still reaches the Plan Editor as an ordinary revision.
TRIP_IDENTITY_FIELDS = ("destination", "days", "budget", "group")


def _required_changed(before, after):
    """The fields that define the *identity of the trip* and differ between two profiles.

    A deterministic backstop to the model's intent call. The product rule is that changing
    where, how long, for whom, or at what budget means a new trip rather than an edit, so a
    non-empty result forces the confirm/plan flow even if intake classified the turn as a
    revision.

    Returns the field names rather than a bare True. Intake re-extracts the whole profile
    from the transcript on every turn, so a difference here can be a real change of plan or
    the model rephrasing itself; naming the field is the difference between a trace that
    shows why a turn was re-confirmed and one that only shows that it was.
    """
    changed = []
    for f in TRIP_IDENTITY_FIELDS:
        b, a = before.get(f), after.get(f)
        if isinstance(b, str) and isinstance(a, str):
            if b.strip().lower() != a.strip().lower():
                changed.append(f)
        elif b != a:
            changed.append(f)
    return changed


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
                # Only an actual journey. Having a venue_from is not enough: the planner
                # sets it on ordinary stops too, and overwriting those replaced a 120-minute
                # Colosseum visit with a 1-minute walking time.
                if not (it.get("venue_from") and _is_transfer(it.get("name"))):
                    continue
                # A measured time far larger than planned means the route was computed in the
                # wrong mode — walking across a city returns hours for a metro ride. Trust
                # the planner rather than replace an hour's transit with a four-hour walk.
                planned = max(1, int(it.get("duration_min") or 0))
                if minutes > planned * 2 and minutes > 60:
                    break
                if abs(planned - minutes) >= 5:
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


def _plan(prof, steps, feedback=None, run_id=None, deadline=None, draft=None,
          forecast=None, prices=None, hours_seen=None, coords_seen=None):
    usage.set_module("ReAct Planner")
    sys = (
        "You are the ReAct Planner for a trip. Work in a Thought -> Action -> Observation loop.\n"
        "Tools:\n" + TOOL_CATALOG + "\n\n"
        "On EACH turn return ONLY JSON, one of:\n"
        '  {"thought":"...","tool":"<tool_name>","tool_input":{...}}\n'
        '  {"thought":"...","done":true,"draft_plan":{"days":[{"day":1,"title":"...","items":['
        '{"time":"09:00","name":"...","venue":"...","venue_from":"","duration_min":90,'
        '"cost_eur":0,"note":"..."}]}],"total_cost_eur":0}}\n\n'
        'FIELDS\n'
        '- venue: the exact searchable name of the place ("Charles Bridge", "Karlstejn '
        'Castle"). No descriptions, no "or similar". Empty only when the item happens '
        "nowhere. A wrong venue is worse than an empty one.\n"
        '- venue_from: for a journey, where it starts; venue is where it ends. Empty '
        "otherwise.\n"
        "- For MEALS and free time, venue is a NEIGHBOURHOOD or street, never a named "
        "restaurant. Put the cuisine, price level and where to look in the note.\n\n"
        "PLAN RULES\n"
        "- If the profile gives start_time/start_point, day 1 begins there and then. If it "
        "gives end_time/end_point, the last day must reach it with a 2-3h buffer and nothing "
        "after.\n"
        "- Use lodging, when named, as each day's base.\n"
        "- Unless start_time, end_time or details say otherwise, a day runs about 09:00 to "
        "22:00.\n"
        "- Never schedule sleep: a night at the lodging is not an item, and lodging is "
        "costed once, not once per day.\n"
        "- Honour every entry in details, or say in a note why you could not.\n"
        "- EVERY item needs a cost_eur, as a plain number. Never omit the field, never "
        "write \"varies\" or \"depends\", and use 0 only for something actually free. An "
        "item with no number is read as free and the plan is sent back to be re-priced.\n"
        "- When live prices are given above, cost every item from them rather than from "
        "memory, and name the figure you used in the note (\"~EUR18, mid-range trattoria\") "
        "so the price can be checked against its source.\n"
        "- Every item that happens somewhere needs a venue, including free time. If an item "
        "offers alternatives, choose one and mention the other in the note.\n"
        "- Account for travel between locations, as its own item or inside the times. CHOOSE "
        "one way to make each journey and cost it — pick what suits the stated budget, put the "
        "price in cost_eur, and mention the alternatives in the note. \"Depends which you "
        "choose\" priced at 0 is not a plan; a train or taxi to an airport always costs "
        "something.\n"
        "- Note anything needing advance booking or timed entry.\n"
        "- Check opening hours with maps_tool for the few timed sites where it matters. "
        "open_hours null means UNKNOWN: write the note for the traveller (\"check opening "
        "hours before you go\") and never mention tools or lookups.\n"
        "Costs are per person. Keep the plan realistic for the stated budget and pace.\n"
    )
    user = "Traveller profile:\n" + json.dumps(schemas.compact_profile(prof), ensure_ascii=False)
    if prices:
        user += ("\n\nCurrent local prices, from a live search. Cost the itinerary from these "
                 "rather than from memory:\n" + json.dumps(prices, ensure_ascii=False))
    if forecast:
        # Handed over rather than left for the planner to fetch: it is the same data, minus
        # an LLM round trip spent deciding to ask for it.
        user += ("\n\nForecast for the destination (use it — put outdoor days on the dry ones):\n"
                 + json.dumps(forecast, ensure_ascii=False))
    if feedback:
        user += ("\n\nFix the plan below against the listed defects. Do NOT look anything up — "
                 "everything you need is already here. Change only what the defects require "
                 "and return the corrected draft_plan.")
        user += "\n\nDefects to fix:\n" + json.dumps(feedback, ensure_ascii=False)
        if draft:
            user += "\n\nCurrent plan:\n" + json.dumps(draft, ensure_ascii=False)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    seen_calls = set()  # repetition guard: (tool, canonical tool_input)
    tool_seconds = [0.0]
    failures = [0]         # consecutive failed lookups; see MAX_TOOL_FAILURES
    venues_seen = []       # real places found this run, so an abort has something to salvage   # list so the loop body can mutate it; see MAX_TOOL_SECONDS

    # A fix already has the plan and the list of defects, so it needs a fact or two at most
    # — not another round of research. Given the full budget it rebuilds from scratch.
    steps_allowed = MAX_FIX_STEPS if feedback else MAX_PLANNER_STEPS
    for _ in range(steps_allowed):
        if _expired(deadline):
            # Hard gate. The previous `break` fell through to the forced finalize below,
            # which spent another LLM call after the time budget was already gone.
            return _abort_plan(prof, steps, run_id, venues=list(venues_seen))
        # Leave room to write the plan. Without this the loop researches right up to the
        # deadline and the hard gate above then aborts to a salvaged skeleton — the run
        # spends its whole budget on lookups and delivers no itinerary. Breaking here
        # instead falls through to the forced finalize, which turns that research into a
        # plan. (This check was a copy of the gate above, so it could never fire.)
        if _expired(deadline, reserve=PLAN_WRITE_RESERVE_SECONDS):
            obs.log("research_budget_spent", run_id=run_id)
            break
        # 3000, not 1100. For a reasoning model this becomes a 3000 + llm._REASONING_HEADROOM
        # completion cap, and roughly 2000 of that goes on hidden reasoning. A 7-day plan is
        # ~2000 tokens of JSON, so a smaller budget truncates it mid-object on the finalize
        # attempt: unparseable -> repair retry -> ~36s burned -> repeat until the deadline.
        # This is a ceiling, not a target — tool-call turns still emit tiny JSON.
        try:
            turn = _chat_json(msgs, temperature=0.3, max_tokens=3000)
        except (ConfigError, CallLimitExceeded):
            # Misconfiguration and the spend guard must fail loudly: neither is a transient
            # fault and both need an operator, not a degraded itinerary.
            raise
        except Exception as e:
            # Everything else — a timeout, a provider 5xx, a malformed reply — degrades. The
            # lookups already made are salvaged rather than discarded with the turn.
            obs.log("planner_call_failed", run_id=run_id, where="loop",
                    error=type(e).__name__)
            return _abort_plan(prof, steps, run_id, reason="a planner call did not complete",
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
        obs_json = json.dumps(observation, ensure_ascii=False)[:MAX_OBS_CHARS]
        msgs.append({"role": "assistant", "content": json.dumps(turn, ensure_ascii=False) if turn is not None else "{}"})
        msgs.append({"role": "user", "content": "Observation: " + obs_json + "\nContinue."})

    # Safety net: force a finalize and always return a valid plan (never None). Attempted
    # even with the research budget spent — that is the normal way the loop ends, and this
    # call is what turns the research into an itinerary. The wall truncates it, so it cannot
    # overrun; if there is genuinely no time it fails and the salvage path takes over.
    msgs.append({"role": "user",
                 "content": 'Stop now. Return ONLY {"thought":"...","done":true,"draft_plan":{...}}.'})
    try:
        turn = _chat_json(msgs, temperature=0.2, max_tokens=3000)   # see the loop call above
    except Exception as e:
        # Whatever went wrong — a timeout, a provider 5xx, a malformed reply — the research
        # already done is worth keeping. Salvage lays those venues out rather than binning
        # the turn.
        obs.log("planner_finalize_failed", run_id=run_id, error=type(e).__name__)
        return _abort_plan(prof, steps, run_id,
                           reason="the final planning call did not complete",
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
    out = {"days": days, "total_cost_eur": total}
    # An edit to one day does not resolve defects found elsewhere, so the record of what is
    # still outstanding survives it. A later "continue" then re-reviews the edited plan,
    # which is also what clears any entry the edit happened to fix.
    if base.get("open_issues"):
        out["open_issues"] = base["open_issues"]
    return out, sorted(changed)


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
    user = ("Traveller profile:\n" + json.dumps(schemas.compact_profile(prof), ensure_ascii=False) +
            "\n\nCurrent itinerary:\n" + json.dumps(plan, ensure_ascii=False) +
            "\n\nConversation (the requested change is the LAST user message):\n" + conversation)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    # The editor must return each changed day COMPLETE, so the completion has to fit every
    # item of every day it touches. At 1100 a two-day edit ("make days 2 and 3 lighter")
    # truncated mid-object: unparseable, one repair retry, then an empty patch — which the
    # caller reports to the traveller as "I could not apply that change" for a change that
    # was perfectly applicable. This is a ceiling, not a target; a one-day edit still emits
    # a small completion and is billed for what it writes.
    turn = schemas.as_obj(_chat_json(msgs, temperature=0.3, max_tokens=2400))
    new_plan, changed = _apply_patch(plan, turn)
    summary = schemas._as_str(turn.get("summary"))

    _trace(steps, "Plan Editor",
           {"profile": schemas.compact_profile(prof), "current_itinerary": plan,
            "conversation": conversation},
           {"thought": turn.get("thought"), "changed_days": changed, "summary": summary,
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
    user = ("Profile:\n" + json.dumps(schemas.compact_profile(prof), ensure_ascii=False) +
            "\n\nItinerary:\n" + json.dumps(plan, ensure_ascii=False) +
            "\n\nConversation (the question is the LAST user message):\n" + conversation)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = chat(msgs, temperature=0.2, max_tokens=400) or ""
    # The brief requires the step to carry the prompt. This used to record
    # {"plan_total_eur": N} — a summary of the input, not the input — so the trace could not
    # be read to see what the module was actually asked. The system prompt stays out: it is
    # the module's instructions rather than its prompt, and publishing it serves nobody.
    _trace(steps, "Itinerary Q&A",
           {"profile": schemas.compact_profile(prof), "itinerary": plan,
            "conversation": conversation},
           {"answer": text})
    obs.log("question_answered", run_id=run_id, chars=len(text))
    return text


# ---------- Module: Reflection Layer ----------
def _reflect(prof, draft, steps, run_id=None, found=None, suspect=None):
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
    user = ("Profile:\n" + json.dumps(schemas.compact_profile(prof), ensure_ascii=False) +
            "\n\nDraft:\n" + json.dumps(draft, ensure_ascii=False))
    if found:
        # Arithmetic and looked-up facts, computed deterministically. Handing these over
        # means the critic does not have to re-derive clock sums it checks unreliably, and
        # can spend its attention on judgement instead.
        user += ("\n\nAlready VERIFIED defects — these are computed, not guesses. Include "
                 "every one in must_fix verbatim, then add anything else you find:\n"
                 + json.dumps(found, ensure_ascii=False))
    if suspect:
        # A second channel, deliberately weaker than the one above. These are shapes that
        # usually mean something went wrong, but a correct plan can produce the same shape,
        # and only reading the itinerary tells the two apart. Asserted as fact they would
        # force a fix cycle over a meal the hotel includes; asked as questions they cost
        # the critic a moment's attention and nothing else.
        user += ("\n\nPossible problems — these are NOT verified. Judge each against the "
                 "itinerary: put it in must_fix only if it is really wrong, and ignore it "
                 "if the plan already accounts for it:\n"
                 + json.dumps(suspect, ensure_ascii=False))
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": user}]
    # A reasoning model reviewing a whole plan needs room: too small a completion budget is
    # spent entirely on hidden reasoning, returning nothing parseable. A timeout here must
    # not discard a finished plan, so a failure degrades to a verdict with no issues.
    try:
        raw = _chat_json(msgs, temperature=0.2, max_tokens=1400)
    except Exception as e:
        if not is_timeout(e):
            raise
        obs.log("reflect_timeout", run_id=run_id)
        raw = None
    verdict = schemas.validate_verdict(raw)
    # Whether the critic actually replied. `validate_verdict` defaults an unreadable reply
    # to FAIL with empty lists, which is deliberate — an unreadable critic must never
    # green-light a plan — but it makes "the critic failed" and "the critic found nothing"
    # identical in the result. They mean opposite things to the traveller, so the caller
    # needs to tell them apart.
    verdict["answered"] = raw is not None
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
        "Start with a title line: \"# <N>-day itinerary — <destination>\".\n"
        "LAYOUT — the itinerary is scanned, not read:\n"
        "- One bullet per item, and put a blank line between bullets so the times separate.\n"
        "- Start each bullet with the time in bold, then the name: **09:00** — [Name](url)\n"
        "- The tip, duration and cost go on indented sub-lines under that bullet. Write the tip "
        "as a plain sentence with NO \"Tip:\" label — the reader can see it is a tip.\n"
        "Costs are per person. End with a per-day and grand total cost. Be concise."
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(schemas.compact_profile(prof), ensure_ascii=False) +
                                        "\n\nPlan:\n" + json.dumps(linked, ensure_ascii=False)}]
    # Writing a whole itinerary needs most of a full timeout. Starting the call with less
    # than that spends the remaining time and returns nothing, so below the threshold the
    # deterministic renderer is used directly.
    text = ""
    if _expired(deadline):
        obs.log("format_skipped_low_time", run_id=run_id)
    else:
        try:
            text = chat(msgs, temperature=0.4, max_tokens=3000) or ""
        except Exception as e:
            # Nothing that happens here is worth losing a finished plan over. Whatever the
            # cause — a timeout, the call budget, an upstream 500 — the deterministic
            # renderer below produces the same itinerary without a model call.
            obs.log("format_failed", run_id=run_id, error=type(e).__name__)
            text = ""

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
        # Call count only. Token and cost figures are not recorded anywhere and the cost is
        # always zero without per-token rates, so logging them was noise.
        obs.log("run_usage", branch=out.get("branch"), calls=meter.snapshot()["calls"])
        return out
    except (ConfigError, CallLimitExceeded):
        # Not "something unforeseen broke": the server has no credentials, or the run hit
        # the spend ceiling. Neither is retryable and neither is the traveller's fault, so
        # they must reach the API layer, which turns each into its own error envelope. The
        # blanket handler below would answer a misconfigured deployment with status "ok"
        # and "try again", which can only ever fail — and hides the outage from the
        # operator. `_plan` re-raises both for the same reason.
        raise
    except Exception as e:
        # Last line of defence. Every stage already degrades on its own, so reaching here
        # means something unforeseen broke — and the traveller should still get a usable
        # reply rather than an error page. Whatever the run managed to produce is delivered
        # with an honest note; the real cause goes to the logs.
        obs.log("run_recovered", error=type(e).__name__, detail=str(e)[:200],
                steps=len(steps))
        return {
            "response": _salvage_response(state, steps),
            "steps": steps,
            "state": schemas.as_obj(state),
            "branch": "error",
            "usage": meter.snapshot(),
            "ms": int((time.monotonic() - t0) * 1000),
        }
    finally:
        usage.end_run()


def _salvage_response(state, steps):
    """The best reply available after an unexpected failure.

    A stored itinerary is worth showing again — the traveller can then edit it, or say
    "carry on" — and is far more use than an apology. With nothing stored, say plainly what
    happened and what to do next.
    """
    plan = schemas.as_obj(schemas.as_obj(state).get("plan"))
    if plan.get("days"):
        prof = schemas.as_obj(schemas.as_obj(state).get("profile"))
        return _with_caveats(
            _format_fallback(prof, plan),
            ["Something went wrong partway through this turn, so here is the itinerary as it "
             "stood. Tell me what to change and I will pick up from here."])
    return ("Something went wrong while I was working on that, and I do not have a plan to "
            "show you yet. Sending the same request again usually works — trips of three or "
            "four days are the most reliable.")


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


def _resume_fix(prof, plan, outstanding, steps, run_id, deadline):
    """Work through defects an earlier turn delivered but never resolved.

    A plan can ship with a caveat listing what the critic found and nothing corrected — the
    run reached its cycle or time budget mid-review. The defects are recorded on the plan,
    so a later "continue" is a request to finish the job rather than a description of a new
    trip. One fix pass against the recorded list, then one review of the result: whatever is
    still outstanding is recorded again, so asking a second time makes further progress
    instead of repeating this turn.

    Returns (plan, response).
    """
    warnings = []
    if _expired(deadline):
        return plan, _with_caveats(
            _format_fallback(prof, plan),
            ["Ran out of time before I could work on the outstanding issues."] + outstanding)

    # The fix pass is told not to look anything up, so anything it needs has to be in the
    # prompt already. Costs are the commonest thing a review leaves outstanding, and a
    # correction made from model memory is years out of date; this is one HTTP call and no
    # extra model turn. It degrades to None, exactly as it does on the planning path.
    prices = _prices_for(prof)
    try:
        fixed = _plan(prof, steps, draft=plan, feedback={"issues": outstanding, "fixes": []},
                      run_id=run_id, deadline=deadline, prices=prices)
    except Exception:
        # Keep the itinerary the traveller already has rather than losing the turn to a
        # failed correction, and repeat the defects so nothing quietly disappears.
        obs.log("resume_fix_failed", run_id=run_id)
        fixed, remaining, notes = plan, list(outstanding), []
    else:
        computed = audit.check_schedule(fixed, prof)
        if _expired(deadline):
            remaining, notes = computed, []
        else:
            try:
                verdict = _reflect(prof, fixed, steps, run_id, found=computed,
                                   suspect=audit.check_costs(fixed, prof))
                remaining, notes = verdict["must_fix"], verdict["be_aware"]
            except Exception:
                # A review that cannot finish must not take the corrected plan with it. The
                # computed checks above cost nothing and already ran, so their findings still
                # reach the traveller.
                obs.log("resume_review_failed", run_id=run_id)
                remaining, notes = computed, []

    fixed = dict(fixed)
    if remaining:
        fixed["open_issues"] = list(dict.fromkeys(remaining))
        warnings.extend(fixed["open_issues"])
    else:
        fixed.pop("open_issues", None)
    response = _deliver(prof, fixed, steps, run_id, deadline, warnings, notes,
                        coords=_safe_geocode(prof.get("destination", "")))
    return fixed, response


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

    # A stored profile answers for whatever this turn did not restate. Intake re-extracts
    # from the transcript alone, so a short follow-up on a conversation the server already
    # holds — "what does day 2 cost?" sent with its conversation_id — reads as a brand-new
    # request with every required field missing, and the traveller is asked where they want
    # to go while their itinerary is on screen. Only gaps are filled: anything the turn did
    # state still wins, so changing a required field is still seen and still re-confirmed.
    if missing and prior_profile:
        merged = dict(prior_profile)
        merged.update({k: v for k, v in schemas.as_obj(decision["profile"]).items()
                       if v not in (None, "", [], {})})
        if not schemas.missing_required(merged):
            obs.log("intake_filled_from_state", run_id=run_id, filled=missing)
            decision["profile"], missing = merged, []

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
    #
    # A question is exempt. The override exists to stop an itinerary being thrown away
    # without the traveller agreeing to it, and answering a question mutates nothing — it
    # replies from the stored profile and hands the stored plan straight back. Applying it
    # there costs the guard nothing and gains a failure mode: intake re-reads the whole
    # transcript every turn, so a profile it merely rephrases can differ from the stored one
    # even on a turn that asked for no change at all, and the traveller is answered with a
    # confirmation prompt for the plan they were asking about.
    changed_fields = (_required_changed(prior_profile, prof)
                      if prior_plan and prior_profile and intent != "question" else [])
    replacing = bool(changed_fields)
    if replacing:
        obs.log("intake_replace", run_id=run_id, changed=changed_fields)
        intent, confirmed = "replace", False

    # ---- Branch F: carry on with the plan we already have. -----------------------------
    # "continue" states nothing about where, how long, for whom or at what budget, so it
    # cannot be a request for a different trip — but intake reads the whole transcript and
    # would classify it off the trip described earlier, sending the traveller back to a
    # confirmation prompt for an itinerary they already have. This runs ahead of the intent
    # branches, and ahead of the `replacing` override above, for that reason.
    #
    # With defects recorded against the plan it means "finish what you started", which is a
    # fix cycle rather than a description of the plan. Without them, a "carry on" is an
    # ordinary edit request and Branch E handles it, while a "validate" has nothing left to
    # act on and is left to intake to route.
    resuming = _resume_phrase(conversation) if prior_plan else ""
    if resuming:
        base = prior_profile or prof
        outstanding = [i for i in (prior_plan.get("open_issues") or []) if i]
        if outstanding:
            _trace(steps, "Conversational Intake", {"conversation": conversation},
                   {"stage": "resume", "open_issues": outstanding})
            obs.log("intake_resume", run_id=run_id, outstanding=len(outstanding))
            plan, response = _resume_fix(base, prior_plan, outstanding, steps, run_id, deadline)
            obs.log("run_end", run_id=run_id, branch="resume", steps=len(steps))
            return {"response": response, "steps": steps,
                    "state": {"profile": base, "plan": plan}, "branch": "resume"}
        if resuming == "resume":
            intent, confirmed = "revise", True

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
               {"stage": "confirm", "intent": intent, "replacing": replacing,
                "changed_fields": changed_fields, "profile": prof, "reply": reply})
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
    unresolved = []     # defects being shipped rather than fixed; see _unfixed() below

    def _unfixed(defects):
        """Ship these defects as caveats and remember that nothing corrected them.

        `warnings` also carries things that are not the plan's fault — a destination that
        would not geocode, a run that timed out — so it cannot double as the record of what
        is still wrong with the itinerary itself.
        """
        unresolved.extend(defects)
        warnings.extend(defects)

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
        found = audit.check_schedule(draft, prof) + \
            audit.check_opening_hours(draft, hours_seen)
        # Costs go down the weaker channel. Live prices were handed to the planner but
        # nothing proves it used them, and an item at EUR0 is the shape of having ignored
        # them — it is also the shape of a genuinely free one, which is why this is put to
        # the critic rather than counted as a defect. If it were counted as one, an
        # itinerary would be sent back for a fix over a breakfast the hotel includes.
        suspect = audit.check_costs(draft, prof)
        # The review is optional; writing the plan up is not. With little time left, skipping
        # it leaves the formatter enough to work with — a good plan reviewed by nobody beats
        # a reviewed plan nobody can read.
        if _expired(deadline):
            # Out of time for a review, but the checks above already ran and cost nothing,
            # so their findings still reach the traveller.
            _unfixed(found)
            break
        # A review that fails must not take the plan with it. The draft is already valid;
        # losing the whole turn because the critic timed out is the worst possible trade.
        try:
            v = _reflect(prof, draft, steps, run_id, found=found, suspect=suspect)
        except (CallLimitExceeded, Exception):
            obs.log("review_failed", run_id=run_id)
            _unfixed(found)
            break
        notes.extend(v["be_aware"])   # kept even on PASS — these are useful, not failures
        if v["verdict"] == "PASS":
            break
        # Only the clock stops a fix. Reserving room for a whole fresh plan plus another
        # review meant the guard demanded most of the budget be unspent, so the fix never
        # ran — but a fix is a small call: it patches a draft against a list of defects
        # rather than researching a trip from scratch. Every call is truncated by the wall,
        # so an optimistic start cannot overrun; the worst case is this draft ships as-is.
        if c == MAX_REFLECT_CYCLES - 1 or _expired(deadline):
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
                _unfixed(v["must_fix"])
            elif not v["answered"]:
                # Only when the critic genuinely did not answer. This used to fire whenever
                # it returned nothing at all, which is also what a clean second pass looks
                # like after the fix cycle repaired everything the first pass found — so a
                # plan whose defects had all been corrected was delivered under a banner
                # saying it had not been validated.
                warnings.append(
                    "The reviewer could not complete its check, so this itinerary has not "
                    "been validated. Times, opening hours and costs are worth confirming "
                    "yourself.")
            break
        # Same for the fix: if it cannot finish, keep the draft we already have.
        previous_draft = draft
        try:
            draft = _plan(prof, steps, draft=previous_draft,
                          feedback={"issues": v["must_fix"], "fixes": v["fixes"]},
                          run_id=run_id, deadline=deadline, forecast=forecast,
                          prices=prices, hours_seen=hours_seen, coords_seen=coords_seen)
        except Exception:
            obs.log("fix_failed", run_id=run_id)
            draft = previous_draft
            _unfixed(v["must_fix"])
            break

    # Recorded on the plan, not just announced in the reply. The caveat above the itinerary
    # tells the traveller what is still wrong; this is what lets them then say "continue"
    # and have the agent act on it, and it survives being stored and reloaded.
    if unresolved:
        draft["open_issues"] = list(dict.fromkeys(unresolved))
    response = _deliver(prof, draft, steps, run_id, deadline, warnings, notes, coords)
    obs.log("run_end", run_id=run_id, branch="plan", steps=len(steps), warnings=len(warnings))
    return {"response": response, "steps": steps,
            "state": {"profile": prof, "plan": draft}, "branch": "plan"}
