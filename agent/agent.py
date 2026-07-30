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
import json
import os
import re
import time
import urllib.parse

from .llm import chat, is_timeout, parse_json
from .tools import run_tool, TOOL_CATALOG, geocode_place
from . import schemas, obs, usage

MAX_PLANNER_STEPS = 5     # tool-calling iterations per planner run before a forced finalize
MAX_REFLECT_CYCLES = 1    # one critic pass; a re-plan cycle does not fit the 300s budget
MAX_OBS_CHARS = 1200      # trim tool observations fed back to the model to keep context small

# Wall-clock budget for one turn. Vercel terminates the function at vercel.json's
# maxDuration (300s). A call that starts just before the deadline still runs for up to
# llm._TIMEOUT_S afterwards, so this must satisfy:
#     MAX_RUN_SECONDS + llm._TIMEOUT_S < maxDuration
# 180 + 110 = 290 < 300. The split favours the per-call ceiling because the Output Formatter
# is the last and slowest call (97s measured on a 7-day plan) and killing it would discard
# ~140s of finished work. A measured 7-day turn totals ~241s end to end, and overrunning
# degrades to the deterministic renderer rather than failing.
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "180"))


def _trace(steps, module, prompt, response):
    """Append one step in the required schema: {module, prompt, response}."""
    steps.append({"module": module, "prompt": prompt, "response": response})


def _expired(deadline):
    """Hard time gate: once this is True, no further LLM call may start.

    Checked immediately before every call in the pipeline, so a run degrades to whatever
    it has already built rather than being killed mid-flight by the platform.
    """
    return deadline is not None and time.monotonic() >= deadline


def _abort_plan(prof, steps, run_id, reason="the run deadline was reached"):
    """Return a valid plan without spending an LLM call, for when time has run out.

    Used both when the run deadline expires and when a single call times out. In either case
    more model work is not an option, and handing back a schema-valid skeleton beats
    discarding everything the turn has already done.
    """
    plan = schemas.minimal_plan(prof)
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
        'budget ("low"|"mid-range"|"luxury"), and optionally when (string: travel dates or '
        "season), origin (string: departure city), dietary (string[]), "
        'walking ("light"|"moderate"|"high"|"unlimited" walking tolerance), '
        "accessibility (bool), priorities (string[]: specific places or experiences the user "
        "named as must-see), avoid (string[]).\n"
        "RULES:\n"
        "- For any REQUIRED field the user has NOT stated or clearly implied "
        "(days, destination, style, group, budget), set it to null. NEVER invent required fields.\n"
        '- If the user names interests anywhere ("love food and culture"), that IS style. '
        "Set style from them and never ask about pace.\n"
        "- Do not ask about optional fields; only capture them if the user mentions them.\n"
        "- confirmed = true ONLY IF an earlier 'Agent:' turn already presented a complete profile "
        "and asked to confirm, AND the user's LATEST message clearly agrees to proceed "
        "(e.g. 'yes', 'yep', 'looks good', 'correct', 'go ahead'). Otherwise false.\n"
        "- question: if any required field is null, ask ONE friendly message that requests ALL the "
        'missing required fields together (not one at a time). If nothing is missing, set it to "".'
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
    return {
        "profile": profile,
        "missing": schemas.missing_required(profile),
        "confirmed": bool(obj.get("confirmed")),
        "question": q if isinstance(q, str) else "",
        # Unrecognised or absent intent falls back to "revise": editing the existing plan is
        # the cheap, reversible option, where guessing "replace" would silently discard it.
        "intent": intent if intent in INTENTS else "revise",
    }


# Fields whose change means a genuinely different trip. `style` is deliberately excluded:
# it is free text that the model rephrases every turn ("cultural and food-focused" ->
# "cultural/food-focused"), which used to read as a new destination and force a spurious
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
    """Deterministic clarifying question if the model did not supply one."""
    labels = {"destination": "where you'd like to go", "days": "how many days you're travelling",
              "budget": "your budget (low, mid-range, or luxury)", "group": "who's travelling",
              "style": "what you enjoy (e.g. food, culture, nature)"}
    parts = [labels.get(m, m) for m in missing]
    if len(parts) == 1:
        return f"Could you tell me {parts[0]}?"
    return "Could you tell me " + ", ".join(parts[:-1]) + f" and {parts[-1]}?"


def _confirmation_message(prof):
    """Branch B reply: show the structured profile and ask the user to confirm."""
    keys = ("destination", "days", "group", "budget", "style", "when", "origin",
            "dietary", "priorities", "avoid")
    shown = {k: prof[k] for k in keys if prof.get(k)}
    body = json.dumps(shown, ensure_ascii=False, indent=2)
    note = ""
    if prof.get("assumptions"):
        note = "\n\n_(Assumptions I made: " + "; ".join(prof["assumptions"]) + ")_"
    return ("Here is your trip profile:\n\n```json\n" + body + "\n```" + note +
            "\n\nDoes this look correct? Type **'yes'** to start planning — or tell me what to change.")


# ---------- Module: ReAct Planner ----------
def _plan(prof, steps, feedback=None, run_id=None, deadline=None):
    usage.set_module("ReAct Planner")
    sys = (
        "You are the ReAct Planner for a trip. Work in a Thought -> Action -> Observation loop.\n"
        "Tools:\n" + TOOL_CATALOG + "\n\n"
        "On EACH turn return ONLY JSON, one of:\n"
        '  {"thought":"...","tool":"<tool_name>","tool_input":{...}}\n'
        '  {"thought":"...","done":true,"draft_plan":{"days":[{"day":1,"title":"...","items":['
        '{"time":"09:00","name":"...","duration_min":90,"cost_eur":0,"note":"..."}]}],"total_cost_eur":0}}\n'
        "Call a tool only when it adds real information. weather_tool returns LIVE data. "
        "booking_tool/flights_tool are fictive. Costs and the budget are PER PERSON for the whole "
        "trip; estimate cost_eur per person. Finish within %d tool calls.\n\n"
        "PLAN RULES — the critic checks these, so build them in rather than leaving them out:\n"
        "- Account for travel between locations. Either add a short transfer item, or start the "
        "next item late enough to absorb it and say so in the note. Never schedule two places "
        "back to back as though they were adjacent.\n"
        "- On any day with more than about 6 hours of activity, include at least one rest, coffee "
        "or downtime item. Queueing and standing count as effort.\n"
        "- If an item normally needs advance booking or a timed entry (major museums, popular "
        'restaurants, guided experiences), begin its note with "Book ahead:".\n'
        "- Keep days balanced. Do not stack several multi-hour, high-queue sites on one day while "
        "another day is nearly empty.\n"
        "- Respect typical opening hours; never schedule a site before it usually opens."
        % MAX_PLANNER_STEPS
    )
    user = "Traveller profile:\n" + json.dumps(schemas.compact_profile(prof))
    if feedback:
        user += "\n\nCritic feedback to fix:\n" + json.dumps(feedback)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    seen_calls = set()  # repetition guard: (tool, canonical tool_input)

    for _ in range(MAX_PLANNER_STEPS):
        if _expired(deadline):
            # Hard gate. The previous `break` fell through to the forced finalize below,
            # which spent another LLM call after the time budget was already gone.
            return _abort_plan(prof, steps, run_id)
        # 3000, not 1100. For a reasoning model this becomes a 3000 + llm._REASONING_HEADROOM
        # completion cap, and roughly 2000 of that goes on hidden reasoning. A 7-day plan is
        # ~2000 tokens of JSON, so the old budget truncated it mid-object on every finalize
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
            return _abort_plan(prof, steps, run_id, reason="a planner call timed out")
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
            key = (tool, json.dumps(tool_input, sort_keys=True, default=str))
            if key in seen_calls:
                observation = {"ok": False, "note": "Repeated identical call ignored. "
                                                     "Choose a different tool/input or finalize with a draft_plan."}
            else:
                seen_calls.add(key)
                observation = run_tool(tool, tool_input)
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
        return _abort_plan(prof, steps, run_id)
    msgs.append({"role": "user",
                 "content": 'Stop now. Return ONLY {"thought":"...","done":true,"draft_plan":{...}}.'})
    try:
        turn = _chat_json(msgs, temperature=0.2, max_tokens=3000)   # see the loop call above
    except Exception as e:
        if not is_timeout(e):
            raise
        obs.log("planner_timeout", run_id=run_id, where="finalize")
        return _abort_plan(prof, steps, run_id, reason="the final planning call timed out")
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
def _reflect(prof, draft, steps, run_id=None):
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
        "weather, price drift, date-dependent risks.\n"
        '- verdict is "FAIL" only when must_fix is non-empty. be_aware items alone are a PASS.'
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
                                        "\n\nDraft:\n" + json.dumps(draft)}]
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
    obs.log("reflected", run_id=run_id, verdict=verdict["verdict"],
            must_fix=len(verdict["must_fix"]), be_aware=len(verdict["be_aware"]))
    _trace(steps, "Reflection Layer", {"profile": prof, "draft": draft}, verdict)
    return verdict


# ---------- Module: Output Formatter ----------
# Control characters have no place in a venue name, but models do emit them — a raw U+001F
# turned up where an em-dash belonged. Left in, quote_plus faithfully encodes it as %1F and
# the map link stops resolving.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _map_query(name, destination):
    """Build the query string for a maps link, stripped of control characters."""
    cleaned = _CONTROL_CHARS.sub(" ", f"{name} {destination}")
    return urllib.parse.quote_plus(" ".join(cleaned.split()))


def _with_map_links(plan, destination):
    """Add a Google Maps search link to every item, built deterministically from the
    place name and destination. The links are real (not model-generated), so the
    formatter never has to invent a URL."""
    out = {**plan, "days": []}
    for day in plan.get("days", []):
        items = []
        for it in day.get("items", []):
            query = _map_query(it.get("name", ""), destination)
            items.append({**it, "map_url": f"https://www.google.com/maps/search/?api=1&query={query}"})
        out["days"].append({**day, "items": items})
    return out


def _format_fallback(prof, plan):
    """Render the plan as Markdown without an LLM.

    Used when the time budget is spent before the Output Formatter can run, so a turn
    always returns the itinerary it actually built instead of nothing. Deterministic,
    costs no tokens, and reuses the same real map links as the LLM path.
    """
    plan = _with_map_links(plan, prof.get("destination", ""))
    dest = prof.get("destination") or "your destination"
    out = [f"# {prof.get('days', '')}-day itinerary — {dest}".strip(), ""]
    for day in plan.get("days", []):
        items = day.get("items", [])
        title = f" — {day['title']}" if day.get("title") else ""
        out.append(f"## Day {day.get('day', '')}{title}")
        for it in items:
            name = it.get("name") or "Activity"
            link = f"[{name}]({it['map_url']})" if it.get("map_url") else name
            head = " · ".join(p for p in (it.get("time"), link) if p)
            out.append(f"- **{head}** — {it.get('duration_min', 0)} min · €{it.get('cost_eur', 0)}")
            if it.get("note"):
                out.append(f"  - {it['note']}")
        out += ["", f"_Day total: €{sum(i.get('cost_eur', 0) for i in items)}_", ""]
    out.append(f"**Total: €{plan.get('total_cost_eur', 0)} per person**")
    return "\n".join(out)


def _format(prof, plan, steps, run_id=None, warnings=None):
    """Render the plan via the LLM, degrading to the deterministic renderer if it cannot.

    Two ways this call fails to produce text, and both were previously shipped straight to
    the user as an empty itinerary — a caveats banner with nothing underneath:

      * it times out (this is the last and slowest call in the pipeline), or
      * a reasoning model spends its entire completion budget thinking and returns "".

    Either way the plan itself is intact, so falling back to _format_fallback always beats
    returning nothing.
    """
    usage.set_module("Output Formatter")
    linked = _with_map_links(plan, prof.get("destination", ""))
    sys = (
        "You are the Output Formatter. Turn the validated plan into a clear, friendly day-by-day "
        "itinerary in Markdown. For each item show time, name, a one-line tip, duration and cost. "
        "Each item has a map_url: render the item name as a Markdown link to that URL. Costs are "
        "per person. End with a per-day and grand total cost. Be concise."
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
                                        "\n\nPlan:\n" + json.dumps(linked)}]
    try:
        text = chat(msgs, temperature=0.4, max_tokens=3000) or ""
    except Exception as e:
        if not is_timeout(e):
            raise
        obs.log("format_timeout", run_id=run_id)
        text = ""

    if not text.strip():
        text = _format_fallback(prof, plan)
        _trace(steps, "Output Formatter", {"plan": linked},
               {"itinerary_markdown": text, "fallback": True})
        obs.log("format_empty_fallback", run_id=run_id)
        if warnings is not None:
            warnings.append("The formatter returned nothing, so this is the plain itinerary.")
        return text

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
        head += ("> ℹ️ **Good to know before you go** — not problems with the plan:\n"
                 + "\n".join(f"> - {n}" for n in note) + "\n\n")
    return head + response


def _deliver(prof, plan, steps, run_id, deadline, warnings, notes=None):
    """Shared tail for the branches that hand back an itinerary.

    Runs the deterministic budget guard, then formats — via the LLM when there is time
    left, deterministically when there is not, so a turn always returns the plan it built.
    """
    ceiling = schemas.budget_ceiling_eur(prof)
    total = plan.get("total_cost_eur", 0)
    if total > ceiling:
        warnings.append(f"Estimated cost €{total} exceeds the ~€{ceiling} guide for a "
                        f"{prof['budget']} {prof['days']}-day trip.")
    if plan.get("timed_out"):
        warnings.append("The agent ran out of time before it could build a full itinerary; "
                        "this is a skeleton. Try a shorter trip, or send the request again.")
    elif plan.get("degraded"):
        warnings.append("The planner could not fully build this itinerary; some days are placeholders.")

    if _expired(deadline):
        response = _format_fallback(prof, plan)
        _trace(steps, "Output Formatter", {"plan": plan},
               {"itinerary_markdown": response, "fallback": True})
        obs.log("format_fallback", run_id=run_id)
        warnings.append("Ran out of time before the final polish — this is the plain itinerary.")
    else:
        response = _format(prof, plan, steps, run_id, warnings)
    return _with_caveats(response, warnings, notes)


def _run_turn(user_prompt, steps, state=None):
    """Run one turn. `user_prompt` is the conversation transcript; `state` is the prior
    {profile, plan}. Returns {"response", "steps", "state", "branch"} for every branch."""
    run_id = obs.new_run_id()
    deadline = time.monotonic() + MAX_RUN_SECONDS
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
        response = _deliver(base, new_plan, steps, run_id, deadline, warnings)
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
    if _safe_geocode(prof["destination"]) is None:
        warnings.append(f'Could not locate "{prof["destination"]}" — it may be invalid or the '
                        "itinerary may be generic.")

    draft = _plan(prof, steps, run_id=run_id, deadline=deadline)
    notes = []          # advisory travel tips; shown apart from the defect warnings
    for c in range(MAX_REFLECT_CYCLES):
        if _expired(deadline):
            break  # no budget left to review the draft; ship what we have
        v = _reflect(prof, draft, steps, run_id)
        notes.extend(v["be_aware"])   # kept even on PASS — these are useful, not failures
        if v["verdict"] == "PASS":
            break
        if c == MAX_REFLECT_CYCLES - 1 or _expired(deadline):
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
                      run_id=run_id, deadline=deadline)

    response = _deliver(prof, draft, steps, run_id, deadline, warnings, notes)
    obs.log("run_end", run_id=run_id, branch="plan", steps=len(steps), warnings=len(warnings))
    return {"response": response, "steps": steps,
            "state": {"profile": prof, "plan": draft}, "branch": "plan"}
