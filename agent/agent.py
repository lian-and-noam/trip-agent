"""Agent orchestrator with multi-turn Conversational Intake.

One intake call reads the full conversation transcript and routes to one of three branches:
  A) required info still missing  -> ask ONE clarifying question, stop.
  B) complete but unconfirmed     -> show the typed profile, ask to confirm, stop.
  C) user confirmed               -> Preference Profiler, then ReAct Planner
                                     (<-> Reflection Layer) -> Output Formatter.

The expensive planning loops run only in Branch C, after the user confirms the profile,
so intake-only turns cost a single LLM call and the agent never plans against missing or
wrong assumptions. This is the main lever for staying within budget.

The backend holds no session state: the browser replays the entire conversation as one
string in `prompt`, and intake re-derives the current stage from that transcript alone.

Every LLM call is logged as a step {module, prompt, response} with module names that
match the architecture diagram (Conversational Intake, Preference Profiler, ReAct Planner,
Reflection Layer, Output Formatter).
"""
import json
import time
import urllib.parse

from .llm import chat, parse_json
from .tools import run_tool, TOOL_CATALOG, geocode_place
from . import schemas, obs

MAX_PLANNER_STEPS = 8     # tool-calling iterations per planner run before a forced finalize
MAX_REFLECT_CYCLES = 2    # at most two critic/re-plan cycles
MAX_OBS_CHARS = 1200      # trim tool observations fed back to the model to keep context small
MAX_RUN_SECONDS = 240     # stop starting new LLM work past this, to stay under Vercel's 300s limit


def _trace(steps, module, prompt, response):
    """Append one step in the required schema: {module, prompt, response}."""
    steps.append({"module": module, "prompt": prompt, "response": response})


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


# ---------- Module: Conversational Intake (+ Preference Profiler) ----------
def _profile(conversation):
    """One LLM call over the full conversation. Extracts the typed profile, flags which
    required fields are still missing, and detects whether the user has confirmed. Returns
    a decision dict; the branching happens in run_agent()."""
    sys = (
        "You are the Conversational Intake for a trip planner. You receive the FULL conversation "
        "so far as a transcript ('User:' lines are the traveller; 'Agent:' lines are your own "
        "earlier replies). Extract a typed trip profile from everything the user has said.\n"
        'Return ONLY JSON: {"profile":{...}, "confirmed":bool, "question":string}.\n'
        "profile keys: days (int), destination (string), style (string), group (string), "
        'budget ("low"|"mid-range"|"luxury"), and optionally when (string: travel dates or '
        "season), origin (string: departure city), dietary (string[]), "
        'walking ("light"|"moderate"|"high"|"unlimited" walking tolerance), '
        "accessibility (bool), priorities (string[]), avoid (string[]).\n"
        "RULES:\n"
        "- For any REQUIRED field the user has NOT stated or clearly implied "
        "(days, destination, style, group, budget), set it to null. NEVER invent required fields.\n"
        "- Do not ask about optional fields; only capture them if the user mentions them.\n"
        "- confirmed = true ONLY IF an earlier 'Agent:' turn already presented a complete profile "
        "and asked to confirm, AND the user's LATEST message clearly agrees to proceed "
        "(e.g. 'yes', 'yep', 'looks good', 'correct', 'go ahead'). Otherwise false.\n"
        "- question: if any required field is null, ask ONE friendly message that requests ALL the "
        'missing required fields together (not one at a time). If nothing is missing, set it to "".'
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": conversation}]
    obj = schemas.as_obj(_chat_json(msgs, temperature=0.2, max_tokens=700))
    profile = schemas.as_obj(obj.get("profile"))
    q = obj.get("question")
    return {
        "profile": profile,
        "missing": schemas.missing_required(profile),
        "confirmed": bool(obj.get("confirmed")),
        "question": q if isinstance(q, str) else "",
    }


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
    sys = (
        "You are the ReAct Planner for a trip. Work in a Thought -> Action -> Observation loop.\n"
        "Tools:\n" + TOOL_CATALOG + "\n\n"
        "On EACH turn return ONLY JSON, one of:\n"
        '  {"thought":"...","tool":"<tool_name>","tool_input":{...}}\n'
        '  {"thought":"...","done":true,"draft_plan":{"days":[{"day":1,"title":"...","items":['
        '{"time":"09:00","name":"...","duration_min":90,"cost_eur":0,"note":"..."}]}],"total_cost_eur":0}}\n'
        "Call a tool only when it adds real information. weather_tool returns LIVE data. "
        "booking_tool/flights_tool are fictive. Costs and the budget are PER PERSON for the whole "
        "trip; estimate cost_eur per person. Finish within %d tool calls." % MAX_PLANNER_STEPS
    )
    user = "Traveller profile:\n" + json.dumps(schemas.compact_profile(prof))
    if feedback:
        user += "\n\nCritic feedback to fix:\n" + json.dumps(feedback)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    seen_calls = set()  # repetition guard: (tool, canonical tool_input)

    for _ in range(MAX_PLANNER_STEPS):
        if deadline and time.monotonic() > deadline:
            break  # out of time — fall through to the forced finalize below
        turn = _chat_json(msgs, temperature=0.3, max_tokens=1100)
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
    msgs.append({"role": "user",
                 "content": 'Stop now. Return ONLY {"thought":"...","done":true,"draft_plan":{...}}.'})
    turn = _chat_json(msgs, temperature=0.2, max_tokens=1100)
    draft = (turn or {}).get("draft_plan")
    plan = schemas.validate_draft_plan(draft) or schemas.minimal_plan(prof)
    _trace(steps, "ReAct Planner",
           {"thought": (turn or {}).get("thought", "forced finalize"), "action": "finalize"},
           {"draft_plan": plan})
    obs.log("planned", run_id=run_id, forced=True, degraded=bool(plan.get("degraded")),
            cost_eur=plan.get("total_cost_eur"))
    return plan


# ---------- Module: Reflection Layer ----------
def _reflect(prof, draft, steps, run_id=None):
    sys = (
        "You are the Reflection Layer (critic). Check the draft itinerary against the profile for: "
        "geographic logic, time feasibility, budget, rest breaks, opening hours, and balance. "
        'Return ONLY JSON: {"verdict":"PASS"|"FAIL","issues":[...],"fixes":[...]}'
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
                                        "\n\nDraft:\n" + json.dumps(draft)}]
    verdict = schemas.validate_verdict(_chat_json(msgs, temperature=0.2, max_tokens=600))
    obs.log("reflected", run_id=run_id, verdict=verdict["verdict"], issues=len(verdict["issues"]))
    _trace(steps, "Reflection Layer", {"profile": prof, "draft": draft}, verdict)
    return verdict


# ---------- Module: Output Formatter ----------
def _with_map_links(plan, destination):
    """Add a Google Maps search link to every item, built deterministically from the
    place name and destination. The links are real (not model-generated), so the
    formatter never has to invent a URL."""
    out = {**plan, "days": []}
    for day in plan.get("days", []):
        items = []
        for it in day.get("items", []):
            query = urllib.parse.quote_plus(f"{it.get('name', '')} {destination}".strip())
            items.append({**it, "map_url": f"https://www.google.com/maps/search/?api=1&query={query}"})
        out["days"].append({**day, "items": items})
    return out


def _format(prof, plan, steps, run_id=None):
    plan = _with_map_links(plan, prof.get("destination", ""))
    sys = (
        "You are the Output Formatter. Turn the validated plan into a clear, friendly day-by-day "
        "itinerary in Markdown. For each item show time, name, a one-line tip, duration and cost. "
        "Each item has a map_url: render the item name as a Markdown link to that URL. Costs are "
        "per person. End with a per-day and grand total cost. Be concise."
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(schemas.compact_profile(prof)) +
                                        "\n\nPlan:\n" + json.dumps(plan)}]
    text = chat(msgs, temperature=0.4, max_tokens=1400) or ""
    _trace(steps, "Output Formatter", {"plan": plan}, {"itinerary_markdown": text})
    return text


def _safe_geocode(name):
    """Geocode gate: returns a hit or None; never raises (the network call is best-effort)."""
    try:
        return geocode_place(name)
    except Exception:
        return None


# ---------- Pipeline ----------
def run_agent(user_prompt):
    """Run one turn. `user_prompt` is the entire conversation transcript (stateless).
    Returns {"response": <markdown>, "steps": [...]} for every branch."""
    run_id = obs.new_run_id()
    deadline = time.monotonic() + MAX_RUN_SECONDS
    conversation = user_prompt or ""
    obs.log("run_start", run_id=run_id, chars=len(conversation))
    steps = []

    decision = _profile(conversation)
    missing = decision["missing"]

    # ---- Branch A: required info still missing -> ask, stop (no planner, no token waste). ----
    if missing:
        question = decision["question"] or _fallback_question(missing)
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "clarify", "missing": missing, "reply": question})
        obs.log("intake_clarify", run_id=run_id, missing=missing)
        return {"response": question, "steps": steps}

    prof = schemas.validate_profile(decision["profile"])

    # ---- Branch B: complete but not confirmed -> show profile, ask to confirm, stop. ----
    # Still one LLM call so far; we do not plan against an unconfirmed profile.
    if not decision["confirmed"]:
        reply = _confirmation_message(prof)
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "confirm", "profile": prof, "reply": reply})
        obs.log("intake_confirm", run_id=run_id)
        return {"response": reply, "steps": steps}

    # ---- Branch C: user confirmed -> only now spend tokens on the heavy loops. ----
    obs.log("intake_confirmed", run_id=run_id, destination=prof["destination"], days=prof["days"])
    _trace(steps, "Preference Profiler", {"conversation": conversation}, prof)

    warnings = []
    if _safe_geocode(prof["destination"]) is None:
        warnings.append(f'Could not locate "{prof["destination"]}" — it may be invalid or the '
                        "itinerary may be generic.")

    draft = _plan(prof, steps, run_id=run_id, deadline=deadline)
    for c in range(MAX_REFLECT_CYCLES):
        v = _reflect(prof, draft, steps, run_id)
        if v["verdict"] == "PASS":
            break
        if c == MAX_REFLECT_CYCLES - 1 or time.monotonic() > deadline:
            warnings.extend(v["issues"])  # out of cycles or time — deliver best effort, but say so
            break
        draft = _plan(prof, steps, feedback={"issues": v["issues"], "fixes": v["fixes"]},
                      run_id=run_id, deadline=deadline)

    # Deterministic budget/feasibility guard, independent of the critic.
    ceiling = schemas.budget_ceiling_eur(prof)
    total = draft.get("total_cost_eur", 0)
    if total > ceiling:
        warnings.append(f"Estimated cost €{total} exceeds the ~€{ceiling} guide for a "
                        f"{prof['budget']} {prof['days']}-day trip.")
    if draft.get("degraded"):
        warnings.append("The planner could not fully build this itinerary; some days are placeholders.")

    response = _format(prof, draft, steps, run_id)
    if warnings:
        unique = list(dict.fromkeys(w for w in warnings if w))
        banner = ("> ⚠️ **Delivered with caveats** — this plan was not fully validated:\n"
                  + "\n".join(f"> - {w}" for w in unique) + "\n\n")
        response = banner + response

    obs.log("run_end", run_id=run_id, steps=len(steps), warnings=len(warnings))
    return {"response": response, "steps": steps}
