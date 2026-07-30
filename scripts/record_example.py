"""Record the worked examples served by GET /api/agent_info, from real agent runs.

Writes api/example_run.json — one entry per turn, each {prompt, full_response, steps},
which is the shape the project brief asks for.

    python scripts/record_example.py                    # live: makes real LLM calls
    LLM_CASSETTE_MODE=replay \
    LLM_CASSETTE_DIR=<dir> python scripts/record_example.py   # free, from a recording

Kept as a script rather than inline data so the examples can be regenerated after any
change to the prompts or the pipeline, instead of drifting out of date in a source file.

Supabase is disabled here on purpose: the examples must be reproducible and must not depend
on — or write to — live conversation state.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(ROOT, ".env"))
except Exception:
    pass

from agent.agent import run_agent, AgentError          # noqa: E402
from agent import store                                 # noqa: E402

store.URL = ""          # hermetic: no reads, no writes
store.KEY = ""

OUT = os.path.join(ROOT, "api", "example_run.json")

# The conversation to record. Each entry is what the user types next; the transcript is
# assembled exactly as index.html assembles it.
TURNS = [
    "4 days in Rome, a couple, mid-range budget, we love culture and food — "
    "historical sites, Italian food and gelato.",
    "yes",
]
AGENT_ECHO_LIMIT = 700     # mirrors index.html

# These must never appear in a published trace.
SENTINELS = ["You are the Conversational Intake", "You are the ReAct Planner",
             "You are the Plan Editor", "You are the Reflection Layer",
             "You are the Output Formatter"]
MODULES = {"Conversational Intake", "ReAct Planner", "Plan Editor",
           "Reflection Layer", "Output Formatter", "Itinerary Q&A"}


def transcript(history):
    out = []
    for role, text in history:
        if role == "user":
            out.append("User: " + text)
        else:
            t = (text or "").strip()
            out.append("Agent: " + t if len(t) <= AGENT_ECHO_LIMIT
                       else "Agent: [delivered an itinerary — the current plan is supplied separately]")
    return "\n".join(out)


examples, history, state = [], [], None
for i, msg in enumerate(TURNS, 1):
    history.append(("user", msg))
    try:
        out = run_agent(transcript(history), state=state)
    except AgentError as e:
        print(f"turn {i} failed: {e.__cause__ or e}")
        sys.exit(1)
    state = out.get("state")
    history.append(("agent", out["response"] or ""))
    examples.append({
        "prompt": msg,
        "full_response": out["response"],
        "steps": out["steps"],
    })
    print(f"  turn {i}: branch={out['branch']:<8} steps={len(out['steps'])} "
          f"calls={out.get('usage', {}).get('calls')}")

# --- validate before writing: an example that leaks or misnames is worse than none ---
blob = json.dumps(examples, ensure_ascii=False)
problems = []
for s in SENTINELS:
    if s in blob:
        problems.append(f"system prompt leaked: {s!r}")
for ex in examples:
    if not isinstance(ex["full_response"], str) or not ex["full_response"].strip():
        problems.append("an example has an empty full_response")
    for st in ex["steps"]:
        if list(st.keys()) != ["module", "prompt", "response"]:
            problems.append(f"bad step schema: {list(st.keys())}")
        if st["module"] not in MODULES:
            problems.append(f"unknown module: {st['module']!r}")
if problems:
    print("\nNOT WRITTEN — validation failed:")
    for p in dict.fromkeys(problems):
        print("  -", p)
    sys.exit(1)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)
print(f"\nwrote {OUT}  ({len(examples)} examples, {len(blob):,} chars)")
