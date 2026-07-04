# Trip Planning AI Agent

An autonomous trip-planning agent. From a single natural-language request, it profiles the
traveller through a short dialogue, plans an itinerary with a ReAct loop over travel tools,
self-critiques the draft, and returns a costed, day-by-day plan.

The agent is exposed over HTTP and comes with a minimal web UI for operating and inspecting it.

## What it does

Given a request such as *"7 days in Japan, a couple, mid-range budget, love food and culture,
must see Kyoto temples and Mt Fuji"*, the agent:

1. Confirms it has the details it needs (asking follow-up questions if not).
2. Builds a typed traveller profile.
3. Plans an itinerary, calling tools for live and structured data.
4. Reviews the draft against the profile and re-plans if it finds problems.
5. Returns a Markdown itinerary with times, durations, and per-day and total costs.

## Architecture

The pipeline is five modules, each an LLM call with its own system prompt, matching the
diagram returned by `GET /api/model_architecture`:

| Module | Role |
|---|---|
| **Conversational Intake** | Reads the whole conversation, decides what is still missing, and asks for it. |
| **Preference Profiler** | Turns the confirmed conversation into a typed, validated profile. |
| **ReAct Planner** | Thought → Action → Observation loop that calls tools and drafts the plan. |
| **Reflection Layer** | A critic that checks the draft (geography, time, budget, balance) and can trigger a re-plan (max 2 cycles). |
| **Output Formatter** | Renders the validated plan as a friendly day-by-day itinerary. |

The intake step routes each turn to one of three branches:

- **A — missing info:** ask one clarifying question and stop.
- **B — complete but unconfirmed:** show the profile and ask the user to confirm.
- **C — confirmed:** run the planner, reflection, and formatter.

The expensive planning loops run only in branch C, after the user confirms, so a turn that is
still gathering information costs a single LLM call.

### Tools available to the planner

`weather_tool` returns **live** data (Open-Meteo, no API key). `maps_tool`, `search_tool`, and
`reviews_tool` are structured mocks with a stable shape, ready to be swapped for a real API.
`flights_tool` and `booking_tool` are **fictive** and never make a real reservation or purchase.
`calendar_tool` builds an `.ics` string. Two side-effecting tools (`booking_confirm_tool`,
`flight_book_tool`) are **gated**: they require explicit user approval and are never callable
from the loop.

## Project layout

```
api/team_info.py            GET  /api/team_info           student details
api/agent_info.py           GET  /api/agent_info          agent meta + a worked example
api/model_architecture.py   GET  /api/model_architecture  architecture diagram (PNG)
api/execute.py              POST /api/execute             main entry point
agent/agent.py              orchestrator, branching, and step tracing (module prompts live here)
agent/llm.py                LLMod client (via the OpenAI SDK) and JSON parsing
agent/tools.py              tool implementations and the deny-by-default dispatcher
agent/schemas.py            validation/coercion of every LLM output
agent/obs.py                structured logging
index.html                  web UI, served at /
architecture.png            served by /api/model_architecture
scripts/make_architecture.py  regenerates architecture.png
tests/                      unit tests and contract evals
```

## API

- `GET /` — web UI
- `GET /api/team_info` — student names and emails
- `GET /api/agent_info` — description, purpose, prompt template, and a worked example
- `GET /api/model_architecture` — the architecture diagram as a PNG
- `POST /api/execute` — body `{ "prompt": "..." }`

`/api/execute` always responds with the same envelope:

```json
{ "status": "ok", "error": null, "response": "…markdown itinerary…", "steps": [ … ] }
```

On failure, `status` is `"error"`, `error` holds a human-readable message, and `response` is
`null`. `steps` is an ordered list of every LLM call made, each `{ "module", "prompt", "response" }`,
with module names matching the architecture diagram.

The agent is stateless: the browser keeps the conversation and replays the whole transcript in
`prompt` on every request, so multi-turn intake works without any server-side session.

## Configuration

The agent needs an LLMod.ai key. Copy the example env file and fill it in:

```bash
cp .env.example .env
# then edit .env and set LLMOD_API_KEY (and, if needed, LLMOD_BASE_URL / LLMOD_MODEL)
```

`.env` is git-ignored and is loaded automatically at startup. The three variables are
`LLMOD_API_KEY`, `LLMOD_BASE_URL`, and `LLMOD_MODEL`.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
vercel dev                                           # serves the UI at / and the API under /api/*
```

Then open `http://localhost:3000`, describe a trip, confirm the profile, and read the plan. The
UI also shows the full step trace for the latest turn.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests mock the LLM boundary, so they are deterministic and make no network or LLM calls.
They cover:

- **Intake branching** — that a turn only plans once the profile is confirmed, and that
  clarify/confirm turns cost a single LLM call.
- **Contract shape** — that `/api/execute` returns exactly `{status, error, response, steps}`
  and that each step is `{module, prompt, response}` with a valid module name.
- **Crash-proofing** — that malformed model output degrades gracefully instead of raising.
- **Tool safety** — that unknown and gated tools are refused and tool inputs are filtered.
- **JSON parsing** — that the parser never raises and extracts balanced objects.

## Regenerating the architecture diagram

```bash
pip install matplotlib
python scripts/make_architecture.py   # rewrites architecture.png
```
