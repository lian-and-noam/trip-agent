# Trip Planning AI Agent

An autonomous trip-planning agent. From a natural-language request it profiles the traveller
through a short dialogue, plans an itinerary with a ReAct loop over travel tools,
self-critiques the draft, and returns a costed, day-by-day plan — then lets you revise that
plan without re-planning it from scratch.

The agent is exposed over HTTP and comes with a minimal web UI for operating and inspecting it.

## What it does

Given a request such as *"7 days in Japan, a couple, mid-range budget, love food and culture,
must see Kyoto temples and Mt Fuji"*, the agent:

1. Confirms it has the details it needs (asking follow-up questions if not).
2. Builds a typed traveller profile and asks you to confirm it.
3. Plans an itinerary, calling tools for live and structured data.
4. Reviews the draft against the profile and re-plans if it finds problems.
5. Returns a Markdown itinerary with times, durations, and per-day and total costs.

Afterwards you can say *"make day 2 lighter"* and it edits that day — or ask *"what does day 3
cost?"* and it just answers. Neither re-runs the planner.

## Architecture

Six LLM modules, matching the diagram returned by `GET /api/model_architecture`:

| Module | Role |
|---|---|
| **Conversational Intake** | Reads the dialogue, extracts the typed profile, decides what is missing, and classifies intent. |
| **ReAct Planner** | Thought → Action → Observation loop that calls tools and drafts the plan. |
| **Plan Editor** | Applies a requested change to an existing itinerary, patching only the days involved. |
| **Reflection Layer** | A critic that checks the draft (geography, time, budget, balance) and can trigger a re-plan. |
| **Output Formatter** | Renders the validated plan as a friendly day-by-day itinerary. |
| **Itinerary Q&A** | Answers a question about the delivered plan without changing it. |

A seventh component, **Validation & Coercion** (`agent/schemas.py`), is deterministic Python —
it makes no model call, so it appears on the diagram but never in the `steps` trace.

### Branching, and what each branch costs

One intake call routes every turn:

| Branch | Condition | Behaviour | LLM calls |
|---|---|---|---|
| **A** | A required field is missing | Ask once for everything missing, stop | 1 |
| **B** | Complete but unconfirmed | Show the typed profile, ask to confirm, stop | 1 |
| **C** | Confirmed, no plan yet | Planner → Reflection → Formatter | ~9 |
| **D** | Plan exists, user asks about it | Itinerary Q&A | 2 |
| **E** | Plan exists, user wants a change | Plan Editor → Formatter | 3 |

The expensive loop runs only in branch C. A follow-up edit costs roughly a third of a re-plan,
which is the main lever for staying inside the project budget.

**Changing a required field is not an edit.** If the destination, day count, group, budget, or
style changes, the turn is routed back through confirmation rather than the Plan Editor — a
different destination means a different trip, and the existing itinerary is kept until the
replacement is confirmed. That override is deterministic and does not depend on the model
classifying the intent correctly.

### The revision path

The itinerary is stored server-side as structured JSON, keyed by an anonymous
`conversation_id`, and handed to the Plan Editor as state. Two consequences:

- The rendered itinerary never travels through the prompt, so the transcript stays flat
  instead of growing by ~9,000 characters per revision.
- The Plan Editor returns **only the days it changes**. Days nobody asked about are merged
  through byte-identical rather than re-emitted, which removes any chance of a 35-item plan
  being silently truncated or reworded on the way back.

### Tools available to the planner

`weather_tool` returns **live** data (Open-Meteo, no API key). `maps_tool`, `search_tool`, and
`reviews_tool` are structured mocks with a stable shape, ready to be swapped for a real API —
they are not wired to any provider today. `flights_tool` and `booking_tool` are **fictive** and
never make a real reservation or purchase. `calendar_tool` builds an `.ics` string. Two
side-effecting tools (`booking_confirm_tool`, `flight_book_tool`) are **gated**: they require
explicit approval and are never callable from the loop.

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
agent/usage.py              token/cost metering and the budget ceilings
agent/store.py              Supabase persistence (optional)
agent/cassette.py           record/replay of LLM calls for cost-free development
agent/obs.py                structured logging
db/schema.sql               Supabase tables, RLS, and the spend function
index.html                  web UI, served at /
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

The request body accepts two **optional** extra fields, `conversation_id` and `device_id`
(both UUIDs). They enable the revision path by telling the server which stored itinerary to
edit. A request carrying only `prompt` behaves exactly as documented above — the response
envelope never changes shape.

## Configuration

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

`LLMOD_API_KEY` is the only required variable. Everything else has a working default; see the
comments in `.env.example` for the budget guards, timing invariant, and Supabase settings.

### Supabase (optional)

Persistence is an enhancement, never a dependency — unconfigured, slow, or erroring, the agent
runs exactly as it did without it. To enable it, run `db/schema.sql` in the Supabase SQL editor
and set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.

The browser never talks to Supabase; only the serverless function does. RLS is therefore
enabled with no public policies, and the service-role key must stay server-side.

Because the brief forbids authentication guards, there is no user account: `device_id` is a
UUID in `localStorage` that identifies a **browser**, not a person, and it is forgeable. Store
only low-sensitivity trip preferences against it.

## Staying inside the budget

The course allowance is **$9 total**, so cost is measured rather than estimated:

- Every response's `usage` is recorded per call and attributed to the module that spent it.
- `LLM_MAX_CALLS_PER_RUN` is a hard per-request ceiling that fires whether or not prices are
  configured — it is what stops a runaway loop.
- `LLM_BUDGET_USD` stops a run once cumulative spend passes a limit.
- Each turn is written to the `runs` table. Current total:

```sql
select public.total_spend_usd();

-- spend by branch, to confirm revisions really are cheaper than re-planning
select branch, count(*) turns, round(avg(llm_calls), 2) avg_calls, round(avg(cost_usd), 5) avg_usd
from public.runs group by 1 order by avg_usd desc;
```

### Developing without spending

Record each distinct LLM request once, then replay it for free:

```bash
LLM_CASSETTE_MODE=auto pytest        # records on first run, replays afterwards
LLM_CASSETTE_MODE=replay pytest      # strict: a cache miss is an error, so CI cannot spend
```

The cache key covers the model, messages, temperature, token cap, and JSON mode, so editing a
prompt correctly misses the cache and re-records only what changed. Never set this in production.

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
- **The revision path** — that an edit patches only the days requested, that untouched days
  come through identical, that a revision costs fewer calls than a re-plan, and that changing
  a required field routes to re-confirmation instead of an edit.
- **Contract shape** — that `/api/execute` returns exactly `{status, error, response, steps}`
  and that each step is `{module, prompt, response}` with a valid module name.
- **Budget guards** — that token counts come from the provider, that the per-run call ceiling
  and spend ceiling both raise, and that cost attributes to the right module.
- **Persistence** — that a missing, slow, or failing Supabase degrades silently, and that only
  well-formed UUIDs ever reach a database filter.
- **Crash-proofing** — that malformed model output degrades gracefully instead of raising.
- **Tool safety** — that unknown and gated tools are refused and tool inputs are filtered.
- **JSON parsing** — that the parser never raises and extracts balanced objects.

## Deployment

Deployed on Vercel. `vercel.json` sets `maxDuration` to 300s for `/api/execute`; the agent's own
budget is lower so it can always finish and respond:

```
MAX_RUN_SECONDS + LLM_TIMEOUT_S < maxDuration      (200 + 40 = 240 < 300)
```

Every LLM call is gated on that deadline. If the budget runs out mid-turn the agent returns the
plan it has, rendered deterministically, with a visible caveat — rather than being killed
mid-flight by the platform.

Python is pinned to 3.12 via `.python-version`, matching CI.

## Regenerating the architecture diagram

```bash
pip install -r requirements-dev.txt
python scripts/make_architecture.py   # rewrites architecture.png
```

Module names in the diagram, in the `steps` trace, and in `/api/agent_info` must stay in sync.
