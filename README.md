# Atlas — an autonomous trip-planning agent

Atlas turns a sentence like *"5 days in Rome, a couple, mid-range, we love food"* into a
costed, day-by-day itinerary with real map links — then keeps talking, so *"make day 2
lighter"* edits the plan you already have instead of starting over.

**Live:** the Vercel URL in the submission · **UI:** the site root · **API:** `/api/*`

---

## What it does

A traveller describes a trip in their own words. The agent works out what is still missing,
asks for it in one message, reads the trip back for confirmation, then researches and writes
an itinerary: times, durations, per-person costs, a map link for every place, and a short
list of things worth knowing before going.

Once a plan exists, the conversation continues. *"Swap the Tuesday afternoon for something
indoors"* rebuilds that day only. *"What does day 3 cost?"* answers from the plan without
touching it.

Two ideas shape the whole design:

**Only pay for the work the turn actually needs.** A single intake call reads the entire
dialogue and decides which of five paths the turn takes. A clarifying question costs one LLM
call; a full plan costs around nine. Nothing plans until the traveller has confirmed what is
being planned.

**Never present a guess as a fact.** Anything checkable is checked in code rather than left
to a model's judgement — arithmetic, budgets, opening hours, travel times. Anything that
cannot be verified is either labelled or left out. The agent will say hours are unconfirmed
rather than assert opening times it never looked up.

---

## How it works

One **Conversational Intake** call classifies every turn and routes it:

| Branch | When | Cost |
|---|---|---|
| **A · clarify** | required details missing | 1 call |
| **B · confirm** | complete but unconfirmed | 1 call |
| **C · plan** | confirmed — full pipeline | ~9 calls |
| **D · answer** | a question about the plan | 2 calls |
| **E · revise** | edit an existing plan | 3 calls |
| **F · resume** | "continue" — clear defects the critic left open | 4-5 calls |

Required: destination, days, group, budget. Everything else — interests, dates, arrival and
departure points, walking tolerance, dietary needs — is captured when mentioned and never
demanded.

On the planning branch:

**ReAct Planner** — a Thought → Action → Observation loop over the travel tools. It looks
things up only where the answer changes the plan, bounded by both a step count and a clock.

**Deterministic audit** (`agent/audit.py`) — no model, no network. Checks the draft for
overlapping items, days ending after midnight, activities scheduled outside opening hours the
run actually retrieved, a first day starting before the traveller arrives, and a last day
that misses the departure. It never rewrites the plan — it reports, and the critic decides.

Costs are checked the same way but reported separately, because they are judgements rather
than arithmetic. Live local prices are fetched once per run and given to the planner, and a
meal costed at nothing with no note explaining it is the shape of a planner that ignored
them — but it is also the shape of a breakfast the hotel includes, so the critic is asked
rather than told. An earlier version imposed per-person price floors here; it overwrote
costs that had been researched correctly, and was removed.

**Reflection Layer** — a critic model that receives those computed defects as established
fact and adds its own judgement: is this day too packed, does it match the stated interests,
what should the traveller know before going. Handing it the arithmetic means it spends its
attention on the things only judgement can settle.

**Output Formatter** — writes the final Markdown. Every URL it uses is built in code from
place names, so no link can be invented.

If the critic finds defects it cannot resolve, the plan is still delivered — with the
unresolved issues stated at the top. A flawed itinerary you can see the flaws in is more
useful than none. Those issues are recorded on the plan itself rather than only announced,
so "continue" on the next turn re-enters the fix cycle instead of describing the plan back.

### Tools

| Tool | Data |
|---|---|
| `weather_tool` | Real — Open-Meteo, no key |
| `maps_tool` | Real — Geoapify: address, coordinates, OpenStreetMap opening hours |
| `search_tool`, `reviews_tool` | Real — Tavily: summarised web text with sources |
| `flight_search_tool`, `booking_tool` | Fictive, labelled as such |
| `booking_confirm_tool`, `flight_book_tool` | Tier 2 — gated, never fire autonomously |

Where a lookup returns nothing, the agent says so. `maps_tool` returns `open_hours: null`
when OpenStreetMap has no hours recorded, and that is treated as *unknown*, never as closed.
`reviews_tool` returns quoted text with source links and no star rating, because a
manufactured number reads as verified in a way a quotation does not.

Anything that cannot be sourced is deliberately vague rather than confidently wrong: meals
name a neighbourhood ("dinner in Monti — Roman trattorias, options around Via dei Serpenti")
rather than a specific restaurant that may have closed since the model last heard of it.

---

## API

`GET /api/team_info` · `GET /api/agent_info` · `GET /api/model_architecture` (PNG) ·
`POST /api/execute`

### `POST /api/execute`

```json
{ "prompt": "5 days in Rome, a couple, mid-range, we love culture and food" }
```

```json
{ "status": "ok", "error": null, "response": "…markdown itinerary…", "steps": [ … ] }
```

`steps` lists every LLM call in order, each with the module name — matching the architecture
diagram — plus its prompt and response.

**The backend is stateless, so the conversation lives in `prompt`.** Send the running
transcript as `User:` / `Agent:` lines. To confirm a trip, include the turn where the agent
asked, then the reply:

```json
{ "prompt": "User: 5 days in Rome, a couple, mid-range, we love food\nAgent: **Here's your trip so far** … Does this look right? Reply yes…\nUser: yes" }
```

Editing or asking about an existing plan works the same way — but the plan itself is large,
so it is passed by reference instead:

```json
{
  "prompt": "User: 5 days in Rome…\nAgent: [delivered an itinerary]\nUser: make day 2 lighter",
  "conversation_id": "8f14e45f-ceea-467a-9f8b-2c1d3e4a5b6c",
  "device_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

`conversation_id` retrieves the stored profile and plan so a revision patches the existing
itinerary rather than replanning the trip. `device_id` identifies an anonymous traveller
across conversations. Both are optional: a request carrying only `prompt` works, it simply
plans afresh each time.

Errors return the same envelope with `"status": "error"`, a human-readable `error`, and the
steps completed before the failure.

`GET /api/agent_info` carries worked examples recorded from real runs — prompt, full
response, and the step trace for each.

---

## Built with

Python 3.12 on Vercel serverless · LLMod.ai (OpenAI-compatible) · Supabase for conversation
state · Open-Meteo, Geoapify and Tavily for live data · a plain
HTML/JS front end with no build step.

`agent/` holds the pipeline: `agent.py` (routing and modules), `audit.py` (deterministic
checks), `schemas.py` (validation and coercion), `tools.py`, `llm.py`, `store.py`,
`usage.py`. `api/` holds the four endpoints. `tests/` holds 209 tests, including a browser
suite for the UI.

### Notes on the constraints

Vercel caps a serverless request at 300 seconds, and it enforces that by killing the
process — which no error path here can catch, so the only defence is finishing early. Work
stops at 230s and no model call may begin after 250s, leaving the rest for the cold start,
both database round trips and the response. The arithmetic behind those two numbers is
written out beside them in `agent/agent.py` and pinned by a test. If time runs short the
agent degrades to a deterministic renderer rather than failing.

Every run counts its LLM calls and stops at a hard per-run ceiling, which is what prevents a
runaway loop. Tokens are counted from what the provider reports and broken down per module.
Prompts carry a compacted profile rather than the raw conversation, and the plan is passed by
reference on revision turns, so the context stays small.

Tier-2 tools — anything that would spend money — are gated in code and cannot be triggered by
the model, whatever a prompt asks for. Outbound HTTP is restricted to an allow-list of hosts.
