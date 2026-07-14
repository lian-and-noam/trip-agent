-- Supabase schema for the Trip Planning Agent.
-- Paste this whole file into the Supabase SQL Editor and run it once.
-- Safe to re-run: every statement is idempotent.
--
-- SECURITY MODEL
-- The browser NEVER talks to Supabase. Only the serverless function does, using the
-- service-role key held in an environment variable. Row Level Security is therefore
-- enabled with no public policies at all: the anon and authenticated roles get nothing,
-- and the service role bypasses RLS by design. If you ever move reads client-side, write
-- explicit policies first — until then, deny-by-default is the correct posture.

-- ===========================================================================
-- 1. conversations — the current state of one planning session.
--
-- `plan` is what makes the revision path possible: the agent edits this object instead
-- of re-reading a rendered itinerary out of the chat transcript. It is also what lets a
-- browser refresh keep the itinerary.
-- ===========================================================================
create table if not exists public.conversations (
  id          uuid primary key,
  device_id   uuid,
  title       text,
  profile     jsonb,
  plan        jsonb,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create index if not exists conversations_device_idx
  on public.conversations (device_id, updated_at desc);

-- ===========================================================================
  id                bigserial primary key,
  run_id            text not null,
  conversation_id   uuid,
  branch            text,               -- clarify | confirm | plan | revise | question | resume | error
  llm_calls         integer not null default 0,
  prompt_tokens     integer not null default 0,
  completion_tokens integer not null default 0,
  -- Cost columns were removed with the spend guard: the rates they needed were never
  -- configured, so every figure recorded was zero. Call counts and timings are real.

  ms                integer,
  created_at        timestamptz not null default now()
);

-- ===========================================================================
-- 2. Keep updated_at current without the client having to send it.
--
-- This matters: PostgREST would pass a client-supplied "now()" through as a string
-- literal, and Postgres rejects 'now()' as timestamptz input. Letting the database own
-- the column removes that failure mode entirely. BEFORE UPDATE triggers also fire on the
-- ON CONFLICT DO UPDATE path that upserts take.
-- ===========================================================================
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists conversations_touch on public.conversations;
create trigger conversations_touch
  before update on public.conversations
  for each row execute function public.touch_updated_at();

-- ===========================================================================
-- 3. Deny by default. The service role bypasses these; anon/authenticated get nothing.
-- ===========================================================================
alter table public.conversations   enable row level security;

-- ===========================================================================
-- 4. Grants for service_role.
--
-- RLS and GRANTs are SEPARATE mechanisms. service_role bypasses RLS, but it still needs
-- table-level privileges, and on newer Supabase projects those are not handed out
-- automatically for tables created here. Without this block every request comes back
-- 403 / SQLSTATE 42501 "permission denied for table ...".
--
-- Only service_role is granted anything: the browser never talks to Supabase, so anon and
-- authenticated stay with no access at all.
-- ===========================================================================
grant usage on schema public to service_role;

grant select, insert, update, delete on public.conversations   to service_role;

grant usage, select on all sequences in schema public to service_role;

-- ===========================================================================
-- Operational queries — run these by hand; the app does not use them.
-- ===========================================================================
-- Did anything land at all?
--
-- Usage per turn, newest first:
--   select created_at, branch, llm_calls, prompt_tokens, completion_tokens, ms
--
-- Is the revision path actually cheaper than re-planning?
--   select branch,
--          count(*)                as turns,
--          round(avg(llm_calls),2) as avg_calls,
--          round(avg(ms))          as avg_ms
--
-- Stored itineraries:
--   select id, title, updated_at, jsonb_array_length(plan->'days') as days
--   from public.conversations order by updated_at desc;
--
