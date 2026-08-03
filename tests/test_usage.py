"""Token metering, the per-run call ceiling, and the wall-clock budget.

The call ceiling is what stops a runaway loop, and the wall-clock arithmetic is what keeps
a run from being killed by the platform mid-reply, so both need to hold under test rather
than being trusted by inspection.
"""
import pytest

from agent import usage


@pytest.fixture(autouse=True)
def _clean():
    yield
    usage.end_run()


def test_tokens_are_counted_from_the_provider_not_estimated():
    m = usage.start_run("r1")
    m.record(1000, 500)
    m.record(250, 100)
    snap = m.snapshot()
    assert snap["calls"] == 2
    assert snap["prompt_tokens"] == 1250 and snap["completion_tokens"] == 600
    assert snap["total_tokens"] == 1850


def test_per_run_call_ceiling_stops_a_runaway_loop():
    m = usage.start_run("r5", max_calls=3)
    for _ in range(3):
        m.before_call()
        m.record(10, 10)
    with pytest.raises(usage.CallLimitExceeded) as e:
        m.before_call()
    assert "call ceiling" in str(e.value)


def test_cost_attributes_to_the_module_that_spent_it(monkeypatch):
    m = usage.start_run("r7")
    usage.set_module("ReAct Planner")
    m.record(1000, 1000)
    m.record(500, 0)
    usage.set_module("Output Formatter")
    m.record(200, 300)
    by = m.snapshot()["by_module"]
    assert by["ReAct Planner"]["calls"] == 2
    assert by["Output Formatter"]["calls"] == 1
    assert by["ReAct Planner"]["prompt_tokens"] == 1500


def test_no_active_run_is_not_an_error():
    """Direct llm.chat() calls outside a metered run must not raise."""
    usage.end_run()
    assert usage.current() is None


# ---- The hard stop must bound when work starts, not just how long a call runs -------
def test_wall_exceeded_tracks_the_hard_stop():
    import time
    from agent import llm

    llm.set_wall(None)
    assert llm.wall_exceeded() is False          # no wall configured: nothing to exceed
    llm.set_wall(time.monotonic() + 30)
    assert llm.wall_exceeded() is False
    llm.set_wall(time.monotonic() - 1)
    assert llm.wall_exceeded() is True
    llm.set_wall(None)


def test_a_repair_is_not_attempted_past_the_wall():
    """A repair is a second call, and _call_timeout floors at 5s, so one begun past the
    wall pushes the request further into the platform's kill window."""
    import time
    from agent import agent as A, llm

    calls = []

    def _junk(messages, temperature=0.3, json_mode=False, max_tokens=1200):
        calls.append(1)
        return "not json"

    original = A.chat
    try:
        A.chat = _junk
        llm.set_wall(time.monotonic() + 30)
        A._chat_json([{"role": "system", "content": "x"}], 0.2, 100)
        assert len(calls) == 2                   # inside the wall: one repair attempted

        calls.clear()
        llm.set_wall(time.monotonic() - 1)
        assert A._chat_json([{"role": "system", "content": "x"}], 0.2, 100) is None
        assert len(calls) == 1                   # past it: the caller degrades instead
    finally:
        A.chat = original
        llm.set_wall(None)


def test_the_budget_leaves_room_for_the_platform_limit():
    """The constants are load-bearing: the arithmetic beside them is what keeps a run from
    being killed mid-reply, which returns a 504 no error path here can convert."""
    from agent import agent as A
    from agent.llm import _TIMEOUT_S

    PLATFORM_LIMIT = 300      # vercel.json maxDuration for api/execute.py
    MARGIN = 15               # one slow provider response must not be enough to overrun

    assert A.MAX_RUN_SECONDS < A.HARD_WALL_SECONDS
    # Wall + the 5s per-call floor + cold start, both Supabase round trips and the response.
    spent = A.HARD_WALL_SECONDS + 5 + 3 + 2.5 + 2.5 + 1
    assert spent <= PLATFORM_LIMIT - MARGIN, f"only {PLATFORM_LIMIT - spent}s of margin"
    # A single call must still be able to run to its own ceiling inside the work budget.
    assert A.MAX_RUN_SECONDS >= _TIMEOUT_S
