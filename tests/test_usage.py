"""Budget metering and the ceilings that protect the project's $9 allowance.

These guards are the reason a runaway loop cannot quietly drain the course budget, so they
need to hold under test rather than being trusted by inspection.
"""
import pytest

from agent import usage


@pytest.fixture(autouse=True)
def _clean():
    usage.reset_process_totals()
    yield
    usage.end_run()
    usage.reset_process_totals()


def test_tokens_are_counted_from_the_provider_not_estimated():
    m = usage.start_run("r1")
    m.record(1000, 500)
    m.record(250, 100)
    snap = m.snapshot()
    assert snap["calls"] == 2
    assert snap["prompt_tokens"] == 1250 and snap["completion_tokens"] == 600
    assert snap["total_tokens"] == 1850


def test_cost_uses_configured_rates(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_IN_PER_1M", 0.15)
    monkeypatch.setattr(usage, "PRICE_OUT_PER_1M", 0.60)
    m = usage.start_run("r2")
    m.record(1_000_000, 1_000_000)
    assert m.snapshot()["cost_usd"] == pytest.approx(0.75)


def test_unpriced_model_still_counts_tokens(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_IN_PER_1M", 0.0)
    monkeypatch.setattr(usage, "PRICE_OUT_PER_1M", 0.0)
    m = usage.start_run("r3")
    m.record(5000, 2000)
    snap = m.snapshot()
    assert snap["cost_usd"] == 0.0 and snap["total_tokens"] == 7000


def test_replayed_calls_count_tokens_but_cost_nothing(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_IN_PER_1M", 1.0)
    monkeypatch.setattr(usage, "PRICE_OUT_PER_1M", 1.0)
    m = usage.start_run("r4")
    m.record(1_000_000, 0, billed=False)     # served from the cassette
    assert m.snapshot()["prompt_tokens"] == 1_000_000
    assert m.snapshot()["cost_usd"] == 0.0


def test_per_run_call_ceiling_stops_a_runaway_loop():
    m = usage.start_run("r5", max_calls=3)
    for _ in range(3):
        m.before_call()
        m.record(10, 10)
    with pytest.raises(usage.BudgetExceeded) as e:
        m.before_call()
    assert "call ceiling" in str(e.value)


def test_budget_ceiling_stops_spending(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_IN_PER_1M", 1000.0)   # deliberately expensive
    monkeypatch.setattr(usage, "PRICE_OUT_PER_1M", 0.0)
    monkeypatch.setattr(usage, "BUDGET_USD", 0.5)
    m = usage.start_run("r6", max_calls=99)
    m.before_call()
    m.record(1_000_000, 0)                                   # $1.00 spent, over the $0.50 cap
    with pytest.raises(usage.BudgetExceeded) as e:
        m.before_call()
    assert "budget ceiling" in str(e.value)


def test_cost_attributes_to_the_module_that_spent_it(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_IN_PER_1M", 1.0)
    monkeypatch.setattr(usage, "PRICE_OUT_PER_1M", 1.0)
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


def test_process_totals_accumulate_across_runs(monkeypatch):
    monkeypatch.setattr(usage, "PRICE_IN_PER_1M", 1.0)
    monkeypatch.setattr(usage, "PRICE_OUT_PER_1M", 0.0)
    for _ in range(3):
        m = usage.start_run()
        m.record(1_000_000, 0)
        usage.end_run()
    assert usage.process_totals()["cost_usd"] == pytest.approx(3.0)
