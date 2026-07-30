"""Token and cost accounting for every LLM call, plus the guards that protect the budget.

The project has a hard $9 ceiling across the whole course, and a single confirmed turn can
make ~9 model calls, so spend has to be observable per call and stoppable mid-run rather
than reconciled afterwards.

Three layers, cheapest first:
  1. `RunMeter` counts tokens reported by the provider on each response (no estimation).
  2. `MAX_CALLS_PER_RUN` is a hard per-request ceiling that works even when prices are
     unknown — it is the real runaway protection.
  3. `LLM_BUDGET_USD` stops a run once cumulative spend for the process passes a limit.

Prices are configuration, not constants: LLMod fronts several models and the rate depends
on which one is selected, so nothing is hard-coded. With prices unset the meter still
counts tokens and still enforces the call ceiling — it just reports $0.
"""
import os
import threading

# Per-1M-token rates for the configured model. Left at 0 the meter counts tokens only.
PRICE_IN_PER_1M = float(os.environ.get("LLM_PRICE_IN_PER_1M") or 0)
PRICE_OUT_PER_1M = float(os.environ.get("LLM_PRICE_OUT_PER_1M") or 0)

# Hard ceiling on LLM calls in one /api/execute request. Works with or without prices and
# is what actually stops a misbehaving loop. Sized just above the worst legitimate turn.
MAX_CALLS_PER_RUN = int(os.environ.get("LLM_MAX_CALLS_PER_RUN") or 12)

# Cumulative spend ceiling for this process. 0 disables the check.
BUDGET_USD = float(os.environ.get("LLM_BUDGET_USD") or 0)

_lock = threading.Lock()
_process_cost_usd = 0.0      # accumulates across runs served by this warm instance
_process_calls = 0


class BudgetExceeded(RuntimeError):
    """Raised instead of starting an LLM call that would breach a ceiling.

    The orchestrator lets this propagate so the turn ends with a clear error rather than
    quietly spending past the limit.
    """


def _cost_usd(prompt_tokens, completion_tokens):
    return (prompt_tokens * PRICE_IN_PER_1M + completion_tokens * PRICE_OUT_PER_1M) / 1_000_000


class RunMeter:
    """Accounting for one /api/execute request."""

    def __init__(self, run_id=None, max_calls=None):
        self.run_id = run_id
        self.max_calls = MAX_CALLS_PER_RUN if max_calls is None else max_calls
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.by_module = {}       # module name -> {"calls", "prompt_tokens", ...}
        self.module = None        # set by the orchestrator so costs attribute correctly

    def before_call(self):
        """Gate checked immediately before every LLM call. Raises BudgetExceeded."""
        if self.max_calls and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"per-run LLM call ceiling reached ({self.calls}/{self.max_calls})")
        if BUDGET_USD and _process_cost_usd + self.cost_usd >= BUDGET_USD:
            raise BudgetExceeded(
                f"budget ceiling reached (${_process_cost_usd + self.cost_usd:.4f} "
                f"of ${BUDGET_USD:.2f})")

    def record(self, prompt_tokens, completion_tokens, billed=True):
        """Fold one response's reported usage into the run and process totals.

        `billed=False` is used for cassette replays: the tokens are real and worth
        reporting (they show what the run *would* cost), but nothing was spent.
        """
        global _process_cost_usd, _process_calls
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        cost = _cost_usd(p, c) if billed else 0.0

        self.calls += 1
        self.prompt_tokens += p
        self.completion_tokens += c
        self.cost_usd += cost

        slot = self.by_module.setdefault(
            self.module or "unattributed",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        slot["calls"] += 1
        slot["prompt_tokens"] += p
        slot["completion_tokens"] += c
        slot["cost_usd"] = round(slot["cost_usd"] + cost, 6)

        with _lock:
            _process_cost_usd += cost
            _process_calls += 1
        return cost

    def snapshot(self):
        """Serialisable summary, persisted to the `runs` table and emitted to logs."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "by_module": self.by_module,
        }


# A module-level "current run" keeps chat() out of the business of threading a meter
# through every call signature. One request per process on Vercel makes this safe; tests
# call start_run()/end_run() explicitly.
_current = None


def start_run(run_id=None, max_calls=None):
    global _current
    _current = RunMeter(run_id, max_calls)
    return _current


def current():
    """The active meter, or None when nothing is being metered (direct llm.chat calls)."""
    return _current


def end_run():
    global _current
    meter, _current = _current, None
    return meter


def set_module(name):
    """Attribute subsequent calls to a pipeline module, so cost breaks down per module."""
    if _current is not None:
        _current.module = name


def process_totals():
    """Cumulative spend for this warm instance. Useful for a quick local sanity check."""
    return {"calls": _process_calls, "cost_usd": round(_process_cost_usd, 6),
            "budget_usd": BUDGET_USD}


def reset_process_totals():
    """Test hook only."""
    global _process_cost_usd, _process_calls
    with _lock:
        _process_cost_usd = 0.0
        _process_calls = 0
