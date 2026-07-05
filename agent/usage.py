"""Token accounting for every LLM call, plus the per-run call ceiling.

A single confirmed turn can make many model calls, so calls have to be observable per
module and stoppable mid-run rather than reconciled afterwards.

Two layers:
  1. `RunMeter` counts tokens reported by the provider on each response (no estimation).
  2. `MAX_CALLS_PER_RUN` is a hard per-request ceiling — the real runaway protection.

There is deliberately no cost accounting here. It needed per-token rates as configuration,
and with those unset — which is how it ran — every figure it produced was zero and the
spend ceiling it fed could never fire. A guard that silently does nothing is worse than no
guard, because it reads like protection. The call ceiling works without knowing any rate.
"""
import os

# Hard ceiling on LLM calls in one /api/execute request, and what actually stops a
# misbehaving loop. Sized just above the worst legitimate turn: a confirmed turn runs
# intake + up to 6 planner turns + finalize + critic + a fix + the formatter, which lands
# on 12, so 15 leaves the fix cycle room to complete.
MAX_CALLS_PER_RUN = int(os.environ.get("LLM_MAX_CALLS_PER_RUN") or 15)


class CallLimitExceeded(RuntimeError):
    """Raised instead of starting an LLM call that would breach the per-run ceiling.

    The orchestrator lets this propagate so the turn ends with a clear error rather than
    quietly running past the limit.
    """


class RunMeter:
    """Accounting for one /api/execute request."""

    def __init__(self, run_id=None, max_calls=None):
        self.run_id = run_id
        self.max_calls = MAX_CALLS_PER_RUN if max_calls is None else max_calls
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.by_module = {}       # module name -> {"calls", "prompt_tokens", ...}
        self.module = None        # set by the orchestrator so tokens attribute correctly

    def before_call(self):
        """Gate checked immediately before every LLM call. Raises CallLimitExceeded."""
        if self.max_calls and self.calls >= self.max_calls:
            raise CallLimitExceeded(
                f"per-run LLM call ceiling reached ({self.calls}/{self.max_calls})")

    def record(self, prompt_tokens, completion_tokens):
        """Fold one response's reported usage into the run totals."""
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)

        self.calls += 1
        self.prompt_tokens += p
        self.completion_tokens += c

        slot = self.by_module.setdefault(
            self.module or "unattributed",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        slot["calls"] += 1
        slot["prompt_tokens"] += p
        slot["completion_tokens"] += c

    def snapshot(self):
        """Serialisable summary of the run. `calls` is what is logged; the rest is
        available for the ceiling check and for tests."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
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
    """Attribute subsequent calls to a pipeline module, so usage breaks down per module."""
    if _current is not None:
        _current.module = name
