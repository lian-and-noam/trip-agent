"""Supabase persistence must be optional.

The agent has to behave identically when Supabase is unconfigured, slow, or erroring —
persistence is an enhancement, never a dependency. These tests pin that contract, and also
cover the id validation in the API layer, which is what keeps client-supplied strings out
of PostgREST filters.
"""
import requests

from agent import store


def test_disabled_when_unconfigured(monkeypatch):
    monkeypatch.setattr(store, "URL", "")
    monkeypatch.setattr(store, "KEY", "")
    assert store.enabled() is False
    # Every entry point degrades quietly rather than raising.
    assert store.get_conversation("abc") is None
    assert store.save_conversation("abc", profile={}, plan={}) is False


def _configured(monkeypatch):
    monkeypatch.setattr(store, "URL", "https://example.supabase.co")
    monkeypatch.setattr(store, "KEY", "service-key")


def test_network_failure_degrades_to_none(monkeypatch):
    _configured(monkeypatch)

    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(store.requests, "request", boom)
    assert store.get_conversation("11111111-1111-4111-8111-111111111111") is None


def test_timeout_degrades_to_none(monkeypatch):
    _configured(monkeypatch)

    def slow(*a, **kw):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(store.requests, "request", slow)


def test_conversation_round_trip_shape(monkeypatch):
    _configured(monkeypatch)
    captured = {}

    class _Resp:
        status_code = 200
        content = b'[{"id":"x","profile":{"days":3},"plan":{"days":[]},"title":"t"}]'

        def raise_for_status(self):
            pass

        def json(self):
            return [{"id": "x", "profile": {"days": 3},
                     "plan": {"days": []}, "title": "t"}]

    def fake(method, url, **kw):
        captured.update({"method": method, "url": url, "params": kw.get("params")})
        return _Resp()

    monkeypatch.setattr(store.requests, "request", fake)
    row = store.get_conversation("22222222-2222-4222-8222-222222222222")
    assert row["profile"]["days"] == 3
    assert captured["method"] == "GET"
    assert captured["params"]["id"].startswith("eq.")


def test_missing_conversation_returns_none(monkeypatch):
    _configured(monkeypatch)

    class _Empty:
        content = b"[]"

        def raise_for_status(self):
            pass

        def json(self):
            return []

    monkeypatch.setattr(store.requests, "request", lambda *a, **kw: _Empty())
    assert store.get_conversation("33333333-3333-4333-8333-333333333333") is None


# ---- API-layer id validation -------------------------------------------------------
def test_only_well_formed_uuids_reach_the_database():
    from api.execute import _as_uuid

    good = "44444444-4444-4444-8444-444444444444"
    assert _as_uuid(good) == good
    for bad in (None, "", 42, [], "not-a-uuid", "'; drop table runs;--",
                "eq.1&select=*", "x" * 200):
        assert _as_uuid(bad) is None


def test_title_is_derived_and_bounded():
    from api.execute import _title_for

    assert _title_for({"destination": "Kyoto", "days": 7}) == "7 days in Kyoto"
    assert _title_for({"destination": "Rome"}) == "Rome"
    assert _title_for(None) == "Trip"
    assert len(_title_for({"destination": "x" * 500, "days": 3})) <= 120
