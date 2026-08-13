"""Browser tests for index.html.

These exist because the UI is a single hand-edited HTML file with no build step, so nothing
else catches a template that silently stops rendering: a syntax check passes, the Python
suite passes, and the broken element simply never appears. Every assertion here is about
behaviour a user would notice.

Skipped automatically when Playwright or its browser binary is unavailable, so the suite
still runs in a bare environment:

    pip install playwright && playwright install chromium
"""
import pathlib

import pytest

playwright_api = pytest.importorskip("playwright.sync_api",
                                     reason="playwright not installed")

INDEX = pathlib.Path(__file__).resolve().parent.parent / "index.html"

SAMPLE_PLAN = """> ⚠️ **Delivered with caveats** — this plan was not fully validated:
> - Day 1 arrival time may be the departure time, not local arrival.
> - Hotel check-in is assumed at 11:15 but is often 14:00.
> ℹ️ **Good to know before you go** — not problems with the plan:
> - Timed castle tours must be booked in advance.
> - Allow 2-3 hours for the airport on an international flight.

# Prague — 1-day itinerary

## Day 1 — Viewpoints
- **12:30 · [Letná Park](https://www.google.com/maps/search/?api=1&query=Letna)** — 90 min · €18
- **15:30 · [Karlštejn Castle](https://maps.example/c)** — 150 min · €20 · [official site](https://example.com/s)

_Day total: €38_

**Total: €257 per person**
"""

INTAKE_STEPS = [
    {"module": "Conversational Intake", "prompt": {},
     "response": {"profile": {"destination": "Prague", "days": 2}, "missing": ["group", "budget"]}},
]
PLANNING_STEPS = INTAKE_STEPS + [
    {"module": "ReAct Planner", "prompt": {}, "response": {}},
    {"module": "Output Formatter", "prompt": {}, "response": {}},
]


@pytest.fixture
def page():
    try:
        with playwright_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:                      # binary not installed
                pytest.skip(f"chromium unavailable: {e}")
            pg = browser.new_page(viewport={"width": 1400, "height": 950})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(INDEX.as_uri())
            pg.wait_for_timeout(300)
            pg.errors = errors
            yield pg
            browser.close()
    except Exception as e:
        pytest.skip(f"playwright unavailable: {e}")


def _render(pg, text, steps):
    pg.evaluate("""([txt, s]) => {
        history.push({ role: 'agent', text: txt });
        renderChat(); renderSteps(s); updateProgress(s);
    }""", [text, steps])
    pg.wait_for_timeout(200)


def test_caveats_render_as_two_separate_bullet_lists(page):
    """Warnings and advisories are distinct: one is a defect, the other is a travel tip.
    Flattened into one block they read as an equally alarming wall."""
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    assert page.eval_on_selector_all(".border-amber-300 li", "e => e.length") == 2
    assert page.eval_on_selector_all(".border-sky-300 li", "e => e.length") == 2


def test_intake_tracker_retires_once_planning_starts(page):
    _render(page, "Here's your trip so far…", INTAKE_STEPS)
    assert page.is_visible("#intakeProgress")
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    assert not page.is_visible("#intakeProgress")


def test_new_trip_brings_the_tracker_back(page):
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    page.evaluate("resetChat()")
    assert page.is_visible("#intakeProgress")


def test_currency_switch_converts_and_round_trips(page):
    """The EUR original is kept per text node, so switching back and forth must not
    compound the conversion."""
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    assert "€257" in page.inner_text("#chat")
    page.click("#curToggle")
    page.click(".cur-btn[data-cur='ILS']")
    page.wait_for_timeout(150)
    converted = page.inner_text("#chat")
    assert "₪" in converted and "€257" not in converted
    page.click("#curToggle")
    page.click(".cur-btn[data-cur='EUR']")
    page.wait_for_timeout(150)
    assert "€257" in page.inner_text("#chat")


def test_currency_control_survives_the_tracker_hiding(page):
    """It moved into the header precisely so it outlives the intake bar."""
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    assert not page.is_visible("#intakeProgress")
    assert page.is_visible("#curToggle")


def test_loader_lives_in_the_conversation_and_cleans_up(page):
    """It is appended to #chat so it appears where the reply will and scrolls with the
    thread, rather than being pinned above the composer."""
    page.evaluate("() => { history.push({ role:'user', text:'hi' }); renderChat(); }")
    page.evaluate("startLoader()")
    page.wait_for_timeout(150)
    assert page.evaluate("!!document.querySelector('#chat #flightLoader')")
    # Starting twice must not leave two planes in the thread.
    page.evaluate("startLoader()")
    page.wait_for_timeout(100)
    assert page.eval_on_selector_all("#flightLoader", "e => e.length") == 1
    page.evaluate("stopLoader()")
    page.wait_for_timeout(100)
    assert page.eval_on_selector_all("#flightLoader", "e => e.length") == 0
    assert page.evaluate("loaderTimer === null")      # no leaked interval


def test_loader_claims_nothing_about_what_the_agent_is_doing(page):
    """The response is not streamed, so any phase label would be invention. Only the
    elapsed counter — which is real — is shown."""
    page.evaluate("() => { history.push({ role:'user', text:'hi' }); renderChat(); }")
    page.evaluate("startLoader()")
    page.wait_for_timeout(150)
    sub = page.inner_text("#loaderSub").lower()
    assert sub.endswith("s") and any(c.isdigit() for c in sub)
    for claim in ("weather", "route", "reviewing", "sketching", "balancing"):
        assert claim not in sub
    page.evaluate("stopLoader()")


def test_plane_rides_on_the_ring_not_inside_it(page):
    """The orbit radius must equal the ring radius; it was 26px inside a 32px ring."""
    page.evaluate("() => { history.push({ role:'user', text:'hi' }); renderChat(); }")
    page.evaluate("startLoader()")
    page.wait_for_timeout(300)
    geo = page.evaluate("""() => {
        const box = document.querySelector('.plane-dial').getBoundingClientRect();
        const pl  = document.querySelector('.plane-nose').getBoundingClientRect();
        const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
        return { r: box.width / 2,
                 d: Math.hypot(pl.left + pl.width/2 - cx, pl.top + pl.height/2 - cy) };
    }""")
    assert abs(geo["r"] - geo["d"]) < 1.5, geo
    page.evaluate("stopLoader()")


def test_no_javascript_errors(page):
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    page.click(".expand-btn")
    page.wait_for_timeout(200)
    assert page.errors == []


def test_currency_button_shows_the_active_symbol(page):
    """It showed a fixed exchange glyph regardless of the selected currency."""
    assert page.inner_text("#curSymbol") == "€"
    page.click("#curToggle")
    page.click(".cur-btn[data-cur='ILS']")
    page.wait_for_timeout(150)
    assert page.inner_text("#curSymbol") == "₪"
    assert page.inner_text("#curLabel") == "ILS"


def test_partial_rate_response_does_not_convert_at_one_to_one(page):
    """Regression: rates were replaced wholesale, so a response missing a symbol left that
    rate undefined and `|| 1` displayed the EUR figure under a different sign."""
    page.evaluate("""async () => {
        window.fetch = async () => ({ ok: true, json: async () => ({
            date: '2026-08-11', rates: { USD: 1.09 } }) });   // ILS and GBP missing
        await loadRates();
    }""")
    assert page.evaluate("rates.ILS") > 3          # fallback retained, not clobbered
    assert page.evaluate("ratesDate === null")     # and not advertised as live


def test_long_bare_urls_stay_inside_the_bubble(page):
    long_url = ("https://www.google.com/maps/dir/?api=1&origin=Top+up+public+transport"
                "+%2F+buy+day+tickets+Prague&destination=Dinner+%28Old+Town+area%29+Prague"
                "&waypoints=Popular+spot+stroll+and+photos+Prague")
    _render(page, f"Route: {long_url}", PLANNING_STEPS)
    fits = page.evaluate("""() => {
        const md = document.querySelector('.md');
        return md.scrollWidth <= md.clientWidth + 1;
    }""")
    assert fits


def test_large_view_can_be_resized(page):
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    page.click(".expand-btn")
    page.wait_for_timeout(250)
    before = page.evaluate("document.getElementById('bigView').getBoundingClientRect().width")
    page.click("#bigView button[aria-label=Larger]")
    page.wait_for_timeout(200)
    assert page.evaluate("document.getElementById('bigView').getBoundingClientRect().width") > before
    assert page.evaluate("getComputedStyle(document.getElementById('bigView')).resize") == "both"


def test_currency_menu_paints_above_the_trace_panel(page):
    page.evaluate("""() => renderSteps([{module:'Conversational Intake',prompt:{},response:{}}])""")
    page.click("#curToggle")
    page.wait_for_timeout(200)
    topmost = page.evaluate("""() => {
        const m = document.getElementById('curMenu').getBoundingClientRect();
        const el = document.elementFromPoint(m.left + m.width/2, m.top + m.height/2);
        return el && el.closest('#curMenu') ? 'curMenu' : 'other';
    }""")
    assert topmost == "curMenu"


def test_markdown_and_banner_lists_show_their_bullets(page):
    """Tailwind's preflight sets `ul { list-style:none }`. The .md rules restored padding but
    not the marker, so itinerary bullets rendered as invisible ones — indented text with
    nothing beside it."""
    _render(page, SAMPLE_PLAN, PLANNING_STEPS)
    # marked is loaded from a CDN, so inject the markup directly rather than depending on it.
    styles = page.evaluate("""() => {
        const md = document.querySelector('.md');
        md.innerHTML = '<ul><li>one</li><li>two</li></ul>';
        const ul = md.querySelector('ul'), li = md.querySelector('li');
        const bl = document.querySelector('.banner-list');
        return {
            mdType: getComputedStyle(ul).listStyleType,
            mdDisplay: getComputedStyle(li).display,
            bannerType: bl ? getComputedStyle(bl).listStyleType : null,
            bannerDisplay: bl ? getComputedStyle(bl.querySelector('li')).display : null,
        };
    }""")
    assert styles["mdType"] == "disc" and styles["mdDisplay"] == "list-item"
    assert styles["bannerType"] == "disc" and styles["bannerDisplay"] == "list-item"
