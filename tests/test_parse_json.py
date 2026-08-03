"""parse_json must be TOTAL (never raise) and extract balanced objects, not greedy spans."""
from agent.llm import parse_json


def test_plain_object():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_object():
    assert parse_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}


def test_balanced_not_greedy():
    # A greedy \{.*\} would grab across both braces and fail; balanced scan takes the first object.
    assert parse_json('{"a": {"b": 2}} trailing {oops') == {"a": {"b": 2}}


def test_brace_inside_string_literal():
    assert parse_json('{"note": "a } brace"}') == {"note": "a } brace"}


def test_unparseable_returns_none():
    assert parse_json("no json here") is None
    assert parse_json('{"truncated": ') is None      # unbalanced -> None, not a crash


def test_never_raises_on_arbitrary_input():
    for junk in ["", "null", "[1,2,3]", "}{", "{{{", "```", 12345, None, "{'single': 'quotes'}"]:
        parse_json(junk)  # must not raise
