"""Tests for app.utils.json_repair module."""

import json
import pytest
from app.utils.json_repair import repair_json, safe_json_loads, extract_json_object


class TestRepairJson:
    def test_valid_json_unchanged(self):
        s = '{"a": 1, "b": "hello"}'
        assert json.loads(repair_json(s)) == {"a": 1, "b": "hello"}

    def test_empty_string(self):
        assert repair_json("") == ""

    def test_trailing_comma_object(self):
        result = repair_json('{"a": 1, "b": 2,}')
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_trailing_comma_array(self):
        result = repair_json('{"items": [1, 2, 3,]}')
        assert json.loads(result) == {"items": [1, 2, 3]}

    def test_trailing_comma_nested(self):
        result = repair_json('{"a": {"b": 1,}, "c": [1,],}')
        assert json.loads(result) == {"a": {"b": 1}, "c": [1]}

    def test_single_quoted_keys(self):
        result = repair_json("{'name': 'test', 'value': 'ok'}")
        assert json.loads(result) == {"name": "test", "value": "ok"}

    def test_extract_from_code_fence(self):
        text = 'Here is the JSON:\n```json\n{"key": "value"}\n```\nDone.'
        result = repair_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_extract_from_code_fence_no_lang(self):
        text = '```\n{"key": 1}\n```'
        result = repair_json(text)
        assert json.loads(result) == {"key": 1}

    def test_extract_first_json_object(self):
        text = 'Some preamble {"a": 1} trailing stuff'
        result = repair_json(text)
        assert json.loads(result) == {"a": 1}

    def test_control_characters_removed(self):
        s = '{"text": "hello\x00world"}'
        result = repair_json(s)
        assert json.loads(result) == {"text": "helloworld"}

    def test_line_comments_removed(self):
        s = '{"a": 1, // this is a comment\n"b": 2}'
        result = repair_json(s)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_block_comments_removed(self):
        s = '{"a": 1, /* comment */ "b": 2}'
        result = repair_json(s)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_unquoted_string_values(self):
        s = '{"type": task, "status": completed}'
        result = repair_json(s)
        assert json.loads(result) == {"type": "task", "status": "completed"}

    def test_python_true_false_none(self):
        s = '{"a": True, "b": False, "c": None}'
        result = repair_json(s)
        assert json.loads(result) == {"a": True, "b": False, "c": None}

    def test_combined_issues(self):
        s = "{'type': task, 'done': True,}"
        result = repair_json(s)
        parsed = json.loads(result)
        assert parsed["type"] == "task"
        assert parsed["done"] is True

    def test_numeric_values_not_quoted(self):
        s = '{"count": 42, "rate": 3.14}'
        result = repair_json(s)
        parsed = json.loads(result)
        assert parsed["count"] == 42
        assert parsed["rate"] == 3.14

    def test_nested_braces(self):
        s = 'prefix {"a": {"b": {"c": 1}}} suffix'
        result = repair_json(s)
        assert json.loads(result) == {"a": {"b": {"c": 1}}}

    def test_unquoted_values_preserve_json_literals(self):
        """true, false, null should NOT be quoted."""
        s = '{"a": true, "b": false, "c": null}'
        result = repair_json(s)
        parsed = json.loads(result)
        assert parsed["a"] is True
        assert parsed["b"] is False
        assert parsed["c"] is None


class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_empty_string_returns_default(self):
        assert safe_json_loads("") is None
        assert safe_json_loads("", default={}) == {}

    def test_none_returns_default(self):
        assert safe_json_loads(None) is None

    def test_repairs_then_parses(self):
        assert safe_json_loads('{"a": 1,}') == {"a": 1}

    def test_unparseable_returns_default(self):
        assert safe_json_loads("not json at all") is None
        assert safe_json_loads("not json", default="fallback") == "fallback"

    def test_array(self):
        assert safe_json_loads("[1, 2, 3]") == [1, 2, 3]

    def test_code_fence_repair(self):
        text = '```json\n{"x": True,}\n```'
        assert safe_json_loads(text) == {"x": True}


class TestExtractJsonObject:
    def test_none_input(self):
        assert extract_json_object(None) is None

    def test_empty_string(self):
        assert extract_json_object("") is None

    def test_no_json(self):
        assert extract_json_object("just some text") is None

    def test_simple_object(self):
        result = extract_json_object('the result is {"a": 1} ok')
        assert json.loads(result) == {"a": 1}

    def test_nested_object(self):
        result = extract_json_object('output: {"a": {"b": 2}}')
        assert json.loads(result) == {"a": {"b": 2}}

    def test_code_fence(self):
        text = 'Here:\n```json\n{"key": "val"}\n```'
        result = extract_json_object(text)
        assert json.loads(result) == {"key": "val"}

    def test_string_with_braces(self):
        obj = '{"text": "use { and } carefully"}'
        text = "prefix " + obj + " suffix"
        result = extract_json_object(text)
        assert json.loads(result) == {"text": "use { and } carefully"}

    def test_escaped_quotes(self):
        obj = '{"text": "say \\"hello\\""}'
        text = "data: " + obj
        result = extract_json_object(text)
        assert json.loads(result)["text"] == 'say "hello"'

    def test_unmatched_braces_returns_none(self):
        assert extract_json_object("start { but never close") is None
