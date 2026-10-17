#!/usr/bin/env python3
"""XML tool-call arguments follow the tool's schema (#124).

The XML formats carry every value as text. A parameter typed string in the
offered tool's schema must stay the raw text ("1.10" is a version, "true" a
word); the others are JSON-decoded as before. Same four parsers: Qwen3-Coder,
Hy3, GLM, LongCat.

    /Users/sophie/mlx-cluster/.venv/bin/python scripts/test_tool_arg_types.py
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import _string_params, parse_tool_calls  # noqa: E402

TOOLS_OPENAI = [{"type": "function", "function": {
    "name": "bump",
    "parameters": {"type": "object", "properties": {
        "version": {"type": "string"},
        "flag": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "dry_run": {"type": "boolean"},
        "files": {"type": "array", "items": {"type": "string"}},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "tag": {"type": ["string", "null"]},
    }},
}}]
TOOLS_ANTHROPIC = [{"name": "bump", "input_schema": TOOLS_OPENAI[0]["function"]["parameters"]}]

PAIRS = [("version", "1.10"), ("flag", "true"), ("count", "3"), ("ratio", "1.10"),
         ("dry_run", "true"), ("files", '["a.py", "b.py"]'), ("note", "0"), ("tag", "007"),
         ("extra", "42")]
EXPECTED = {"version": "1.10", "flag": "true", "count": 3, "ratio": 1.1, "dry_run": True,
            "files": ["a.py", "b.py"], "note": "0", "tag": "007", "extra": 42}


def qwen(pairs):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in pairs)
    return f"<tool_call>\n<function=bump>\n{body}</function>\n</tool_call>"


def hy3(pairs):
    body = "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in pairs)
    return f"<tool_call>bump<tool_sep>{body}</tool_call>"


def glm(pairs):
    body = "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in pairs)
    return f"<tool_call>bump{body}</tool_call>"


def longcat(pairs):
    body = "".join(f"<longcat_arg_key>{k}</longcat_arg_key><longcat_arg_value>{v}</longcat_arg_value>"
                   for k, v in pairs)
    return f"<longcat_tool_call>bump\n{body}</longcat_tool_call>"


class SchemaTyping(unittest.TestCase):
    def test_string_params_openai_and_anthropic(self):
        want = {"bump": {"version", "flag", "note", "tag"}}
        self.assertEqual(_string_params(TOOLS_OPENAI), want)
        self.assertEqual(_string_params(TOOLS_ANTHROPIC), want)
        self.assertEqual(_string_params(None), {})

    def test_every_xml_format_follows_the_schema(self):
        for fmt in (qwen, hy3, glm, longcat):
            with self.subTest(fmt=fmt.__name__):
                calls, content = parse_tool_calls(fmt(PAIRS), {"bump"}, _string_params(TOOLS_OPENAI))
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), EXPECTED)
                self.assertEqual(content, "")

    def test_without_schema_values_are_decoded_as_before(self):
        calls, _ = parse_tool_calls(glm([("version", "1.10"), ("flag", "true")]))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"version": 1.1, "flag": True})


if __name__ == "__main__":
    unittest.main()
