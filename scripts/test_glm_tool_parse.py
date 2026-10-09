#!/usr/bin/env python3
"""parse_tool_calls must recognise the GLM XML tool-call format.

GLM-5.x emits `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`
(no <tool_sep>, unlike Hy3). Before the GLM pass, the call came back as plain text with
stop_reason end_turn. The other formats must parse exactly as before.

    /Users/sophie/mlx-cluster/.venv/bin/python scripts/test_glm_tool_parse.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import parse_tool_calls  # noqa: E402


class GlmXml(unittest.TestCase):
    def test_single_call(self):
        text = "<tool_call>read_file<arg_key>path</arg_key><arg_value>scripts/ox.py</arg_value></tool_call>"
        calls, content = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(calls[0]["function"]["arguments"], '{"path": "scripts/ox.py"}')
        self.assertEqual(content, "")

    def test_json_valued_argument_is_decoded(self):
        text = "<tool_call>edit<arg_key>lines</arg_key><arg_value>[1, 2]</arg_value></tool_call>"
        calls, _ = parse_tool_calls(text)
        self.assertEqual(calls[0]["function"]["arguments"], '{"lines": [1, 2]}')

    def test_two_calls_and_surrounding_text_kept(self):
        text = ("Je lis le fichier. "
                "<tool_call>read_file<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call>"
                "<tool_call>list_dir<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>")
        calls, content = parse_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["read_file", "list_dir"])
        self.assertEqual(content, "Je lis le fichier.")

    def test_call_without_arguments(self):
        calls, _ = parse_tool_calls("<tool_call>git_status</tool_call>")
        self.assertEqual(calls[0]["function"]["name"], "git_status")
        self.assertEqual(calls[0]["function"]["arguments"], "{}")


class OtherFormatsUnchanged(unittest.TestCase):
    def test_hy3_still_parses_as_hy3(self):
        text = ("<tool_call>read_file<tool_sep><arg_key>path</arg_key>"
                "<arg_value>a.py</arg_value></tool_call>")
        calls, _ = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(calls[0]["function"]["arguments"], '{"path": "a.py"}')

    def test_qwen_xml_unchanged(self):
        text = "<tool_call><function=grep><parameter=pattern>foo</parameter></function></tool_call>"
        calls, _ = parse_tool_calls(text)
        self.assertEqual(calls[0]["function"]["name"], "grep")
        self.assertEqual(calls[0]["function"]["arguments"], '{"pattern": "foo"}')

    def test_hermes_json_unchanged(self):
        text = '<tool_call>{"name": "grep", "arguments": {"pattern": "x"}}</tool_call>'
        calls, _ = parse_tool_calls(text)
        self.assertEqual(calls[0]["function"]["name"], "grep")

    def test_plain_text_has_no_calls(self):
        calls, content = parse_tool_calls("Paris.")
        self.assertEqual(calls, [])
        self.assertEqual(content, "Paris.")


if __name__ == "__main__":
    unittest.main()
