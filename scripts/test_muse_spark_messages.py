#!/usr/bin/env python3
"""Muse-Spark-Open messages reach the client as reasoning, content and tool_calls.

The assistant turn is a run of `<|start|>assistant to=RECIPIENT<|message|>BODY<|eom|>`
messages (self = reasoning, user / commentary = text, a tool name = an ATEM call).
Before the fix the commentary and tool headers and the ATEM block reached
`content` as text (Bench A a03, 2026-10-20). MuseRecipientRouter keeps only the
`to=self` header the API splits on; parse_tool_calls reads ATEM.

    /Users/sophie/mlx-cluster/.venv/bin/python scripts/test_muse_spark_messages.py
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import (MuseRecipientRouter, ToolMarkupGate, _string_params,  # noqa: E402
                    parse_tool_calls)

REASONING = "We need to answer: TARGET FILE(S) + FUNCTION(S)."
COMMENTARY = "Your JSON flag needs stdout — not a file. I'll verify the repo layout."
ATEM = ('<atem:function_calls>\n<atem:invoke name="list_dir">\n'
        '<atem:parameter name="path">.</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')
A03_RAW = (f" to=self<|message|>{REASONING}<|eom|>"
           f"<|start|>assistant to=commentary<|message|>{COMMENTARY}<|eom|>"
           f"<|start|>assistant to=list_dir<|message|>{ATEM}")
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "line": {"type": "integer"}}}}}
    for n in ("list_dir", "read_file", "write_file")]
OFFERED = {"list_dir", "read_file", "write_file"}


def route(raw: str, step: int) -> str:
    r = MuseRecipientRouter(True)
    out = "".join(r.feed(raw[i:i + step]) for i in range(0, len(raw), step))
    return out + r.flush()


class Router(unittest.TestCase):
    def test_headers_whatever_the_chunking(self):
        want = f"to=self<|message|>{REASONING}<|eom|>{COMMENTARY}<|eom|>{ATEM}"
        for step in (1, 2, 3, 5, 7, 11, 64, len(A03_RAW)):
            with self.subTest(step=step):
                self.assertEqual(route(A03_RAW, step), want)

    def test_answer_to_user_reads_as_content(self):
        raw = f" to=self<|message|>r<|eom|><|start|>assistant to=user<|message|>Le plan."
        self.assertEqual(route(raw, 4), "to=self<|message|>r<|eom|>Le plan.")

    def test_no_reasoning_first_message_to_user(self):
        self.assertEqual(route(" to=user<|message|>Bonjour", 3), "Bonjour")

    def test_plain_text_and_a_stray_start_pass(self):
        self.assertEqual(route("set x to=5 here", 2), "set x to=5 here")
        self.assertEqual(route("a <|start|> b", 2), "a <|start|> b")

    def test_inactive_router_changes_nothing(self):
        r = MuseRecipientRouter(False)
        self.assertEqual(r.feed(A03_RAW) + r.flush(), A03_RAW)


class Atem(unittest.TestCase):
    def test_a03_call_is_structured_and_leaves_the_content(self):
        calls, content = parse_tool_calls(A03_RAW, OFFERED, _string_params(TOOLS))
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in calls], [("list_dir", {"path": "."})])
        self.assertNotIn("<atem:", content)
        self.assertNotIn("to=list_dir", content)

    def test_values_follow_the_render_atem_encoding(self):
        raw = ('<atem:function_calls>\n<atem:invoke name="write_file">\n'
               '<atem:parameter name="path">007</atem:parameter>\n'
               '<atem:parameter name="line">12</atem:parameter>\n'
               '<atem:parameter name="flags">{"a": [1, true, null]}</atem:parameter>\n'
               '<atem:parameter name="content">x &lt; y</atem:parameter>\n'
               '</atem:invoke>\n</atem:function_calls>')
        calls, _ = parse_tool_calls(raw, OFFERED, _string_params(TOOLS))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"path": "007", "line": 12, "flags": {"a": [1, True, None]},
                          "content": "x &lt; y"})

    def test_call_to_a_tool_not_offered_stays_text(self):
        raw = ('<atem:function_calls><atem:invoke name="rm_rf">'
               '<atem:parameter name="path">/</atem:parameter></atem:invoke></atem:function_calls>')
        calls, content = parse_tool_calls(raw, OFFERED)
        self.assertEqual(calls, [])
        self.assertIn("rm_rf", content)


class Stream(unittest.TestCase):
    def test_router_then_gate_streams_commentary_and_holds_the_call(self):
        router, gate = MuseRecipientRouter(True), ToolMarkupGate(True)
        streamed = "".join(gate.feed(router.feed(A03_RAW[i:i + 6]))
                           for i in range(0, len(A03_RAW), 6))
        streamed += gate.feed(router.flush()) + gate.release(OFFERED)
        self.assertIn(COMMENTARY, streamed)
        self.assertNotIn("to=commentary", streamed)
        self.assertNotIn("to=list_dir", streamed)
        self.assertNotIn("<atem:", streamed)


if __name__ == "__main__":
    unittest.main()
