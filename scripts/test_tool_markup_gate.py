#!/usr/bin/env python3
"""ToolMarkupGate keeps tool-call markup out of the streamed text.

In stream mode the tokens of a call block went out as content, and the parsed
tool_calls followed in the done event: a client got both. The gate holds the
text back from the first opener, then releases it without the accepted blocks.
What the client receives in stream must equal the non-stream content.

    /Users/sophie/mlx-cluster/.venv/bin/python scripts/test_tool_markup_gate.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import ToolMarkupGate, _gated_event, parse_tool_calls  # noqa: E402

GLM_CALL = "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/notes.txt</arg_value></tool_call>"
QWEN_CALL = ("<tool_call><function=read_file><parameter=path>/tmp/notes.txt</parameter>"
             "</function></tool_call>")
OFFERED = {"read_file", "list_dir"}


def stream(text: str, size: int, active: bool = True, names=OFFERED) -> str:
    """Feed `text` in chunks of `size` characters, then release."""
    gate = ToolMarkupGate(active)
    out = "".join(gate.feed(text[i:i + size]) for i in range(0, len(text), size))
    return out + gate.release(names)


class Gate(unittest.TestCase):
    def test_call_never_streams_whatever_the_chunking(self):
        text = "I will read it.\n" + GLM_CALL
        for size in range(1, len(text) + 1):
            gate = ToolMarkupGate(True)
            sent = [gate.feed(text[i:i + size]) for i in range(0, len(text), size)]
            self.assertNotIn("<tool", "".join(sent), size)
            self.assertNotIn("<", "".join(sent), size)
            self.assertEqual("".join(sent) + gate.release(OFFERED), "I will read it.\n", size)

    def test_stream_equals_non_stream_content(self):
        for call in (GLM_CALL, QWEN_CALL, GLM_CALL + "\n" + GLM_CALL):
            text = "Reading.\n\n" + call
            _, content = parse_tool_calls(text, OFFERED)
            for size in (1, 3, 7, 64):
                self.assertEqual(stream(text, size).strip(), content)

    def test_text_before_the_opener_streams_live(self):
        gate = ToolMarkupGate(True)
        self.assertEqual(gate.feed("Hello there, "), "Hello there, ")
        self.assertEqual(gate.feed("I will check <to"), "I will check ")
        self.assertEqual(gate.feed("ol_call>read_file"), "")

    def test_split_prefix_that_is_not_an_opener_is_released(self):
        gate = ToolMarkupGate(True)
        self.assertEqual(gate.feed("a <tool"), "a ")
        self.assertEqual(gate.feed("box> b"), "<toolbox> b")

    def test_prose_quoting_markup_is_kept(self):
        text = "Use <tool_call> tags to call a tool."
        self.assertEqual(stream(text, 4), text)

    def test_call_to_a_name_not_offered_stays_text(self):
        text = "Example: " + GLM_CALL.replace("read_file", "rm_rf")
        self.assertEqual(stream(text, 5), text)

    def test_block_cut_by_max_tokens_stays_text(self):
        text = "Ok. <tool_call>read_file<arg_key>path</arg_key><arg_val"
        self.assertEqual(stream(text, 6), text)

    def test_text_after_the_call_is_released(self):
        text = GLM_CALL + "\nDone."
        self.assertEqual(stream(text, 9), "Done.")

    def test_quoted_unoffered_call_next_to_a_real_call_survives(self):
        quoted = GLM_CALL.replace("read_file", "rm_rf")
        text = "See " + quoted + " then " + GLM_CALL
        out = stream(text, 11)
        self.assertIn(quoted, out)
        self.assertNotIn("/tmp/notes.txt</arg_value></tool_call>" * 2, out)
        self.assertEqual(out.count("<tool_call>"), 1)

    def test_longcat_opener_is_held(self):
        gate = ToolMarkupGate(True)
        self.assertEqual(gate.feed("x <longcat_tool"), "x ")
        self.assertEqual(gate.feed("_call>read_file"), "")

    def test_inactive_gate_passes_everything(self):
        text = "Raw " + GLM_CALL
        self.assertEqual(stream(text, 3, active=False), text)

    def test_trailing_partial_prefix_is_flushed_at_release(self):
        self.assertEqual(stream("ends with <tool", 4), "ends with <tool")


class GatedEventTest(unittest.TestCase):
    """#112: a batch the gate holds back still says the run advanced."""

    def test_text_before_a_call_is_a_token(self):
        ev = _gated_event(ToolMarkupGate(True), "Je lis le fichier. ", "r1", 5)
        self.assertEqual((ev["event"], ev["text"]), ("token", "Je lis le fichier. "))

    def test_held_batch_is_progress_without_text(self):
        gate = ToolMarkupGate(True)
        _gated_event(gate, "Je lis. ", "r1", 3)
        ev = _gated_event(gate, "<tool_call>read_file<arg_key>path</arg_key>", "r1", 15)
        self.assertEqual(ev, {"event": "progress", "id": "r1", "phase": "decode", "ntoks": 15})

    def test_every_batch_after_the_opener_is_progress(self):
        gate = ToolMarkupGate(True)
        evs = [_gated_event(gate, GLM_CALL[i:i + 7], "r1", i) for i in range(0, len(GLM_CALL), 7)]
        self.assertTrue(all(e["event"] == "progress" for e in evs))
        self.assertEqual(gate.release(OFFERED), "")

    def test_no_tools_no_gate(self):
        self.assertEqual(_gated_event(ToolMarkupGate(False), "<tool_call>x", "r2", 3)["event"], "token")


if __name__ == "__main__":
    unittest.main()
