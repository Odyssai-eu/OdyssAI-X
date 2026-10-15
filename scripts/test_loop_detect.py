#!/usr/bin/env python3
"""loop_detect: the one anti-loop shared by runner.py and the mlx_vlm servers (#113).

    python3 scripts/test_loop_detect.py
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import loop_detect as ld  # noqa: E402


def per_token(ids):
    """The runner's cadence: check after every token, small every 16, large every 64."""
    seen = []
    for t in ids:
        seen.append(t)
        n = len(seen)
        if n % ld.ANTI_LOOP_CHECK_EVERY == 0:
            hit = ld.detect_loop(seen)
            if hit is None and n % ld.ANTI_LOOP_L_CHECK_EVERY == 0:
                hit = ld.detect_loop_large(seen)
            if hit:
                return hit, n
    return None, len(seen)


class DetectorTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(113)

    def varied(self, n, vocab=30000):
        return [self.rng.randrange(vocab) for _ in range(n)]

    def test_short_period_loop(self):
        ids = self.varied(300) + [11, 12, 13] * 20
        self.assertEqual(ld.detect_loop(ids)[0], 3)

    def test_paragraph_loop_needs_the_large_detector(self):
        block = self.varied(450)
        ids = self.varied(200) + block * 4
        self.assertIsNone(ld.detect_loop(ids))
        self.assertEqual(ld.detect_loop_large(ids), (450, 4))

    def test_three_repeats_are_not_a_loop(self):
        block = self.varied(120)
        self.assertIsNone(ld.detect_loop_large(self.varied(300) + block * 3))

    def test_varied_output_never_trips(self):
        for _ in range(200):
            ids = self.varied(self.rng.randint(50, 3000), vocab=self.rng.choice([50, 30000]))
            self.assertIsNone(ld.detect_loop(ids))
            self.assertIsNone(ld.detect_loop_large(ids))

    def test_stylistic_run_is_not_a_loop(self):
        # "!!!!!!" — period 1 needs 48 repeats
        self.assertIsNone(ld.detect_loop(self.varied(100) + [7] * 20))


class GuardTest(unittest.TestCase):
    """mlx_vlm may hand several ids per step (token_ids): the guard must stop at the
    same token as the runner's per-token checks."""

    def test_multi_token_steps_match_per_token(self):
        rng = random.Random(7)
        for _ in range(100):
            ids = [rng.randrange(30000) for _ in range(rng.randint(0, 800))]
            p = rng.choice([2, 5, 40, 70, 300])
            ids += [rng.randrange(30000) for _ in range(p)] * rng.randint(4, 30)
            want, at = per_token(ids)
            g, got, i = ld.LoopGuard(), None, 0
            while i < len(ids) and got is None:
                step = rng.randint(1, 4)
                got = g.feed(ids[i:i + step])
                i += step
            self.assertEqual(got, want)
            if want:
                self.assertEqual(len(g.ids), at)

    def test_runner_uses_the_same_functions(self):
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")).read()
        self.assertIn("detect_loop as _detect_loop, detect_loop_large as _detect_loop_large", src)
        self.assertNotIn("def _detect_loop(", src)


if __name__ == "__main__":
    unittest.main()
