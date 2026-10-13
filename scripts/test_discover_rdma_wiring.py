#!/usr/bin/env python3
"""discover-rdma-wiring: MAC normalization and the ifconfig/ndp parser.

The 2026-10-13 case: `ndp -an` prints ultra-96's MACs without leading zeros
(`36:74:32:fb:47:8`, `36:74:32:fb:47:c`), ifconfig prints them in full. Before
the fix, the 17-character filter dropped them and the mesh showed 4/6 edges.

    python3 scripts/test_discover_rdma_wiring.py
"""
import importlib.util
import os
import sys
import unittest

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "discover_rdma_wiring", os.path.join(_here, "discover-rdma-wiring.py"))
drw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = drw  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(drw)

# What `collect_iface_data` gets back from .42 (ultra-96b) on en4, verbatim shape.
DUMP_42 = """--IFCONFIG en4--
en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	ether 36:b4:ed:8c:ed:48
	inet6 fe80::829:ffb:156d:cfe3%en4 prefixlen 64 secured scopeid 0x10
--NDP--
Neighbor                                Linklayer Address  Netif Expire    St Flgs Prbs
fe80::5f:d5da:e339:5065%en4             36:74:32:fb:47:8     en4 5s        D
fe80::829:ffb:156d:cfe3%en4             36:b4:ed:8c:ed:48    en4 permanent R
"""


class NormalizeMac(unittest.TestCase):
    def test_pads_short_octets(self):
        self.assertEqual(drw.normalize_mac("36:74:32:fb:47:8"), "36:74:32:fb:47:08")
        self.assertEqual(drw.normalize_mac("36:74:32:fb:47:c"), "36:74:32:fb:47:0c")

    def test_full_form_unchanged_and_lowercased(self):
        self.assertEqual(drw.normalize_mac("36:B4:ED:8C:ED:48"), "36:b4:ed:8c:ed:48")

    def test_rejects_non_macs(self):
        for bad in ("(incomplete)", "permanent", "36:74:32:fb:47", "36:74:32:fb:47:8:9", "zz:74:32:fb:47:08"):
            self.assertIsNone(drw.normalize_mac(bad), bad)


class Parser(unittest.TestCase):
    def test_short_ndp_mac_is_a_peer_and_own_mac_is_not(self):
        local, peers = drw.parse_iface_output(DUMP_42, ["en4"])
        self.assertEqual(local, {"en4": "36:b4:ed:8c:ed:48"})
        # ultra-96's en4, printed short by ndp, normalized so it matches its ifconfig form
        self.assertEqual(peers, {"en4": ["36:74:32:fb:47:08"]})


if __name__ == "__main__":
    unittest.main()
