"""Unit test for the dead-pool persistence fix (F1/F2) — no cluster nodes.
Drives save_cluster_state_v2 directly with stub pools to prove:
  1. purge (merge, no remove) keeps a dead pool as down:true — not dropped
  2. empty registry + no allow_empty_delete → file NOT unlinked
  3. explicit unload (remove_aliases + allow_empty_delete) DOES remove/delete
  4. load of B after A purged still keeps A (merge from disk)
"""
import os, sys, json, tempfile, types
os.environ["ODYSSAI_X_TOPOLOGY"] = sys.argv[1]
os.environ["ODYSSAI_X_STATE_DIR"] = sys.argv[2]
sys.argv = ["api.py"]  # neutralize argparse
sys.path.insert(0, sys.argv[0])
import importlib.util
spec = importlib.util.spec_from_file_location("api", os.path.join(os.path.dirname(__file__) if False else "scripts", "api.py"))

# Import api from the repo scripts dir
REPO = "/Users/sophie/Claude/code/MLX Distributed"
sys.path.insert(0, os.path.join(REPO, "scripts"))
import api

CID = "main"
sf = api.state_file_for(CID)
if sf.exists(): sf.unlink()

class StubPool:
    def __init__(self, alias, model, indices):
        self.alias=alias; self.model=model; self.mode="pipeline"; self.use_ap=True
        self.nodes_count=len(indices); self.node_indices=list(indices); self.nodes=[]
        self.kv_q8=False; self.draft_model=None; self.num_draft_tokens=4; self.backend="jaccl"
        self.is_vlm=False; self.is_vlm_dist=False

def set_live(*pools):
    api._pools.clear()
    for p in pools:
        api.set_pool(CID, p.alias, p)

def read():
    return {p["alias"]: p for p in api.load_cluster_state_v2(CID)}

A = StubPool("ornith","/m/ornith",[1,2,3])
B = StubPool("minimax-m3-vl","/m/mmx",[0])

# load A + B
set_live(A, B); api.save_cluster_state_v2(CID)
s = read(); assert set(s)=={"ornith","minimax-m3-vl"}, s
print("OK load A+B persisted:", sorted(s))

# purge A (A leaves registry, B stays) — merge save, no remove
set_live(B); api.save_cluster_state_v2(CID)
s = read()
assert set(s)=={"ornith","minimax-m3-vl"}, ("PURGE DROPPED A!", s)
assert s["ornith"].get("down") is True, ("A not tombstoned down", s["ornith"])
assert "down" not in s["minimax-m3-vl"], s["minimax-m3-vl"]
print("OK purge keeps A as down, B live:", {k:v.get('down') for k,v in s.items()})

# empty registry (both ranks died) — no allow_empty_delete → file preserved
set_live(); api.save_cluster_state_v2(CID)
assert sf.exists(), "FILE UNLINKED ON EMPTY REGISTRY (the bug)!"
s = read(); assert set(s)=={"ornith","minimax-m3-vl"}, s
print("OK empty registry preserves file + both down:", {k:v.get('down') for k,v in s.items()})

# load B back live after the outage — merge keeps A
set_live(B); api.save_cluster_state_v2(CID)
s = read(); assert set(s)=={"ornith","minimax-m3-vl"}, ("reload dropped A", s)
assert s["ornith"].get("down") is True and "down" not in s["minimax-m3-vl"]
print("OK reload B keeps A(down):", {k:v.get('down') for k,v in s.items()})

# explicit unload of A — remove + allow delete
api.save_cluster_state_v2(CID, remove_aliases=["ornith"], allow_empty_delete=True)
s = read(); assert set(s)=={"minimax-m3-vl"}, ("unload didn't remove A", s)
print("OK explicit unload A removes it:", sorted(s))

# explicit unload-all → file deleted
set_live(); api.save_cluster_state_v2(CID, remove_aliases=["minimax-m3-vl"], allow_empty_delete=True)
assert not sf.exists(), "unload-all should delete the file"
print("OK explicit unload-all deletes file")
print("\nALL PERSISTENCE TESTS PASSED")
