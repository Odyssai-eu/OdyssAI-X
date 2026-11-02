"""MLX side: vela3_mlx on the vela bench, compared answer by answer with the oracle dump.

Numbers: max |diff| per question type; everything else (choices, spans, labels, definitions, heads, errors) must be equal.
usage: python vela_mlx_compare.py <decision dir with vela3/> <assembled vela MLX dir> <vela package dir> <oracle.json>
"""
import json
import sys
import time
from pathlib import Path

DEC, MLX, PKG, ORACLE = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(DEC))
sys.path.insert(1, str(Path(__file__).parent))
from vela_bench import requests  # noqa: E402
from vela3 import vela3_mlx  # noqa: E402


def walk(a, b, path, out):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        if a != b:
            out["mismatch"].append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out["num"] = max(out["num"], abs(a - b))
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["mismatch"].append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            if k != "usage":
                walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["mismatch"].append(f"{path}: {len(a)} items != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    else:
        out["mismatch"].append(f"{path}: {type(a).__name__} != {type(b).__name__}")


oracle = json.loads(ORACLE.read_text())["rows"]
t0 = time.perf_counter()
m = vela3_mlx.load(MLX)
load_s = time.perf_counter() - t0
stats = {}
t0 = time.perf_counter()
for (state, questions), ref in zip(requests(PKG), oracle):
    got = json.loads(json.dumps(m.system_one(state=state, questions=questions), default=float))
    for k, q in questions.items():
        kind = q.get("type")
        s = stats.setdefault(kind, {"num": 0.0, "mismatch": [], "n": 0})
        s["n"] += 1
        walk(ref["answers"][k], got["answers"][k], k, s)
print(f"{MLX.name}: load {load_s:.1f} s, bench {time.perf_counter() - t0:.1f} s")
for kind, s in stats.items():
    print(f"  {kind:6s} n={s['n']:2d} max|d|={s['num']:.3e} mismatches={len(s['mismatch'])}")
    for line in s["mismatch"][:8]:
        print("     ", line)
