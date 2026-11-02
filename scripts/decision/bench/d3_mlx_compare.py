"""MLX side: the vendored runtime + D3MLX on the bench, compared with the oracle dump (same ids, |dp|).

usage: python d3_mlx_compare.py <decision dir with d3/> <converted MLX dir> <oracle.json>
"""
import json
import sys
import time
from pathlib import Path

DEC, MLX, ORACLE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(DEC))
sys.path.insert(1, str(Path(__file__).parent))
from d3_bench import requests  # noqa: E402
from d3 import d3_mlx  # noqa: E402

oracle = json.loads(ORACLE.read_text())
rt = d3_mlx.load(MLX)
worst, n, same_ids, same_top, t0 = 0.0, 0, True, 0, time.perf_counter()
dump = []
for (state, questions), ref in zip(requests(), oracle["rows"]):
    prepared = rt.prepare(state, questions)
    probs, tokens = rt.run(prepared)
    dump.append({"probs": probs, "tokens": tokens})
    for k, p in probs.items():
        same_ids &= list(prepared.sequences[k]) == ref["ids"][k]
        d = max(abs(a - b) for a, b in zip(p, ref["probs"][k]))
        worst = max(worst, d)
        same_top += max(range(len(p)), key=p.__getitem__) == max(range(len(p)), key=ref["probs"][k].__getitem__)
        n += 1
ORACLE.with_name(ORACLE.stem + ".mlx.json").write_text(json.dumps({"rows": dump}))
print(f"{MLX.name}: mode={rt.attention_mode} questions={n} ids_identical={same_ids} "
      f"argmax_agree={same_top}/{n} max|dp|={worst:.3e} ({time.perf_counter() - t0:.1f} s)")
