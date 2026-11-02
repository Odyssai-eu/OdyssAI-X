"""max |dp| and argmax agreement between two dumps (oracle bf16 / fp32 / MLX). usage: d3_dump_diff.py a.json b.json"""
import json
import sys

a, b = (json.load(open(p))["rows"] for p in sys.argv[1:3])
worst, agree, n = 0.0, 0, 0
for ra, rb in zip(a, b):
    for k, p in ra["probs"].items():
        q = rb["probs"][k]
        worst = max(worst, max(abs(x - y) for x, y in zip(p, q)))
        agree += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
        n += 1
print(f"{sys.argv[1].split('/')[-1]} vs {sys.argv[2].split('/')[-1]}: max|dp|={worst:.3e} argmax {agree}/{n}")
