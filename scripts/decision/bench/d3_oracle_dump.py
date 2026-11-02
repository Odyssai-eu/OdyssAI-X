"""Oracle side: the package's own d3_runtime (torch, CPU, reference path) on the bench; writes ids + probabilities.

usage: python d3_oracle_dump.py <package dir> <out.json>
"""
import json
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(Path(__file__).parent))
from d3_bench import requests  # noqa: E402
from d3_runtime import D3  # noqa: E402

rt = D3.from_pretrained(str(SRC), device="cpu", verify="none")
if len(sys.argv) > 3 and sys.argv[3] == "round-norms":
    # Diagnostic only: give the oracle the (1 + w) norm scales rounded to bf16, as mlx-vlm stores them.
    import torch
    with torch.no_grad():
        for _, m in rt.backbone.named_modules():
            if type(m).__name__ == "Qwen3_5RMSNorm":
                m.weight.copy_(((1 + m.weight.float()).bfloat16().float() - 1).bfloat16())
if len(sys.argv) > 3 and sys.argv[3] == "fp32":
    # Diagnostic only: the same checkpoint in FP32, the reference both bf16 paths are measured against.
    rt.backbone.float()
rows = []
for state, questions in requests():
    prepared = rt.prepare(state, questions)
    probs, tokens = rt.run(prepared)
    rows.append({"ids": {k: list(prepared.sequences[k]) for k in prepared.runnable},
                 "probs": probs, "tokens": tokens})
OUT.write_text(json.dumps({"attention_mode": rt.attention_mode, "fast": getattr(rt, "fast", None) is not None, "rows": rows}))
print("oracle done", SRC.name, rt.attention_mode, "fast", getattr(rt, "fast", None) is not None)
