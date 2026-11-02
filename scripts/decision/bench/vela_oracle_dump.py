"""Oracle side: the package's own modeling_vela3 (torch, CPU) on the vela bench; writes the system_one answers.

usage: python vela_oracle_dump.py <vela package dir> <d3 base package dir> <out.json>
"""
import json
import sys
from pathlib import Path

VELA, BASE, OUT = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(VELA))
from vela_bench import requests  # noqa: E402
from modeling_vela3 import Vela3MiniModel  # noqa: E402

m = Vela3MiniModel.from_pretrained(str(VELA), base_path=str(BASE), device="cpu", verify="none")
rows = [m.system_one(state=state, questions=questions) for state, questions in requests(VELA)]
OUT.write_text(json.dumps({"rows": rows}, default=float))
print("vela oracle done", VELA.name)
