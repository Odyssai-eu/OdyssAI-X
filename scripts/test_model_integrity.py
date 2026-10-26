#!/usr/bin/env python3
"""A truncated weight file must never pass the load-time completeness probe (1.62.9).

The TB5 sync receiver (`tar xf -`) writes shards under their final names, so a
cancel mid-stream left a truncated shard that the name-only check accepted. The
downloader now writes the HF-announced weight sizes (SIZES_MANIFEST) before
`hf download` starts, the sync job writes the expected sizes on the target before
any data moves, and the probe refuses a weight file whose size differs.

Runs the real probe script locally against temp dirs; no node, no network.

    .venv/bin/python scripts/test_model_integrity.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile

if "CLUSTER_CONFIG_FILE" not in os.environ:
    os.environ["CLUSTER_CONFIG_FILE"] = os.path.join(tempfile.mkdtemp(), "cc.json")
if "ODYSSAI_X_STATE_DIR" not in os.environ:
    os.environ["ODYSSAI_X_STATE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api  # noqa: E402

FAILS = []


def check(label, got, want):
    ok = got == want
    print(("OK  " if ok else "FAIL") + f" {label}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILS.append(label)


def write(path, nbytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * nbytes)


def make_model(sizes, manifest=None, extra=None):
    """A sharded model dir: config, tokenizer, index listing `sizes`' shards."""
    d = tempfile.mkdtemp()
    open(os.path.join(d, "config.json"), "w").write('{"model_type":"x"}')
    open(os.path.join(d, "tokenizer.json"), "w").write("{}")
    json.dump({"weight_map": {f"w{i}": n for i, n in enumerate(sizes)}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    for name, n in {**sizes, **(extra or {})}.items():
        write(os.path.join(d, name), n)
    if manifest is not None:
        open(os.path.join(d, api.SIZES_MANIFEST), "w").write(api._sizes_manifest_json(manifest))
    return d


def probe(d, models_dir=None, rel=None):
    src = api._layout_probe_py(rel or d, models_dir)
    return subprocess.run([sys.executable, "-c", src], capture_output=True,
                          text=True).stdout.strip()


SHARDS = {"model-00001-of-00002.safetensors": 1000, "model-00002-of-00002.safetensors": 600}

# Probe
d = make_model(SHARDS, manifest=SHARDS)
check("complete model with manifest -> OK", probe(d), "OK")

d = make_model({**SHARDS, "model-00002-of-00002.safetensors": 300}, manifest=SHARDS)
check("truncated shard under its final name -> refused", probe(d), "INCOMPLETE_SIZES:1")

d = make_model({**SHARDS, "model-00001-of-00002.safetensors": 1500}, manifest=SHARDS)
check("oversized shard -> refused", probe(d), "INCOMPLETE_SIZES:1")

d = make_model({**SHARDS, "model-00002-of-00002.safetensors": 300})
check("no manifest (local conversion) -> name check only, as before", probe(d), "OK")

d = make_model(SHARDS, manifest={**SHARDS, "consolidated.safetensors": 5000})
check("listed duplicate deleted on purpose -> OK", probe(d), "OK")

d = make_model(SHARDS, manifest=SHARDS)
os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
check("shard missing -> still caught by the index", probe(d), "INCOMPLETE_SHARDS:1")

d = make_model(SHARDS, manifest=SHARDS)
open(os.path.join(d, api.SIZES_MANIFEST), "w").write("{not json")
check("unreadable manifest -> refused", probe(d), "INCOMPLETE_SIZES:manifest unreadable")

d = make_model(SHARDS, manifest=SHARDS)
check("relative path resolved under models_dir",
      probe(None, models_dir=os.path.dirname(d), rel=os.path.basename(d)), "OK")

# HF tree -> sizes
tree = [
    {"type": "file", "path": "config.json", "size": 10},
    {"type": "file", "path": "model.safetensors", "size": 7, "lfs": {"size": 9000}},
    {"type": "directory", "path": "6bit"},
    {"type": "file", "path": "6bit/model.safetensors", "size": 0, "lfs": {"size": 4000}},
    {"type": "file", "path": "6bit/config.json", "size": 20},
]
check("whole repo: every file, `size` first (HF puts the real size there)",
      api._hf_tree_sizes(tree, None),
      {"config.json": 10, "model.safetensors": 7, "6bit/model.safetensors": 4000,
       "6bit/config.json": 20})
check("subfolder spec: only that variant, prefix stripped",
      api._hf_tree_sizes(tree, "6bit"),
      {"model.safetensors": 4000, "config.json": 20})
check("manifest keeps weight files only",
      json.loads(api._sizes_manifest_json({"config.json": 10, "a.safetensors": 5})),
      {"files": {"a.safetensors": 5}})

# Sync: expected sizes for the target


def run_expected(cat_out, cat_rc, listing):
    orig = (api._ssh_exec, api._dir_manifest)
    api._ssh_exec = lambda ssh, cmd, timeout=12: (cat_rc, cat_out, "")

    async def fake_listing(ssh, path, timeout=30.0):
        return listing
    api._dir_manifest = fake_listing
    try:
        return asyncio.run(api._sync_expected_sizes({"ssh": "x"}, "/m/org/name/"))
    finally:
        api._ssh_exec, api._dir_manifest = orig


check("source manifest wins over the source disk",
      run_expected(api._sizes_manifest_json({"a.safetensors": 5}), 0, {"a.safetensors": 3}),
      {"a.safetensors": 5})
check("no source manifest -> source disk sizes, weights only",
      run_expected("", 1, {"a.safetensors": 3, "config.json": 9}),
      {"a.safetensors": 3})
check("garbled source manifest -> source disk sizes",
      run_expected("{oops", 0, {"a.safetensors": 3}), {"a.safetensors": 3})
check("unreadable source -> None (sync slot fails)", run_expected("", 255, None), None)

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
sys.exit(1 if FAILS else 0)
