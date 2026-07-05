#!/usr/bin/env python3
"""Range-fetch selected tensors from HF safetensors shards into a local
sidecar safetensors file. Used to recover MTP/nextn weights that MLX
conversions strip (GLM-5.2 layers.78 tonight; LongCat model.mtp.* later).

Usage:
  python3 sidecar_fetch.py <repo_id> <key_prefix> <out_dir>
Example:
  python3 sidecar_fetch.py zai-org/GLM-5.2 model.layers.78. \
      /Volumes/models/odysseus/sidecar/GLM-5.2-mtp
Writes: <out_dir>/mtp-sidecar.safetensors + sidecar-index.json
Idempotent: skips tensors already present in the output file.
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1,
               "I32": 4, "U32": 4, "I64": 8, "F64": 8, "BOOL": 1}


def http_get(url: str, rng: str | None = None, retries: int = 5) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            headers = {"User-Agent": "sidecar-fetch/1.0"}
            if rng:
                headers["Range"] = rng
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - retry any transient error
            last = e
            print(f"  retry {attempt+1}/{retries} ({e})", flush=True)
    raise RuntimeError(f"giving up on {url} range={rng}: {last}")


def shard_header(base: str, fname: str):
    n = struct.unpack("<Q", http_get(f"{base}/{fname}", "bytes=0-7"))[0]
    hdr = json.loads(http_get(f"{base}/{fname}", f"bytes=8-{8+n-1}").decode())
    return n, hdr


def main() -> None:
    repo, prefix, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main"

    idx = json.loads(http_get(f"{base}/model.safetensors.index.json").decode())
    wm = idx["weight_map"]
    keys = sorted(k for k in wm if k.startswith(prefix))
    if not keys:
        print(f"no keys with prefix {prefix}")
        sys.exit(1)
    shards = sorted({wm[k] for k in keys})
    print(f"{len(keys)} tensors across {len(shards)} shards", flush=True)

    out_file = out_dir / "mtp-sidecar.safetensors"
    done: set[str] = set()
    if out_file.exists():  # resume: read existing header
        with open(out_file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            done = set(json.loads(f.read(n).decode())) - {"__metadata__"}
        print(f"resume: {len(done)} tensors already present", flush=True)

    todo_by_shard: dict[str, list[str]] = {}
    for k in keys:
        if k not in done:
            todo_by_shard.setdefault(wm[k], []).append(k)

    # Collect (key, dtype, shape, bytes) pulling data shard by shard.
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    total = 0
    for shard in shards:
        want = todo_by_shard.get(shard, [])
        if not want:
            continue
        hlen, hdr = shard_header(base, shard)
        data_start = 8 + hlen
        for k in want:
            meta = hdr[k]
            b0, b1 = meta["data_offsets"]
            nbytes = b1 - b0
            total += nbytes
            print(f"  {k}  {meta['dtype']} {meta['shape']}  "
                  f"{nbytes/1e6:.1f} MB", flush=True)
            blob = http_get(f"{base}/{shard}",
                            f"bytes={data_start+b0}-{data_start+b1-1}")
            assert len(blob) == nbytes, f"short read {k}"
            tensors[k] = (meta["dtype"], meta["shape"], blob)
    print(f"fetched {total/1e9:.2f} GB new", flush=True)

    # Merge with pre-existing content (rewrite whole file: simplest correct).
    if done:
        with open(out_file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            old_hdr = json.loads(f.read(n).decode())
            body = f.read()
        for k, meta in old_hdr.items():
            if k == "__metadata__":
                continue
            b0, b1 = meta["data_offsets"]
            tensors[k] = (meta["dtype"], meta["shape"], body[b0:b1])

    # Write safetensors: header (sorted keys, contiguous offsets) + blobs.
    new_hdr: dict = {"__metadata__": {"source": repo, "prefix": prefix}}
    off = 0
    order = sorted(tensors)
    for k in order:
        dt, shp, blob = tensors[k]
        new_hdr[k] = {"dtype": dt, "shape": shp,
                      "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    hdr_bytes = json.dumps(new_hdr).encode()
    tmp = out_file.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for k in order:
            f.write(tensors[k][2])
    tmp.rename(out_file)
    (out_dir / "sidecar-index.json").write_text(json.dumps(
        {"source_repo": repo, "prefix": prefix, "keys": order}, indent=1))
    print(f"OK -> {out_file}  ({off/1e9:.2f} GB, {len(order)} tensors)",
          flush=True)


if __name__ == "__main__":
    main()
