"""Patch longcat2 (LongCat-2.0) — pipeline-parallel + cache wiring (5-node).

WHY. mlx-lm's `longcat2.Longcat2Model` is standalone (`nn.Module`), unlike
`glm_moe_dsa` which INHERITS `deepseek_v32` and gets pipeline() for free. So
longcat2 has `shard()` (tensor) but NO `pipeline()`. LongCat-2.0 4-bit ~1 TB
needs 5 nodes, and:
  - tensor-parallel is impossible: 64 heads / q_b_proj 12288 not divisible by 5;
  - 4-node won't fit: ~1200 GB required > ~964 GB loadable.
So 5-node PIPELINE is the only path — hence this patch.

Two fixes, both required for the model to actually run distributed:

1. pipeline() + a pipelined __call__ on Longcat2Model — the recv/send/all_gather
   block is copied LINE-ACCURATE from deepseek_v32 (rewriting it is what killed
   prior pipeline attempts, see glm_moe_dsa_model.py). Only longcat2 specifics
   are adapted:
     - the input embedding is `ngram_embeddings` with its OWN cache at cache[0];
       it is computed ONLY on the rank holding the first layer (start_idx==0).
       Other ranks build a zeros template of the right shape for recv_like —
       calling ngram there would pollute its rolling-context cache.
     - the cache is [ngram_cache] + [per-LOCAL-layer caches] (a +1 offset vs
       deepseek_v32), so local layer i uses cache[1 + i].

2. make_cache wiring. longcat2's `make_cache` lives on the INNER Longcat2Model,
   but the OUTER Model has none → `make_prompt_cache(model)` falls back to a
   default KVCache-per-layer, which is the WRONG shape for longcat2 (it needs
   the ngram ArraysCache + per-layer dual-block CacheList). We add make_cache
   to the outer Model (delegates to the inner) and make the inner make_cache
   skip the None layers left by pipeline().

Reference (truth): the original mlx-lm longcat2.py; the pipeline block is the
verbatim deepseek_v32 one.
"""

import os
import sys
from typing import Any, Optional

import mlx.core as mx

from mlx_lm.models import longcat2 as _lc
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache


class Longcat2Model(_lc.Longcat2Model):
    """Longcat2Model + pipeline-parallel. __init__ inherited (builds ngram +
    layers + norm); we add the pipeline split fields."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    # Reverse split (rank 0 = last layers), CAPACITY-AWARE. LongCat 4-bit
    # ~1 TB even-split = ~209 GB/rank, which OOMs the 256 GB nodes; the
    # 512 GB .29 must carry proportionally more. RUNNER_RAM_WEIGHTS (CSV of
    # per-rank weights, rank order — set by the engine's remote_cmd) drives
    # the proportional layer counts; even split is the fallback. The reverse
    # ordering + start_idx math matches deepseek_v32 (rank N-1 = first block).
    def pipeline(self, group):
        self.pipeline_rank = rank = group.rank()
        self.pipeline_size = N = group.size()
        L = len(self.layers)

        w_env = os.environ.get("RUNNER_RAM_WEIGHTS", "").strip()
        if w_env:
            try:
                w = [max(float(x), 0.0) for x in w_env.split(",")]
                assert len(w) == N and sum(w) > 0
            except Exception:
                w = None
        else:
            w = None

        if w:
            raw = [L * wi / sum(w) for wi in w]
            counts = [int(r) for r in raw]
            # hand out the remainder to the largest fractional parts, but never
            # leave a rank with 0 layers
            for r in range(N):
                if counts[r] == 0:
                    counts[r] = 1
            rem = L - sum(counts)
            order = sorted(range(N), key=lambda i: raw[i] - int(raw[i]), reverse=True)
            i = 0
            while rem > 0:
                counts[order[i % N]] += 1; rem -= 1; i += 1
            while rem < 0:  # over-allocated by the min-1 floor: trim the largest
                j = max(range(N), key=lambda k: counts[k])
                if counts[j] > 1:
                    counts[j] -= 1; rem += 1
                else:
                    break
        else:
            base, extra = L // N, L - (L // N) * N
            counts = [base + (1 if r < extra else 0) for r in range(N)]

        # rank N-1 holds the first block, rank 0 the last (reverse pipeline)
        self.start_idx = sum(counts[r2] for r2 in range(rank + 1, N))
        self.end_idx = self.start_idx + counts[rank]
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = counts[rank]

        # ngram_embeddings (272.8 GB — 16 embedders x 16.9 GB + word_embeddings)
        # is the INPUT embedding, needed ONLY on the rank holding the first block
        # (start_idx == 0). sharded_load builds `local_files` by walking
        # model.parameters() AFTER this pipeline() call, so a None sub-module is
        # simply absent from the tree and its shards are never downloaded/loaded.
        # Without this null the ~273 GB replicate onto all 5 ranks and OOM the
        # 256 GB nodes at load. __call__ already builds a zeros template instead
        # of calling ngram on the non-first ranks, so nulling it is safe.
        dropped_ngram = False
        if self.start_idx != 0:
            self.ngram_embeddings = None
            dropped_ngram = True

        sys.stderr.write(
            f"[longcat2-pipeline] rank {rank}/{N}: layers [{self.start_idx}, "
            f"{self.end_idx}) = {self.num_layers} (weights={'yes' if w else 'even'}, "
            f"ngram={'DROPPED' if dropped_ngram else 'KEPT (first block)'})\n")
        sys.stderr.flush()

    def make_cache(self):
        # [ngram context] + one CacheList per LOCAL layer (skip the None slots
        # the pipeline split left). Local-aligned: cache[1 + i] ↔ local layer i.
        caches = [ArraysCache(size=1)]
        for layer in self.layers:
            if layer is None:
                continue
            sub = []
            for attn in layer.self_attn:
                sub.append(CacheList(KVCache()) if attn.skip_topk
                           else CacheList(KVCache(), KVCache()))
            caches.append(CacheList(*sub))
        return caches

    def __call__(self, input_ids: mx.array, cache: Optional[Any] = None) -> mx.array:
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * (self.num_layers + 1)

        # Input embedding: ngram (with its cache[0]) ONLY on the rank that holds
        # the first layer; elsewhere a zeros template of the right shape/dtype
        # for recv_like (calling ngram there would advance its context cache).
        if self.start_idx == 0:
            h = self.ngram_embeddings(input_ids, cache=cache[0])
        else:
            B = input_ids.shape[0]
            S = input_ids.shape[-1]
            h = mx.zeros((B, S, self.args.hidden_size), dtype=self.norm.weight.dtype)

        # Mask from the first local layer's KV offset (all layers share offset).
        first = cache[1] if len(cache) > 1 else None
        main_kv = first[0][0] if first is not None else None
        mask = create_attention_mask(h, main_kv, return_array=True)

        # --- verbatim deepseek_v32 pipeline body (only cache index is +1) ---
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[1 + i])

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            last = cache[-1]
            if last is not None:
                # last is a CacheList of dual-block sub-caches; keep the send in
                # the graph via the first sub-cache's first KVCache keys.
                kc = last[0][0]
                if kc.keys is not None:
                    kc.keys = mx.depends(kc.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(_lc.Model):
    """Outer Model — rebuild with the pipelined inner model + expose make_cache
    so make_prompt_cache builds longcat2's real cache (not the default)."""

    def __init__(self, args):
        super().__init__(args)
        self.model = Longcat2Model(args)

    def make_cache(self):
        return self.model.make_cache()


def apply_longcat2_pipeline() -> None:
    """Register this patched module as `mlx_lm.models.longcat2` before the
    loader imports it — same mechanic as apply_glm_dsa (sys.modules swap).
    We reuse every other class from the original module unchanged."""
    orig = sys.modules.get("mlx_lm.models.longcat2", _lc)
    # Copy the patched classes onto the original module object so all other
    # names (ModelArgs, layers, NgramEmbedding wiring, sanitize, …) stay intact.
    orig.Longcat2Model = Longcat2Model
    orig.Model = Model
    sys.modules["mlx_lm.models.longcat2"] = orig
    sys.stderr.write("[patch] longcat2_pipeline: pipeline() + cache wiring registered\n")
    sys.stderr.flush()
