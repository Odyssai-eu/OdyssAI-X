#!/usr/bin/env python3
"""Challenge 2/3 — EMPIRICAL verification of mask composition exactness for the
MSA decode-only block-gather path (OdyssAI-X#53 phase 1, voie A exact).

We do NOT reimplement the indexer or the reference mask from memory: we import
the real classes from minimax_m3 and call the real Indexer.__call__ and
Indexer.build_block_mask (the gold reference). We then construct the gather
path EXACTLY as PLAN.md step 1 prescribes and prove, numerically, that the SET
of keys it effectively attends == the SET build_block_mask keeps.

Judged on CPU (Metal fp32 matmul has a ~1e-3 floor), per parity.py.
"""
import sys
sys.path.insert(0, "/tmp/m3-night")
import numpy as np
import mlx.core as mx
import minimax_m3
mx.set_default_device(mx.cpu)

NEG = mx.finfo(mx.float32).min  # the additive -inf build_block_mask uses

def make_indexer(block=128, topk=16, local=1, ihd=64, inh=4):
    args = minimax_m3.ModelArgs(
        hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=2, head_dim=ihd,
        index_n_heads=inh, index_head_dim=ihd,
        index_block_size=block, index_topk_blocks=topk, index_local_blocks=local,
    )
    idx = minimax_m3.Indexer(args)
    mx.eval(idx.parameters())
    return idx, args


def gather_keyset(inds, num_blocks, k_len, offset, S, block_size):
    """Build the gather path EXACTLY as PLAN.md step 1 prescribes and return a
    boolean [B,S,k_len] of which absolute key positions are EFFECTIVELY attended
    (selected-block AND causal-token AND valid -- i.e. not gathered-then-masked).

    Steps mirrored from the plan:
      - clamp the -1 slots to a SAFE block index, remember which slots were -1;
      - gather block key-position ranges via take_along_axis style indexing;
      - T_kv fixed = topk*block;
      - additive mask [B,1,1,T_kv] = block-selected AND causal AND valid.
    The effective set = gathered slot whose additive mask is 0 (kept).
    """
    B = inds.shape[0]
    topk = inds.shape[2]
    # (a) clamp -1 -> safe index 0; mark invalid slots
    invalid_slot = inds < 0                       # [B,S,topk]
    safe_blk = mx.where(invalid_slot, mx.array(0, mx.int32), inds.astype(mx.int32))

    # absolute key positions covered by each gathered block: blk*block + [0..block)
    within = mx.arange(block_size)                # [block]
    # [B,S,topk,block] -> absolute key index per gathered slot
    base = (safe_blk[..., None] * block_size)     # [B,S,topk,1]
    gathered_pos = base + within[None, None, None, :]   # [B,S,topk,block]
    T_kv = topk * block_size
    gathered_pos = gathered_pos.reshape(B, S, T_kv)     # [B,S,T_kv]

    # valid slot mask (broadcast invalid_slot over the block dim)
    valid = mx.broadcast_to(
        (~invalid_slot)[..., None], (B, S, topk, block_size)
    ).reshape(B, S, T_kv)

    # causal per gathered position: key_abs_pos <= query_pos
    q_pos = offset + mx.arange(S)                 # [S]
    causal = gathered_pos <= q_pos[None, :, None]  # [B?,S,T_kv] -> broadcast B
    # also: gathered position must be a REAL key (< k_len). Padded tail of the
    # last block (block*num_blocks may exceed k_len) is not a real key.
    in_range = gathered_pos < k_len

    keep_slot = valid & causal & in_range          # [B,S,T_kv] over gathered slots

    # Scatter the kept gathered slots back to an absolute [B,S,k_len] set so we
    # can compare against the dense reference on the SAME coordinate system.
    eff = mx.zeros((B, S, k_len), dtype=mx.bool_)
    eff_np = np.array(eff)
    gp = np.array(gathered_pos)
    ks = np.array(keep_slot)
    for b in range(B):
        for s in range(S):
            sel = gp[b, s][ks[b, s]]
            eff_np[b, s, sel] = True
    return mx.array(eff_np), gathered_pos, keep_slot, invalid_slot


def dense_keyset(idx, inds, num_blocks, k_len, offset, S):
    """The reference key set = positions build_block_mask keeps (additive 0)."""
    m = idx.build_block_mask(inds, num_blocks, k_len, offset, S, mx.float32)
    # m: [B,1,S,k_len]; kept iff value == 0.0 (else == finfo.min)
    kept = (m[:, 0] == 0.0)   # [B,S,k_len]
    return kept, m


def run_case(name, offset, block=128, topk=16, local=1, seed=0):
    """Decode case: S_q=1 at absolute position `offset` (k_len = offset+1)."""
    mx.random.seed(seed)
    idx, args = make_indexer(block=block, topk=topk, local=local)
    B, S = 1, 1
    # decode: the indexer is fed the single new hidden state; but it reads its
    # key history from cache. To exercise the real selection logic we drive the
    # indexer directly with a synthetic cache holding offset keys.
    cache = minimax_m3.M3CacheLayer()
    # prime idx_keys cache to length=offset by writing random index-keys, then
    # the live call appends the current token (mirrors update_index at decode).
    if offset > 0:
        # synthesize a prior history by calling update_index with offset keys
        hist = mx.random.normal((B, 1, offset, idx.head_dim))
        cache.offset = 0
        cache.update_index(hist)
        cache.offset = offset  # main KV would have advanced offset to `offset`
    h = mx.random.normal((B, S, args.hidden_size))
    inds, num_blocks, k_len = idx(h, offset, cache)
    mx.eval(inds)
    assert k_len == offset + 1, f"k_len {k_len} != offset+1 {offset+1}"

    dense, dense_mask = dense_keyset(idx, inds, num_blocks, k_len, offset, S)
    eff, gathered_pos, keep_slot, invalid_slot = gather_keyset(
        inds, num_blocks, k_len, offset, S, block
    )
    mx.eval(dense, eff)

    dnp = np.array(dense)[0, 0]      # [k_len]
    enp = np.array(eff)[0, 0]
    n_dense = int(dnp.sum())
    n_eff = int(enp.sum())
    set_equal = bool(np.array_equal(dnp, enp))
    only_dense = np.where(dnp & ~enp)[0]
    only_eff = np.where(enp & ~dnp)[0]

    n_invalid = int(np.array(invalid_slot).sum())
    inds_np = np.array(inds)[0, 0]
    n_minus1 = int((inds_np < 0).sum())

    # (a) prove a clamped -1 slot does NOT leak: every gathered position coming
    # from an invalid slot must be masked out (keep_slot False there).
    leak = 0
    if n_invalid > 0:
        inv = np.array(invalid_slot)[0, 0]          # [topk] over slots
        ks = np.array(keep_slot)[0, 0].reshape(topk, block)
        # any kept slot whose originating block-slot was invalid = LEAK
        leak = int(ks[inv].sum())

    print(f"=== {name} (offset={offset}, k_len={k_len}, num_blocks={num_blocks}, "
          f"topk={topk}) ===")
    print(f"  selected block inds (raw): {inds_np.tolist()}")
    print(f"  #'-1' slots = {n_minus1}  (#invalid flagged = {n_invalid})")
    print(f"  |dense keep set| = {n_dense}   |gather effective set| = {n_eff}")
    print(f"  SET EQUAL = {set_equal}")
    if not set_equal:
        print(f"    keys only-in-dense: {only_dense.tolist()[:20]}")
        print(f"    keys only-in-gather: {only_eff.tolist()[:20]}")
    print(f"  (a) -1 leak count (gathered+kept from invalid slot) = {leak}  "
          f"{'LEAK!' if leak else 'no leak'}")
    return {
        "name": name, "set_equal": set_equal, "leak": leak,
        "n_dense": n_dense, "n_eff": n_eff,
        "only_dense": only_dense.tolist(), "only_eff": only_eff.tolist(),
        "n_minus1": n_minus1, "offset": offset,
        "idx": idx, "inds": inds, "num_blocks": num_blocks,
        "k_len": k_len, "args": args, "dense_mask": dense_mask,
    }


def numeric_sdpa(r, block=128):
    """Numeric proof (b/c): run real SDPA two ways on CPU and compare outputs.
      DENSE: full K/V + build_block_mask additive mask.
      GATHER: gather selected blocks + composed additive mask, SDPA on subset.
    Output must match < 1e-6 (voie A exact)."""
    idx = r["idx"]; inds = r["inds"]; args = r["args"]
    offset = r["offset"]; k_len = r["k_len"]; num_blocks = r["num_blocks"]
    B, S = 1, 1
    H, Hkv, D = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    topk = inds.shape[2]
    scale = D ** -0.5
    mx.random.seed(123)
    q = mx.random.normal((B, H, S, D))
    k = mx.random.normal((B, Hkv, k_len, D))
    v = mx.random.normal((B, Hkv, k_len, D))

    # --- DENSE reference (the current production path) ---
    dense_mask = r["dense_mask"]  # [B,1,S,k_len]
    out_dense = minimax_m3.scaled_dot_product_attention(
        q, k, v, cache=None, scale=scale, mask=dense_mask
    )
    mx.eval(out_dense)

    # --- GATHER path ---
    safe_blk = mx.where(inds < 0, mx.array(0, mx.int32), inds.astype(mx.int32))[0, 0]  # [topk]
    T_kv = topk * block
    # build per-gathered-slot absolute key indices, clamped into [0,k_len-1]
    within = mx.arange(block)
    gpos = (safe_blk[:, None] * block + within[None, :]).reshape(T_kv)  # [T_kv]
    gpos_clamped = mx.minimum(gpos, mx.array(k_len - 1, mx.int32))
    # gather along key axis: take_along_axis over [B,Hkv,k_len,D]
    gidx = mx.broadcast_to(
        gpos_clamped[None, None, :, None], (B, Hkv, T_kv, D)
    )
    kg = mx.take_along_axis(k, gidx, axis=2)  # [B,Hkv,T_kv,D]
    vg = mx.take_along_axis(v, gidx, axis=2)

    # composed additive mask over gathered slots [B,1,S,T_kv]
    invalid_slot = (inds < 0)[0, 0]                 # [topk]
    valid = mx.broadcast_to((~invalid_slot)[:, None], (topk, block)).reshape(T_kv)
    q_pos = offset
    causal = gpos <= q_pos
    in_range = gpos < k_len
    keep = valid & causal & in_range                # [T_kv]
    gmask = mx.where(keep, mx.array(0.0), mx.array(NEG))[None, None, None, :]

    out_gather = minimax_m3.scaled_dot_product_attention(
        q, kg, vg, cache=None, scale=scale, mask=gmask
    )
    mx.eval(out_gather)

    d = float(np.abs(np.array(out_dense) - np.array(out_gather)).max())
    print(f"  (b/c) SDPA dense-vs-gather max|Δ| = {d:.3e}  "
          f"{'EXACT (<1e-6)' if d < 1e-6 else 'NOT exact'}")
    return d


if __name__ == "__main__":
    cases = []
    # Long contexts > 2048 (the useful regime). Various offsets to stress the
    # local block, the partially-filled last block, and -1 padding of topk.
    for off, seed in [(3000, 1), (5000, 7), (9999, 3), (4096, 11), (2049, 5)]:
        r = run_case(f"decode@{off}", offset=off, seed=seed)
        d = numeric_sdpa(r)
        r["sdpa_delta"] = d
        cases.append(r)
        print()

    # Force a case with MANY -1 slots: tiny context just above 2048 so
    # num_blocks < topk -> the indexer pads inds with -1. Build a case where
    # num_blocks(=offset+1 / 128) < topk(=16): offset ~ 1500 -> ~12 blocks < 16.
    print("### forced -1 case (num_blocks < topk) ###")
    r = run_case("decode@1500 (nb<topk)", offset=1500, seed=2)
    d = numeric_sdpa(r)
    r["sdpa_delta"] = d
    cases.append(r)
    print()

    all_set = all(c["set_equal"] for c in cases)
    all_noleak = all(c["leak"] == 0 for c in cases)
    all_exact = all(c["sdpa_delta"] < 1e-6 for c in cases)
    print("SUMMARY")
    for c in cases:
        print(f"  {c['name']:28s} set_equal={c['set_equal']} leak={c['leak']} "
              f"sdpaΔ={c['sdpa_delta']:.2e}")
    print(f"\nVERDICT: set_equal_all={all_set}  no_leak_all={all_noleak}  "
          f"sdpa_exact_all={all_exact}")
    sys.exit(0 if (all_set and all_noleak and all_exact) else 1)
