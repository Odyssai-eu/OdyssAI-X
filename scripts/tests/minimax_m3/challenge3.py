#!/usr/bin/env python3
"""Challenge 3/3 — threshold + cache-order + partial-block empirical verification.

Uses the REAL Indexer / Attention from minimax_m3.py at tiny scale so the math
matches prod (only block_size/topk are smaller: 4/2 here vs 128/16 prod).
threshold analogue: topk*block = 2*4 = 8  (== 2048 in prod).
"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/m3-night")
import mlx.core as mx
import minimax_m3
mx.set_default_device(mx.cpu)  # exactitude jugée sur CPU (plancher Metal fp32 ~1e-3)

cfg = json.load(open("/tmp/m3-night/tiny-hub/config.json"))
args = minimax_m3.ModelArgs.from_dict(cfg)
BLK = args.index_block_size      # 4
TOPK = args.index_topk_blocks    # 2
THRESH = TOPK * BLK              # 8  (prod analogue of 2048)
print(f"config: block_size={BLK} topk_blocks={TOPK} local_blocks={args.index_local_blocks} "
      f"=> useful-regime threshold topk*block={THRESH}")
print(f"index_n_heads={args.index_n_heads} index_head_dim={args.index_head_dim} hidden={args.hidden_size}")
print()

# Build a single sparse Indexer with random but fixed weights.
mx.random.seed(0)
idx = minimax_m3.Indexer(args)
mx.eval(idx.parameters())

def num_blocks_of(k_len):
    return (k_len + BLK - 1) // BLK

# ---------------------------------------------------------------------------
# (a) THRESHOLD: at k_len <= topk*block, num_blocks <= topk => ALL blocks
#     selected => gather == dense (no gain). Confirm boundary at k_len=2048
#     (here THRESH=8): num_blocks <= topk.
# ---------------------------------------------------------------------------
print("=== (a) threshold: num_blocks vs topk across k_len ===")
print(f"{'k_len':>6} {'num_blocks':>11} {'<=topk?':>8} {'topk_eff':>9} {'all_selected?':>14}")
boundary_ok = True
for k_len in [1, BLK, THRESH - 1, THRESH, THRESH + 1, THRESH + BLK, 2 * THRESH]:
    nb = num_blocks_of(k_len)
    topk_eff = min(TOPK, nb)
    le_topk = nb <= TOPK
    # at decode (S_q=1) build a 1-token query at position k_len-1 (offset=k_len-1)
    print(f"{k_len:>6} {nb:>11} {str(le_topk):>8} {topk_eff:>9} {str(le_topk):>14}")
    if k_len == THRESH:
        boundary_ok = (nb <= TOPK)
        print(f"   -> BOUNDARY k_len=={THRESH} (prod 2048): num_blocks={nb} <= topk={TOPK}: {boundary_ok}")

# Decode-time empirical check: at k_len==THRESH the indexer returns ALL blocks
# (the argpartition branch is skipped: topk == num_blocks -> arange path).
def decode_inds(k_len):
    """Run indexer for a 1-token decode query whose context length is k_len."""
    B = 1
    # offset = k_len-1, S=1 query at the last position -> k_len total keys
    offset = k_len - 1
    cache = minimax_m3.M3CacheLayer()
    # seed idx cache with k_len-1 prior keys so update_index brings it to k_len
    if offset > 0:
        prior = mx.random.normal((B, 1, offset, args.index_head_dim))
        cache.idx_keys = None
        # mimic prefill of `offset` tokens through update_index
        cache.update_index(prior)
        cache.offset = offset  # advance as main update_and_fetch would
    h = mx.random.normal((B, 1, args.hidden_size))
    inds, nb, kl = idx(h, offset, cache)
    mx.eval(inds)
    return np.array(inds), nb, kl

for k_len in [THRESH, THRESH + 1]:
    inds, nb, kl = decode_inds(k_len)
    sel = sorted(int(x) for x in inds.ravel() if x >= 0)
    print(f"   decode k_len={k_len}: indexer k_len={kl} num_blocks={nb} selected_blocks={sel} "
          f"({'ALL' if len(sel)==nb else 'SUBSET'})")
print()

# ---------------------------------------------------------------------------
# (b) CACHE ORDER: indexer (update_index) runs BEFORE update_and_fetch; the
#     gather must read K/V RETURNED BY update_and_fetch, not idx_keys.
#     Confirm by reading Attention.__call__ behaviour: offset captured pre-update,
#     indexer called, THEN update_and_fetch advances offset and returns full K/V.
# ---------------------------------------------------------------------------
print("=== (b) cache order: offset / idx_keys vs main KV post-update ===")
B = 1
cache = minimax_m3.M3CacheLayer()
# prefill 6 tokens
attn = minimax_m3.Attention(args, layer_idx=1)  # sparse layer
mx.eval(attn.parameters())
x_pre = mx.random.normal((B, 6, args.hidden_size))
_ = attn(x_pre, None, cache)
mx.eval(_)
print(f"after prefill 6: cache.offset={cache.offset} idx_keys.len={cache.idx_keys.shape[2]} "
      f"main_keys.len={cache.keys.shape[2] if cache.keys is not None else None} "
      f"(allocated, valid={cache.offset})")
off_before = cache.offset
# decode 1 token: capture what offset the indexer sees vs what update_and_fetch returns
x_dec = mx.random.normal((B, 1, args.hidden_size))
# Instrument: monkeypatch update_index and update_and_fetch to log order/offset
log = []
orig_ui = cache.update_index
orig_uf = cache.update_and_fetch
def ui(idx_k):
    log.append(("update_index", cache.offset, idx_k.shape[2]))
    return orig_ui(idx_k)
def uf(k, v):
    log.append(("update_and_fetch", cache.offset, k.shape[2]))
    r = orig_uf(k, v)
    log.append(("post_uf_offset", cache.offset, r[0].shape[2]))
    return r
cache.update_index = ui
cache.update_and_fetch = uf
out = attn(x_dec, None, cache)
mx.eval(out)
for e in log:
    print("   order:", e)
print(f"   => indexer ran at offset={log[0][1]} (pre-update), "
      f"update_and_fetch advanced offset to {cache.offset}, "
      f"returned main K len={[e[2] for e in log if e[0]=='update_and_fetch'][0]}")
print(f"   idx_keys.len now={cache.idx_keys.shape[2]} (offset-indexed, decoupled from main K dtype/quant)")
print()

# ---------------------------------------------------------------------------
# (c) PARTIAL LAST BLOCK: build_block_mask pads to a multiple of block_size.
#     Test k_len = THRESH+1 (here 9 = 2 full blocks of 4 + 1 partial of 1).
#     Confirm the dense mask handles the partial block and that a gather of the
#     SAME selected blocks, masked + sliced to k_len, equals the dense path.
# ---------------------------------------------------------------------------
print("=== (c) partial last block: k_len=THRESH+1 (prod analogue k_len=2049) ===")

def dense_vs_gather(k_len):
    """Compare dense-mask SDPA against an explicit block-gather SDPA on the
    SAME selected blocks, for a single decode query. Exactness target <1e-6."""
    B = 1
    offset = k_len - 1
    Hq = args.num_attention_heads
    Hkv = args.num_key_value_heads
    Dh = args.head_dim
    nb = num_blocks_of(k_len)
    pad = nb * BLK - k_len
    # main K/V for k_len keys
    mx.random.seed(123)
    K = mx.random.normal((B, Hkv, k_len, Dh))
    V = mx.random.normal((B, Hkv, k_len, Dh))
    q = mx.random.normal((B, Hq, 1, Dh))
    scale = Dh ** -0.5

    # fabricate a selection: pick TOPK blocks (use indexer's selection on random h
    # is fine, but to make the test deterministic & meaningful pick blocks {0, last})
    sel_blocks = sorted(set([0, nb - 1]))[:TOPK]
    # represent as inds [B,1,topk] padded with -1 if fewer
    inds_list = sel_blocks + [-1] * (TOPK - len(sel_blocks))
    inds = mx.array(inds_list, dtype=mx.int32).reshape(1, 1, TOPK)

    # --- DENSE path: build_block_mask over k_len, full SDPA ---
    dense_mask = idx.build_block_mask(inds, nb, k_len, offset, 1, mx.float32)
    Kg = mx.repeat(K, Hq // Hkv, axis=1)
    Vg = mx.repeat(V, Hq // Hkv, axis=1)
    out_dense = minimax_m3.scaled_dot_product_attention(
        q, Kg, Vg, cache=None, scale=scale, mask=dense_mask
    )
    mx.eval(out_dense)

    # --- GATHER path: take_along_axis the selected blocks, fixed T_kv=topk*block,
    #     additive mask = selected ∧ causal ∧ valid; SDPA on subset ---
    Tkv = TOPK * BLK
    # build per-key gather index: for each selected block b, keys [b*BLK : b*BLK+BLK]
    safe = mx.where(inds < 0, 0, inds)  # clamp -1 -> 0 (will be -inf masked)
    # key indices into the k_len axis: [B,1,Tkv]
    block_starts = (safe * BLK)  # [B,1,topk]
    within = mx.arange(BLK)      # [BLK]
    key_idx = (block_starts[..., None] + within[None, None, :, None].reshape(1,1,1,BLK)).reshape(1, 1, Tkv)
    # clamp key indices that fall in the pad region (>= k_len) to a safe index
    key_overflow = key_idx >= k_len
    key_idx_safe = mx.where(key_overflow, 0, key_idx)
    # gather K/V along key axis with broadcast over heads
    gi = mx.broadcast_to(key_idx_safe[:, :, :, None], (B, Hkv, Tkv, Dh))
    Ksub = mx.take_along_axis(K, gi, axis=2)
    Vsub = mx.take_along_axis(V, gi, axis=2)
    Ksub = mx.repeat(Ksub, Hq // Hkv, axis=1)
    Vsub = mx.repeat(Vsub, Hq // Hkv, axis=1)
    # additive mask on the subset: valid slot (block index >=0) ∧ key<k_len ∧ causal
    q_pos = offset  # single query at last position
    valid_slot = (inds >= 0)  # [B,1,topk]
    valid_key = mx.broadcast_to(valid_slot[..., None], (B, 1, TOPK, BLK)).reshape(1, 1, Tkv)
    not_overflow = ~key_overflow
    causal = key_idx <= q_pos
    keep = valid_key & not_overflow & causal  # [B,1,Tkv]
    sub_mask = mx.where(
        keep[:, None, :, :].reshape(B, 1, 1, Tkv),
        mx.array(0.0, dtype=mx.float32),
        mx.array(mx.finfo(mx.float32).min, dtype=mx.float32),
    )
    out_gather = minimax_m3.scaled_dot_product_attention(
        q, Ksub, Vsub, cache=None, scale=scale, mask=sub_mask
    )
    mx.eval(out_gather)

    d = float(np.abs(np.array(out_dense) - np.array(out_gather)).max())
    return d, nb, pad, sel_blocks

for k_len in [THRESH, THRESH + 1, THRESH + BLK + 1, 2 * THRESH + 1]:
    d, nb, pad, sel = dense_vs_gather(k_len)
    flag = "OK " if d < 1e-6 else "FAIL"
    print(f"   [{flag}] k_len={k_len:>3} num_blocks={nb} pad={pad} sel_blocks={sel} "
          f"max|dense-gather|={d:.3e}")
print()
print("=== summary ===")
print(f"(a) boundary k_len==threshold: num_blocks<=topk : {boundary_ok}")

# ---------------------------------------------------------------------------
# (c-bis) Same exactness check but using the REAL indexer selection at decode
#         (k_len=9 == prod 2049), so the partial last block is whatever the
#         indexer actually picks — and the gather still equals dense.
# ---------------------------------------------------------------------------
print()
print("=== (c-bis) dense==gather using the REAL indexer selection at decode ===")
def dense_vs_gather_realsel(k_len, seed):
    B = 1
    offset = k_len - 1
    Hq, Hkv, Dh = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    nb = num_blocks_of(k_len)
    mx.random.seed(seed)
    # cache: prefill offset idx-keys then decode 1
    cache = minimax_m3.M3CacheLayer()
    if offset > 0:
        cache.update_index(mx.random.normal((B, 1, offset, args.index_head_dim)))
        cache.offset = offset
    h = mx.random.normal((B, 1, args.hidden_size))
    inds, nb2, kl = idx(h, offset, cache)
    mx.eval(inds)
    sel = sorted(int(x) for x in np.array(inds).ravel() if x >= 0)

    K = mx.random.normal((B, Hkv, k_len, Dh)); V = mx.random.normal((B, Hkv, k_len, Dh))
    q = mx.random.normal((B, Hq, 1, Dh)); scale = Dh ** -0.5
    dense_mask = idx.build_block_mask(inds, nb, k_len, offset, 1, mx.float32)
    Kg = mx.repeat(K, Hq//Hkv, axis=1); Vg = mx.repeat(V, Hq//Hkv, axis=1)
    out_dense = minimax_m3.scaled_dot_product_attention(q, Kg, Vg, cache=None, scale=scale, mask=dense_mask)

    Tkv = TOPK * BLK
    safe = mx.where(inds < 0, 0, inds)
    key_idx = (safe[..., None]*BLK + mx.arange(BLK)[None,None,None,:]).reshape(1,1,Tkv)
    overflow = key_idx >= k_len
    gi = mx.broadcast_to(mx.where(overflow,0,key_idx)[:,:,:,None], (B,Hkv,Tkv,Dh))
    Ksub = mx.repeat(mx.take_along_axis(K, gi, axis=2), Hq//Hkv, axis=1)
    Vsub = mx.repeat(mx.take_along_axis(V, gi, axis=2), Hq//Hkv, axis=1)
    valid_key = mx.broadcast_to((inds>=0)[...,None], (B,1,TOPK,BLK)).reshape(1,1,Tkv)
    causal = key_idx <= offset
    keep = valid_key & (~overflow) & causal
    sub_mask = mx.where(keep[:,None,:,:].reshape(B,1,1,Tkv),
                        mx.array(0.0,dtype=mx.float32),
                        mx.array(mx.finfo(mx.float32).min,dtype=mx.float32))
    out_gather = minimax_m3.scaled_dot_product_attention(q, Ksub, Vsub, cache=None, scale=scale, mask=sub_mask)
    mx.eval(out_dense, out_gather)
    d = float(np.abs(np.array(out_dense)-np.array(out_gather)).max())
    return d, nb, sel

for seed in range(5):
    d, nb, sel = dense_vs_gather_realsel(9, seed)  # k_len=9 == prod 2049
    partial = (nb-1) in sel  # is the partial last block selected?
    flag = "OK " if d < 1e-6 else "FAIL"
    print(f"   [{flag}] seed={seed} k_len=9 num_blocks={nb} indexer_sel={sel} "
          f"partial_block_selected={partial} max|Δ|={d:.3e}")
