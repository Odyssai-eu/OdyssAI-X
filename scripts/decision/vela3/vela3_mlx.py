"""MLX runtime for vLLM Semantic Router Vela 3.0 (lite / mini / flash / turbo). Ours.

A Vela 3.0 package is a d3 base at a pinned revision, unchanged, plus gated LoRA adapters on its Linear layers, span
heads and label definitions. Upstream runs it through d3's torch runtime (modeling_vela3.py, vela3_span.py,
vela3_adapters.py, vela3_labeldef.py); here the base is ``d3_mlx.D3MLX`` and the torch parts are rewritten on MLX:
  * ``LoRALinear``: upstream's MultiLoRA in eval mode, y = base(x) + ((x.float() @ A.T) @ B.T * alpha / r).to(y.dtype),
    at most one adapter on (``GATE``). One question per forward pass, so upstream's row mask is the per-row choice of
    the adapter here; the other rows keep base(x) exactly as upstream.
  * ``SpanHeadV2`` / ``SpanEngine``: the span pass on the final-norm states under the single-block noncausal mask
    (what the span adapters were trained on, every size).
  * ``LabelDefiner``: the definition writer (greedy, causal, every adapter off) for writer "base" (tied base) and
    "grafted_head"; "names" writes nothing; "labeldef_writer" (another model) is not available here: names only, as
    upstream does when its writer fails.
  * ``decide``: lite / mini / flash (rows of a gated id answered with their adapter) and turbo (``decisions.attention``:
    every row through d3-turbo first, adapters off, then the gated rows again with their adapter).
The pure parts are upstream's own code, extracted verbatim by extract.sh: vela3_methods.py (span routing, thresholds,
span_answer, evidence decodes), vela3_render.py (span prompt, word units, windows), vela3_labels.py (definition
prompt and cache), span_decode_v2.py (byte-identical). The folder holds the base converted by mlx-vlm and the vela
package's own files (adapters, heads, vela3_config.json, label_definitions.json, router_questions.json).
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from d3 import d3_format, d3_mlx

from . import span_decode_v2
from .vela3_labels import LabelDefinerBase
from .vela3_methods import Vela3MiniModel
from .vela3_render import RMAX, Render, windows, word_first_tokens


class _Gate:
    name = None   # the active adapter, or None (all off); the server runs one request at a time on one thread


GATE = _Gate()


class active:
    """Adapter ``name`` on for the forwards inside the block (None: every adapter off)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.prev, GATE.name = GATE.name, self.name

    def __exit__(self, *exc):
        GATE.name = self.prev


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self._ad = {}  # name -> (A [r, in] fp32, B [out, r] fp32, scale); underscored: not a model parameter

    def __call__(self, x):
        y = self.base(x)
        ad = self._ad.get(GATE.name)
        if ad is None:
            return y
        A, B, s = ad
        return y + ((x.astype(mx.float32) @ A.T) @ B.T * s).astype(y.dtype)


def _dims(linear) -> tuple[int, int]:
    out_dim, cols = linear.weight.shape
    return out_dim, cols * 32 // linear.bits if isinstance(linear, nn.QuantizedLinear) else cols


def attach(text_model, name: str, path: Path) -> dict:
    """The adapter in ``path`` ('<module path>.A' / '.B' relative to the text model, metadata r / alpha) on its layers."""
    weights, meta = mx.load(str(path), return_metadata=True)
    r, alpha = int(meta["r"]), int(meta["alpha"])
    mods = sorted({k.rsplit(".", 1)[0] for k in weights})
    for mp in mods:
        parent_path, attr = mp.rsplit(".", 1)
        parent = text_model
        for part in parent_path.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        cur = getattr(parent, attr)
        if not isinstance(cur, LoRALinear):
            if not isinstance(cur, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(f"{mp} is not a Linear layer")
            cur = LoRALinear(cur)
            setattr(parent, attr, cur)
        A, B = weights[mp + ".A"], weights[mp + ".B"]
        out_dim, in_dim = _dims(cur.base)
        if tuple(A.shape) != (r, in_dim) or tuple(B.shape) != (out_dim, r):
            raise ValueError(f"adapter shape mismatch at {mp}")
        cur._ad[name] = (A.astype(mx.float32), B.astype(mx.float32), alpha / r)
    if len(weights) != 2 * len(mods):
        raise ValueError(f"adapter {name}: unpaired A / B tensors")
    return dict(modules=len(mods), r=r, alpha=alpha)


class SpanHeadV2(nn.Module):
    """Vela 2.0 span head (word x label bilinear + MLP), FP32 as upstream (autocast off)."""

    def __init__(self, hidden, d=256, slots=64):
        super().__init__()
        self.d = d
        self.wn, self.ln = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.slot = nn.Embedding(slots, hidden)
        self.K, self.Q = nn.Linear(hidden, d, bias=False), nn.Linear(hidden, d, bias=False)
        self.M, self.N = nn.Linear(hidden, d), nn.Linear(hidden, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)
        self.o = nn.Linear(hidden, 1)

    def __call__(self, words, labels):
        w = self.wn(words.astype(mx.float32))
        w = w - w.mean(0, keepdims=True)
        n = labels.shape[0]
        e = self.ln(labels.astype(mx.float32)) + self.slot(mx.minimum(mx.arange(n), self.slot.weight.shape[0] - 1))
        if n >= 2:
            e = e - e.mean(0, keepdims=True)
        k, q = self.K(w), self.Q(e)
        bil = (k @ q.T) / math.sqrt(self.d)
        mlp = self.v(nn.gelu(self.M(w)[:, None, :] + self.N(e)[None, :, :])).squeeze(-1)
        return bil + mlp, self.o(w).squeeze(-1), q


def _head(path: Path, hidden: int) -> SpanHeadV2:
    head = SpanHeadV2(hidden)
    head.load_weights(list(mx.load(str(path)).items()), strict=True)
    mx.eval(head.parameters())
    return head


class SpanEngine:
    """One span head on the final-norm states; its adapter is on for every forward of this engine (vela3_span.SpanEngine)."""

    def __init__(self, runtime, head, adapter="span"):
        self.rt, self.head, self.adapter = runtime, head, adapter
        self.R = Render(runtime.tokenizer, d3_format)

    def n_tokens(self, text):
        return len(self.rt.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _pieces(self, parts, target, question, labels):
        text = dict(parts)[target]
        e = self.rt.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = e["input_ids"]; offs = np.asarray(e["offset_mapping"], dtype=np.int64).reshape(-1, 2)
        if len(ids) <= RMAX:
            return [(self.R(parts, target, question, labels, text), 0)], len(ids)
        wo, ft = word_first_tokens(text, offs); out = []
        for a0, b0 in windows(ft, len(ids)):
            cs, ce = int(offs[a0][0]), int(offs[b0 - 1][1])
            out.append((self.R(parts, target, question, labels, text[cs:ce]), cs))
        return out, len(ids)

    def logits(self, parts, target, question, labels, images=None):
        """Word offsets (W, 2) into the target text, word x label logits (W, L), target tokens, windows."""
        if images:
            raise ValueError(f"image inputs are not available: {self.rt.image_unavailable}")
        acc = {}
        pieces, n_tok = self._pieces(parts, target, question, labels)
        for rec, cs in pieces:
            if len(rec["w_offs"]) == 0:
                continue
            with active(self.adapter):
                H = self.rt.states(rec["ids"], mode="noncausal_full_attention")
                mx.eval(H)
            w = H[mx.array(rec["w_pos"])]
            lab = mx.stack([H[s:e + 1].mean(0) for s, e in rec["ranges"]])
            lg = np.array(self.head(w, lab)[0], dtype=np.float32)
            for wi, (s, e) in enumerate(rec["w_offs"]):
                v = acc.setdefault((int(s) + cs, int(e) + cs), [np.zeros(len(labels)), 0]); v[0] += lg[wi]; v[1] += 1
        keys = sorted(acc)
        wo = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        lg = np.stack([acc[k][0] / acc[k][1] for k in keys]) if keys else np.zeros((0, len(labels)))
        return wo, lg, n_tok, len(pieces)


class LabelDefiner(LabelDefinerBase):
    """vela3_labeldef.LabelDefiner with its writer on MLX: greedy, causal, every adapter off, first line kept."""

    def __init__(self, runtime, root: Path, vc: dict):
        self.rt, self.root = runtime, root
        self.def_cfg = vc.get("definitions") or {}
        self.mem, self.head, self.writer = {}, None, None
        cache = root / self.def_cfg.get("cache_file", "label_definitions.json")
        if cache.exists():
            for e in json.loads(cache.read_text())["sets"]:
                self.mem[self._key(e["labels"])] = dict(e["definitions"])
        mode = os.environ.get("VELA3_DEF_WRITER") or self.def_cfg.get("writer", "base")
        text_cfg = runtime.backbone.config.text_config
        tied = getattr(text_cfg, "tie_word_embeddings", None)
        try:
            if mode == "names":
                return
            if mode == "base":
                assert tied, "refusing to write definitions with an untied base that has no lm_head"
                self.head = runtime.text_model.embed_tokens.as_linear
            elif mode == "grafted_head":
                weight = mx.load(str(root / self.def_cfg["lm_head_file"]))["lm_head.weight"].astype(mx.bfloat16)
                assert weight.shape == (text_cfg.vocab_size, text_cfg.hidden_size), "grafted head does not fit the backbone"
                self.head = lambda h: h @ weight.T
            else:
                raise ValueError(f"definitions.writer {mode!r} is not available on MLX")
            self.writer = mode
        except Exception as e:  # noqa: BLE001 — names only for uncached labels, never a definition from an untied head
            warnings.warn(f"vela3_labeldef: no definition writer ({mode}: {type(e).__name__}: {e}); uncached labels run names only")
            self.head = None
        eos = getattr(text_cfg, "eos_token_id", None)
        self.stop = set(eos if isinstance(eos, list) else [eos]) | {runtime.tokenizer.convert_tokens_to_ids("<|im_end|>")}

    def generate(self, prompts):
        if self.head is None:
            return [None] * len(prompts)
        tok, lm = self.rt.tokenizer, self.rt.text_model
        out = []
        for p in prompts:
            txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            ids = tok(txt, add_special_tokens=False)["input_ids"]
            cache = self.rt.backbone.language_model.make_cache()
            new = []
            with active(None):
                h = lm(mx.array([ids]), cache=cache)
                for _ in range(40):
                    t = int(mx.argmax(self.head(h[:, -1]), axis=-1).item())
                    if t in self.stop:
                        break
                    new.append(t)
                    h = lm(mx.array([[t]]), cache=cache)
            out.append(tok.decode(new, skip_special_tokens=True).strip().split("\n")[0].strip())
        return out


class Vela3MLX(Vela3MiniModel):
    """A Vela 3.0 package on MLX: system_one(state, questions) as upstream, choice / noul / score / span."""

    def __init__(self, root: str | Path):
        root = Path(root)
        vc = json.loads((root / "vela3_config.json").read_text())
        self.vela3_config, self.root = vc, root
        self.rt = d3_mlx.load(root)
        hidden = self.rt.backbone.config.text_config.hidden_size
        for name, spec in vc["adapters"].items():
            info = attach(self.rt.text_model, name, root / spec["file"])
            if (info["r"], info["alpha"], info["modules"]) != (spec["r"], spec["alpha"], spec["modules"]):
                raise ValueError(f"adapter {name} does not match vela3_config.json: {info}")
        sp = vc["span"]
        self.span = SpanEngine(self.rt, _head(root / sp["head_file"], hidden))
        self.span_ner = SpanEngine(self.rt, _head(root / sp["ner"]["head_file"], hidden), "ner") if sp.get("ner") else None
        ih = sp.get("image_halu") or {}
        self.span_img = (SpanEngine(self.rt, _head(root / ih["head_file"], hidden), "image_halu")
                         if ih.get("head_file") and "image_halu" in vc["adapters"] else None)
        rl = sp.get("relation") or {}
        self.span_rel = (SpanEngine(self.rt, _head(root / rl["head_file"], hidden), "relation")
                         if rl.get("head_file") and "relation" in vc["adapters"] else None)
        self.defs = LabelDefiner(self.rt, root, vc)
        self._decode = span_decode_v2
        self.router_qids = frozenset(vc["router_question_ids"])
        rq = json.loads((root / vc["router_questions_file"]).read_text())["trained_questions"]
        self.router_wordings = {}
        for qid, v in rq.items():  # trained instruction texts per id (intent: MASSIVE and SGD)
            vs_ = [v] if "question" in v else list(v.values())
            self.router_wordings[qid] = {" ".join(x["question"]["instructions"].split()) for x in vs_}
        self.router_gate = vc.get("router_gate", "id")
        self.turbo = "decisions" in vc
        self.model_name = "vllm-sr/" + (vc.get("size") or {}).get("name", "Vela-3.0")

    # ---------------------------------------------------------------- decisions
    def _gates(self, prepared, raw):
        """Adapter per gated row: router ids (decision), task ids (their own adapter), opt-in rows (decision_optin)."""
        vc = self.vela3_config
        optin = {k for k, q in raw.items() if isinstance(q, dict) and q.get("decision_adapter") is True} if vc.get("decision_opt_in") else set()
        router = {k for k in prepared.runnable if self._router_row(k, raw.get(k))}
        _ta = vc.get("decision_task_adapters") or {}  # RDF: domain / feedback rows -> their own adapter, own pass
        task = {k: _ta[k] for k in router if k in _ta}
        if not self.turbo:  # RDF2: "prefix:" keys (lite; no other package declares one)
            task.update({k: v for p, v in _ta.items() if p.endswith(":") for k in prepared.runnable if k.startswith(p) and k not in router})
        router = router - set(task)
        oname = "decision_optin" if "decision_optin" in vc["adapters"] else "decision"
        opt = {k for k in prepared.runnable if k in optin and k not in router}
        if oname == "decision":
            router, opt = router | opt, set()
        gate = {k: "decision" for k in router}
        gate.update(task)
        gate.update({k: oname for k in opt})
        return gate

    def decide_probs(self, state, questions, images=None):
        """(prepared, probabilities per runnable question in d3's option order, input tokens) of the decision questions."""
        rt = self.rt
        raw = questions  # "decision_adapter" is ours, never passed to d3
        questions = {k: ({a: b for a, b in q.items() if a != "decision_adapter"} if isinstance(q, dict) else q) for k, q in questions.items()}
        prepared = rt.prepare(state, questions, images)
        gate = self._gates(prepared, raw)
        if self.turbo or not gate:
            probs, tokens = rt.run(prepared)
            probs = dict(probs)
        else:
            probs, tokens = {}, sum(len(prepared.sequences[k]) for k in prepared.runnable)
        mode = (self.vela3_config.get("decisions") or {}).get("attention", "prefix_bidir")
        for k in prepared.runnable:
            if self.turbo and k not in gate:
                continue
            seq, count = prepared.sequences[k], len(prepared.questions[k].keys)
            with active(gate.get(k)):
                if self.turbo and mode == "noncausal":
                    probs[k] = rt.probabilities([seq], [count], mode="noncausal_full_attention")[0]
                else:
                    probs[k] = rt.probabilities([seq], [count], [prepared.states[k]] if rt.blocked else None)[0]
            if self.turbo:
                tokens += len(seq)
        return prepared, probs, tokens

    def decide(self, state, questions, images=None):
        prepared, probs, tokens = self.decide_probs(state, questions, images)
        return self.rt.respond(prepared, probs, tokens)

    def span_error(self, message):
        err = {"type": "span", "error": "invalid_question", "message": message}
        return {**err, "reason": "question_malformed"} if self.turbo else err

    def system_one(self, *, state: Any, questions: dict[str, Any], images: list[Any] | None = None) -> dict[str, Any]:
        """Choice / Noul / Score questions as d3 (decision adapter on the router question ids), plus questions of type "span"."""
        span_q = {k: q for k, q in questions.items() if isinstance(q, dict) and q.get("type") == "span"}
        rest = {k: q for k, q in questions.items() if k not in span_q}
        imgs, img_err = None, None
        if span_q and images:  # decoded once, shared by every span question of the request
            try:
                imgs = self.rt.load_images(images)
            except ValueError as e:
                img_err = str(e)
        if rest:
            res = self.decide(state, rest, images)
        else:
            res = {"model": self.rt.model_name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        answers = dict(res["answers"])
        for k, q in span_q.items():
            try:
                if img_err is not None:
                    raise ValueError(img_err)
                answers[k] = self.span_answer(state, q, imgs)
            except ValueError as e:
                answers[k] = self.span_error(str(e))
        res["answers"] = {k: answers[k] for k in questions}
        res["model"] = self.model_name
        return res


def load(root: str | Path) -> Vela3MLX:
    return Vela3MLX(root)
