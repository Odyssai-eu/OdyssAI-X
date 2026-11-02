"""Word-aware span decoding for the schema encoder's sigmoid span head (PII long-doc fix, 26 Sep).

The v1 decoder (span_data.decode) thresholds every token x label independently and turns each above-threshold
run into a span. With sub-word tokens (the Vela tokenizer splits digits one by one) a single below-threshold
piece fragments one entity into several false positives ('ois', '0', 'example'), and every label is decoded
separately, so overlapping labels each cost a false positive.

decode_v2(probs, offs, names, text, thr, ...) -> [dict(start, end, label)] (code-point offsets)
  1. label resolution: each token takes its argmax label if that probability > thr, else O
     (overlapping labels resolved by max score; upstream Vela PII argmax-BIO likewise gives one label per token).
  2. word units from the text (GLiNER2 word splitter, see _UNIT): URL | run of letters/digits/@._-+ | any other
     single non-space char; every CJK / kana / hangul / full-width character is its own unit (character level).
     A token belongs to the unit that holds its first non-space character.
  3. within a unit, one label: the label with the largest summed probability over the unit's labelled tokens
     wins and the other labelled tokens of the unit are relabelled to it.
  4. fill: unlabelled tokens BETWEEN two tokens of the same label inside one unit take that label;
     extension (extend="unit", default): to the outermost letter/digit tokens of the unit, through inner
     punctuation ('ois' -> 'lois123@x.example.org'), never onto a bare edge '[' / ']' / '.';
     extend="alnum": only through directly adjacent letter/digit tokens; extend=False: off.
  5. contiguous tokens of one label form a span (never joins different labels: upstream decode_entities rule);
     whitespace is trimmed (upstream), and so are enclosing bracket / quote characters at the span edges.
  6. optional: bridge=True joins two same-label spans separated by one unit whose probability for that
     label is > thr * bridge_frac; min_chars={label: n} drops shorter spans.
Upstream semantics kept: exact code-point spans, whitespace trim, different types never joined. Not
reproducible from a sigmoid head: the B- tag that splits two adjacent same-type entities (it has no B/I).
"""
import re
import numpy as np

_CJK = ("⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿ꀀ-꓏"
        "가-힯豈-﫿︰-﹏＀-￯")
# Word units = GLiNER2's splitters (gliner2/processing/word_splitter.py, Apache-2.0): WhitespaceTokenSplitter's URL
# rule, CharLevelSplitter's [A-Za-z0-9@._\-+]+ runs (e-mails, IPs, phones, IDs stay one word), widened to Unicode
# letters/digits (Straße, título), and every CJK / kana / hangul / full-width character on its own (char level).
_UNIT = re.compile(rf"(?:https?://|www\.)[^\s\]\)\}}>「」『』\"'“”{_CJK}]+"
                   rf"|(?:(?![{_CJK}])[^\W_]|[@._\-+])+"
                   rf"|\S")
_ALNUM = re.compile(r"[^\W_]")
EDGE = set("[](){}<>「」『』【】《》〈〉\"'“”‘’«»‹›")


def units(text):
    """char index -> unit id (-1 for whitespace)."""
    u = np.full(len(text) + 1, -1, dtype=np.int64)
    for k, m in enumerate(_UNIT.finditer(text)):
        u[m.start():m.end()] = k
    return u


def _first_nonspace(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    return a


def trim(text, s, e):
    while s < e and (text[s].isspace() or text[s] in EDGE):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in EDGE):
        e -= 1
    return s, e


def decode_v2(probs, offs, names, text, thr, fill=True, extend="unit", unit_vote=True, trim_edges=True,
              bridge=False, bridge_frac=0.5, min_chars=None, u=None):
    probs = np.asarray(probs, dtype=np.float32); offs = np.asarray(offs)
    T = len(offs)
    if T == 0:
        return []
    valid = offs[:, 1] > offs[:, 0]
    k = probs.argmax(1); pk = probs[np.arange(T), k]
    lab = np.where(valid & (pk > thr), k, -1)
    u = units(text) if u is None else u
    tu = np.full(T, -1, dtype=np.int64)
    alnum = np.zeros(T, dtype=bool)
    for t in range(T):
        if not valid[t]:
            continue
        a, b = int(offs[t, 0]), int(offs[t, 1])
        f = _first_nonspace(text, a, b)
        tu[t] = u[f] if f < b and f < len(text) else -1
        alnum[t] = bool(_ALNUM.search(text[a:b]))
    # group tokens by unit (tokens are in text order, units contiguous)
    if unit_vote or fill or extend:
        t = 0
        while t < T:
            if tu[t] < 0:
                t += 1; continue
            t2 = t
            while t2 + 1 < T and (tu[t2 + 1] == tu[t] or not valid[t2 + 1]):
                t2 += 1
            idx = [i for i in range(t, t2 + 1) if valid[i]]
            L = [i for i in idx if lab[i] >= 0]
            if L:
                if unit_vote and len({int(lab[i]) for i in L}) > 1:
                    sc = {}
                    for i in L:
                        sc[int(lab[i])] = sc.get(int(lab[i]), 0.0) + float(pk[i])
                    w = max(sc, key=sc.get)
                    for i in L:
                        lab[i] = w
                if fill:
                    for c in set(int(lab[i]) for i in L):
                        pos = [i for i in L if lab[i] == c]
                        for i in idx:
                            if pos[0] < i < pos[-1] and lab[i] < 0:
                                lab[i] = c
                if extend:
                    for c in set(int(lab[i]) for i in idx if lab[i] >= 0):
                        pos = [idx.index(i) for i in idx if lab[i] == c]
                        if extend == "unit":   # to the outermost letter/digit tokens of the unit, through inner punctuation
                            an = [j for j, i in enumerate(idx) if alnum[i]]
                            lo, hi = min([pos[0]] + an[:1]), max([pos[-1]] + an[-1:])
                            if all(lab[idx[j]] < 0 or lab[idx[j]] == c for j in range(lo, hi + 1)):
                                for j in range(lo, hi + 1):
                                    lab[idx[j]] = c
                        else:                  # "alnum": only through adjacent letter/digit tokens
                            j = pos[0] - 1
                            while j >= 0 and lab[idx[j]] < 0 and alnum[idx[j]]:
                                lab[idx[j]] = c; j -= 1
                            j = pos[-1] + 1
                            while j < len(idx) and lab[idx[j]] < 0 and alnum[idx[j]]:
                                lab[idx[j]] = c; j += 1
            t = t2 + 1
    # contiguous runs of one label (skipping zero-width tokens)
    spans = []
    cur = None
    for t in range(T):
        if not valid[t]:
            continue
        c = int(lab[t])
        if cur is not None and c == cur[0]:
            cur[2] = int(offs[t, 1]); cur[3].append(t); continue
        if cur is not None:
            spans.append(cur)
        cur = [c, int(offs[t, 0]), int(offs[t, 1]), [t]] if c >= 0 else None
    if cur is not None:
        spans.append(cur)
    if bridge and len(spans) > 1:
        out = [spans[0]]
        for sp in spans[1:]:
            pv = out[-1]
            if sp[0] == pv[0]:
                gap = [t for t in range(pv[3][-1] + 1, sp[3][0]) if valid[t]]
                gu = {int(tu[t]) for t in gap if tu[t] >= 0}
                if gap and len(gu) <= 1 and all(probs[t, sp[0]] > thr * bridge_frac for t in gap if tu[t] >= 0):
                    pv[2] = sp[2]; pv[3] = pv[3] + gap + sp[3]; continue
            out.append(sp)
        spans = out
    res = []
    for c, s, e, _ in spans:
        s, e = trim(text, s, e) if trim_edges else trim_ws(text, s, e)
        if e <= s:
            continue
        if min_chars and e - s < min_chars.get(names[c], 0):
            continue
        res.append(dict(start=s, end=e, label=names[c]))
    return res


def trim_ws(text, s, e):
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def decode_v1_trimmed(probs, offs, names, text, thr):
    """The scorer's current rule (span_data.decode + whitespace trim), for before/after tables."""
    from span_data import decode
    out = []
    for s in decode(probs, offs, names, thr):
        a, b = trim_ws(text, s["start"], s["end"])
        if b > a:
            out.append(dict(start=a, end=b, label=s["label"]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# row-level API (rows = dicts with text, spans [{start,end,label}], names, offs, probs; see dump_span_probs_v2.py)
# ---------------------------------------------------------------------------------------------------------------
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
DECODERS = {
    "v1": lambda r, thr: decode_v1_trimmed(r["probs"], r["offs"], r["names"], r["text"], thr),
    "v2": lambda r, thr: decode_v2(r["probs"], r["offs"], r["names"], r["text"], thr),
    "v2_noext": lambda r, thr: decode_v2(r["probs"], r["offs"], r["names"], r["text"], thr, extend=False),
    "v2_alnum": lambda r, thr: decode_v2(r["probs"], r["offs"], r["names"], r["text"], thr, extend="alnum"),
    "v2_notrim": lambda r, thr: decode_v2(r["probs"], r["offs"], r["names"], r["text"], thr, trim_edges=False),
    "v2_bridge": lambda r, thr: decode_v2(r["probs"], r["offs"], r["names"], r["text"], thr, bridge=True),
    "v2_argmax_only": lambda r, thr: decode_v2(r["probs"], r["offs"], r["names"], r["text"], thr, fill=False,
                                               extend=False, unit_vote=False, trim_edges=False),
}


def decode_rows(rows, thr, decoder="v2"):
    f = DECODERS[decoder] if isinstance(decoder, str) else decoder
    return [f(r, thr) for r in rows]


def _psd(span_data_path):
    """Load the upstream semantic-router span_data.py (src/training/model_classifier/pii_model_fine_tuning_lora/span_data.py)
    from span_data_path. Eval helper only (dev threshold picks); inference never calls it."""
    import importlib.util
    from pathlib import Path
    if span_data_path is None:
        raise ValueError("pii_metrics needs span_data_path: the path of the upstream span_data.py")
    spec = importlib.util.spec_from_file_location("pii_span_data_up", Path(span_data_path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PII_TYPES = ["AGE", "CREDIT_CARD", "DATE_TIME", "DOMAIN_NAME", "EMAIL_ADDRESS", "GPE", "IBAN_CODE", "IP_ADDRESS", "NRP",
             "ORGANIZATION", "PERSON", "PHONE_NUMBER", "STREET_ADDRESS", "TITLE", "US_DRIVER_LICENSE", "US_SSN", "ZIP_CODE"]


def pii_metrics(rows, preds, span_data_path):
    """Upstream Vela PII exact-entity metric (span_data.entity_metrics, whitespace-trimmed gold); span_data_path: see _psd."""
    P = _psd(span_data_path)
    recs = [dict(full_text=r["text"], spans=[dict(start_position=s["start"], end_position=s["end"], entity_type=s["label"],
                 entity_value=r["text"][s["start"]:s["end"]]) for s in r["spans"]]) for r in rows]
    return P.entity_metrics(recs, [{(s["label"], s["start"], s["end"]) for s in p} for p in preds], PII_TYPES)


def toxic_score(rows, preds):
    from span_data import char_f1
    return float(np.mean([char_f1(p, r["spans"]) for r, p in zip(rows, preds)]))


def halu_metrics(rows, preds):
    from train_schema_span import ragtruth_metrics_spans
    return ragtruth_metrics_spans([dict(text=r["text"], spans=r["spans"]) for r in rows], preds)


def family_objective(family, span_data_path=None):
    """Dev objective per span label family (what the threshold is chosen to maximise); span_data_path: PII only (see _psd)."""
    if family == "pii":
        return lambda rows, preds: pii_metrics(rows, preds, span_data_path)["micro"]["f1"]
    if family == "toxic":
        return toxic_score
    if family == "halu":
        return lambda rows, preds: halu_metrics(rows, preds)["example_f1"]
    raise KeyError(family)


def pick_threshold(dev_sets, family, decoder="v2", grid=GRID, span_data_path=None):
    """Choose one threshold per label family on DEV rows only.
    dev_sets: list of row lists; the objective is the unweighted mean over the sets (for PII: dev short, dev 4096,
    dev 8192 -> upstream's length-macro selection, so the dense long rows cannot dominate). Returns (thr, curve)."""
    obj = family_objective(family, span_data_path)
    curve = {t: float(np.mean([obj(rows, decode_rows(rows, t, decoder)) for rows in dev_sets])) for t in grid}
    best = max(curve.values())
    thr = min(t for t, v in curve.items() if v >= best - 1e-12)  # ties -> lowest threshold
    return thr, curve


if __name__ == "__main__":  # self-test on the fragment cases seen in run C
    text = "Le formulaire indique [pavel_200015@summit15.example.org] dans adresse courriel."
    names = ["EMAIL_ADDRESS", "PERSON"]
    import unicodedata  # noqa
    # fake token offsets: split at every non-alnum boundary
    offs = [(m.start(), m.end()) for m in re.finditer(r"\s?\w+|\s?[^\w\s]", text)]
    p = np.zeros((len(offs), 2), dtype=np.float32)
    s0 = text.index("pavel"); e0 = text.index("]")
    for i, (a, b) in enumerate(offs):
        if a >= s0 - 1 and b <= e0 + 1:
            p[i, 0] = 0.9 if i % 3 else 0.2        # fragmented
    p[[i for i, (a, b) in enumerate(offs) if text[a:b].strip() == "["][0], 0] = 0.8  # bracket above threshold
    out = decode_v2(p, np.array(offs), names, text, 0.5)
    assert [(o["label"], text[o["start"]:o["end"]]) for o in out] == [("EMAIL_ADDRESS", "pavel_200015@summit15.example.org")], out
    t2 = "表格在日期一栏填写了「2015-06-06」。"
    print(units(t2)[:14], "ok")
