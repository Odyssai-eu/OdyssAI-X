#!/usr/bin/env python3
"""test_eagle_mistral.py — bench EAGLE speculative sur Telemak (.32:8003).

Charge supposee deja faite (target Q8 + draft EAGLE paires). Mesure :
  - tok/s decode sur N runs (prompt fixe, greedy temp=0) vs baseline 5.33 ;
  - acceptance_rate du draft (via /.well-known/inference-engine.json) ;
  - coherence du texte (le speculatif est EXACT -> sortie = greedy de la cible,
    donc le francais doit etre correct quel que soit le draft ; c'est le tok/s
    qui revele si le draft accelere).
"""
import json
import time
import urllib.request

BASE = "http://localhost:8003"
PROMPT = "Explique en detail le cycle de l'eau sur Terre, etape par etape."
BASELINE = 5.33  # Telemak Swift Q8 sans speculatif (valide WU0/WU1)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(mid, max_tokens):
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", {
        "model": mid,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0})
    dt = time.perf_counter() - t0
    ct = resp["usage"]["completion_tokens"]
    return ct, dt, resp


def main():
    mid = get("/v1/models")["data"][0]["id"]
    print("model:", mid)

    print("warmup...")
    run(mid, 16)

    print(f"\nmeasured (max_tokens=256, temp=0) vs baseline {BASELINE} tok/s :")
    rates = []
    last = None
    for i in range(3):
        ct, dt, resp = run(mid, 256)
        r = ct / dt
        rates.append(r)
        last = resp
        print(f"  run {i+1}: {ct} tok / {dt:.2f}s = {r:.2f} tok/s")
    mean = sum(rates) / len(rates)
    speedup = mean / BASELINE
    print(f"  MEAN: {mean:.2f} tok/s  ({speedup:.2f}x baseline)")

    print("\ncoherence (debut de la generation) :")
    txt = last["choices"][0]["message"]["content"]
    print("  " + txt[:300].replace("\n", " "))

    print("\nspeculative telemetry :")
    try:
        spec = get("/.well-known/inference-engine.json").get("speculative_decoding", {})
        print(json.dumps(spec, indent=2, ensure_ascii=False))
    except Exception as e:
        print("  well-known:", e)
    try:
        print("avg_tok_s_recent:", get("/health").get("avg_tok_s_recent"))
    except Exception as e:
        print("  health:", e)


if __name__ == "__main__":
    main()
