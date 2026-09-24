#!/usr/bin/env python3
"""laya-serve on local checkpoints — the decision service behind OdyssAI-X's
`protocol: systemone` providers.

Upstream `laya-serve` resolves checkpoint names to Hugging Face repos and
downloads them. On our nodes the weights already sit on the models volume, so
this launcher builds the same `laya.Router` with `models=` pointing at local
directories (the constructor's documented override) and hands it to
`laya.serve.create_app(router)` (its documented injection point). Wire protocol,
auth and health probe are laya-serve's own: `POST /v1/systemone`, `GET /health`.

Environment (in addition to laya-serve's LAYA_HOST/LAYA_PORT/LAYA_DEVICE/
LAYA_API_KEY/LAYA_THREADS/LAYA_LOG_LEVEL):
  LAYA_MODELS_ROOT  directory holding laya, laya-multilingual,
                    laya-typed-decisions   (default /Volumes/models/odysseus/convaiinnovations)
  LAYA_MODELS       checkpoints to preload (default multilingual)
  LAYA_DEFAULT      checkpoint used when a request names none (default multilingual)
  LAYA_MAX_LEN      context window in tokens for the preloaded checkpoints
                    (0 = checkpoint default, 1024 for multilingual; up to 8192)
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never download: weights are local

import laya
from laya import serve

ROOT = os.environ.get("LAYA_MODELS_ROOT", "/Volumes/models/odysseus/convaiinnovations")
LOCAL = {
    "english": os.path.join(ROOT, "laya"),
    "multilingual": os.path.join(ROOT, "laya-multilingual"),
    "typed-decisions": os.path.join(ROOT, "laya-typed-decisions"),
}


def build_router():
    models = {name: path for name, path in LOCAL.items() if os.path.isdir(path)}
    if "multilingual" not in models:
        raise SystemExit(f"laya-multilingual not found under {ROOT}")
    preload = [m.strip() for m in os.environ.get("LAYA_MODELS", "multilingual").split(",") if m.strip()]
    router = laya.Router(
        models=models,
        device=os.environ.get("LAYA_DEVICE") or None,
        default=os.environ.get("LAYA_DEFAULT", "multilingual"),
        max_loaded=int(os.environ.get("LAYA_MAX_LOADED", "1")),
    )
    router.preload(preload)
    # Context window of the preloaded checkpoints. The shipped config says
    # 1024 tokens for multilingual (mmBERT accepts up to 8192): a bench answer
    # longer than that was cut silently and a fact at its end was never seen
    # (2026-09-24). LAYA_MAX_LEN raises it; head_max_len (option budget) stays.
    max_len = int(os.environ.get("LAYA_MAX_LEN", "0") or 0)
    if max_len:
        for name in preload:
            router.load(name).cfg["max_len"] = max_len
    return router


def main() -> None:
    import uvicorn

    serve._apply_thread_limit()
    uvicorn.run(
        serve.create_app(build_router()),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAYA_PORT", "8790")),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
