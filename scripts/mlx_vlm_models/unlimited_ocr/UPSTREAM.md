# unlimited_ocr (vendored)

Source: Blaizzy/mlx-vlm `mlx_vlm/models/unlimited_ocr/` at main commit
5061bd7f3848f720cb2ba8d0bc1a1a181d733359 (PR #2601, merged 2026-10-27,
in no release yet). MIT, copyright mlx-vlm authors. Files copied unchanged except one local change:
`UnlimitedOCRProcessor.__call__` evaluates the arrays it returns. Without it
`mlx_vlm.server` aborts on the first image request ("There is no Stream(gpu, 2)
in current thread": lazy graph built on the request thread, evaluated on the
generation thread; same pattern as mlx-vlm #2352 / #1591).

Adds Unlimited-OCR-2 (page2d RoPE + DASH block-sparse attention, the per-layer
`dash_router` weights). Unlimited-OCR v1 is unchanged by the PR.

Overlays the pinned mlx-vlm, whose own unlimited_ocr is v1 only and refuses the
60 `dash_router` tensors. Its imports (models.base, models.cache, deepseekocr)
were checked against the pin; installed only over that pin:

overlays-pin: 87020830d4dde238f111ade6d592f204a89cd527

Drop this directory once MLX_VLM_REF moves to a commit that has PR #2601.
