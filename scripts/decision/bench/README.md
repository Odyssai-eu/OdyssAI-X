# d3 / Vela 3.0 fidelity bench

The MLX runtimes (`../d3/d3_mlx.py`, `../vela3/vela3_mlx.py`) checked against each package's own torch runtime on CPU.
Oracle side: a venv with torch + transformers (+ `accelerate`); MLX side: the node's `.venv-mlxvlm`.

    # d3: oracle bf16, oracle FP32 (the reference), MLX, then the distances
    python d3_oracle_dump.py <d3 package> oracle/d3-x.json
    python d3_oracle_dump.py <d3 package> oracle/d3-x-fp32.json fp32
    python d3_mlx_compare.py <decision dir> <MLX folder> oracle/d3-x.json      # same token ids? writes d3-x.mlx.json
    python d3_dump_diff.py oracle/d3-x.json oracle/d3-x-fp32.json             # upstream bf16 vs FP32
    python d3_dump_diff.py oracle/d3-x.mlx.json oracle/d3-x-fp32.json         # MLX bf16 vs FP32

    # Vela: the package's modeling_vela3 on CPU (base_path = the d3 package of its pinned revision), then MLX
    python vela_oracle_dump.py <vela package> <d3 base package> oracle/vela3-x.json
    python vela_mlx_compare.py <decision dir> <vela MLX folder> <vela package> oracle/vela3-x.json

Pass: the same token ids; MLX within the order of upstream's own bf16 distance to FP32 and the same argmax; Vela
spans, labels, routes and definitions equal.
