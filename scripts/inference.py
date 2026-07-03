"""Standalone tensor-parallel smoke script — run directly, never imported.

    python inference.py MODEL [PROMPT]

#29: the distributed init + sharded_load used to run at MODULE TOP LEVEL, so a
bare `import inference` on a non-JACCL node crashed immediately (same class as
the 2026-05-26 auto_parallel regression). Everything now lives under main() +
the `__main__` guard, so importing the module is a harmless no-op. Still shipped
to nodes by bootstrap-node.sh / provision-node-local.sh as a hand-run diagnostic.
"""
import socket
import sys
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else 'mlx-community/Qwen3.5-122B-A10B-8bit'
    prompt_text = sys.argv[2] if len(sys.argv) > 2 else 'Bonjour ! Présente-toi en 3 phrases.'

    t0 = time.time()
    print(f'[{socket.gethostname()}] loading {repo}...', flush=True)
    group = mx.distributed.init(backend='jaccl')
    model, tokenizer = sharded_load(repo, tensor_group=group)
    rank = group.rank()
    print(f'[{socket.gethostname()}] rank {rank}/{group.size()} loaded in {time.time()-t0:.1f}s', flush=True)

    messages = [{'role': 'user', 'content': prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    t1 = time.time()
    ntoks = 0
    for res in stream_generate(model, tokenizer, prompt, max_tokens=200):
        if rank == 0:
            print(res.text, end='', flush=True)
        ntoks += 1
    elapsed = time.time() - t1
    if rank == 0:
        print(f'\n--- {ntoks} tokens in {elapsed:.1f}s = {ntoks/elapsed:.1f} tok/s ---', flush=True)


if __name__ == "__main__":
    main()
