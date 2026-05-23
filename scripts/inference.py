import mlx.core as mx
from mlx_lm.utils import sharded_load
from mlx_lm import stream_generate
import time, socket, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else 'mlx-community/Qwen3.5-122B-A10B-8bit'
PROMPT = sys.argv[2] if len(sys.argv) > 2 else 'Bonjour ! Présente-toi en 3 phrases.'

t0 = time.time()
print(f'[{socket.gethostname()}] loading {REPO}...', flush=True)
group = mx.distributed.init(backend='jaccl')
model, tokenizer = sharded_load(REPO, tensor_group=group)
rank = group.rank()
print(f'[{socket.gethostname()}] rank {rank}/{group.size()} loaded in {time.time()-t0:.1f}s', flush=True)

messages = [{'role':'user', 'content': PROMPT}]
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
