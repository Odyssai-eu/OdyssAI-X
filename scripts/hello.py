import mlx.core as mx
import socket

g = mx.distributed.init(backend="jaccl", strict=True)
print(f'rank {g.rank()}/{g.size()} on {socket.gethostname()}', flush=True)

x = mx.array([float(g.rank())])
total = mx.distributed.all_sum(x, group=g)
mx.eval(total)
expected = g.size() * (g.size() - 1) / 2
print(f'rank {g.rank()} all_sum={total.item()} (expected {expected})', flush=True)
