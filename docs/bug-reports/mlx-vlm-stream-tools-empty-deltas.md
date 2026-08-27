# mlx-vlm `mlx_vlm.server`: streaming with `tools` produces empty deltas

**Target:** https://github.com/Blaizzy/mlx-vlm/issues

## Summary

When `stream:true` is combined with a non-empty `tools` array, every SSE
chunk emitted by `mlx_vlm.server` has `delta.content = ""` and
`delta.tool_calls = []`, even though `output_tokens` in the `usage` field
increments correctly. The model is generating tokens, but the streaming
serializer never surfaces them.

The same request with `stream:false` returns the complete tool call
correctly, so the bug appears to be in the streaming code path's tool
call handling.

## Reproduction

```bash
curl -N -X POST http://HOST:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "/path/to/Qwen3.6-35B-A3B-MLX-8bit",
    "messages": [{"role":"user","content":"What is the weather in Paris? Use the tool."}],
    "tools": [{"type":"function","function":{
      "name":"get_weather","description":"Get weather",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
    }}],
    "stream": true,
    "max_tokens": 80
  }'
```

### Actual output

```
data: {"choices":[{"index":0,"finish_reason":null,
  "delta":{"role":"assistant","content":"","tool_calls":[]}}],
  "usage":{"output_tokens":1,...}}

data: {"choices":[{"index":0,"finish_reason":null,
  "delta":{"role":"assistant","content":"","tool_calls":[]}}],
  "usage":{"output_tokens":2,...}}

# ... 26 chunks like this ...
```

All deltas are empty; `output_tokens` increments. The stream ends without
ever surfacing the tool call.

### Expected output

Either incremental tool_call chunks (one shard per token) or one complete
tool_call chunk, e.g.:

```
data: {"choices":[{"index":0,"finish_reason":null,
  "delta":{"role":"assistant","tool_calls":[{
    "index":0,"id":"...","type":"function",
    "function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}
  }]}}]}

data: {"choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}

data: [DONE]
```

## Workaround we use

In our proxy (Odysseus), when the upstream provider is `mlx-vlm` and the
request has `tools`, we force `stream:false` upstream, then re-emit the
full response as a single SSE chunk + `[DONE]`. Pseudocode:

```python
if is_stream and tools and prov_id == "mlx-vlm":
    unary = {**fwd, "stream": False}
    payload = await client.post(upstream_url, json=unary).json()
    yield "data: " + json.dumps(make_chunk_with_delta(payload)) + "\n\n"
    yield "data: " + json.dumps(make_chunk_with_finish(payload)) + "\n\n"
    yield "data: [DONE]\n\n"
```

The streaming client sees deltas as expected; just with a higher TTFT
because we wait for the full unary response.

## Hypothesis

The streaming generator probably calls a tool_call parser only at end-of-
stream, but the per-chunk serializer doesn't include the accumulated
tool_calls in any delta. Either:

1. Stream the accumulated tool_call(s) at end of generation as one final
   delta chunk, OR
2. Stream tool_calls incrementally (token-by-token shards) as OpenAI's
   reference servers do.

## Environment

- mlx-vlm: (your version)
- Model: `Qwen3.6-35B-A3B-MLX-8bit`
- Platform: macOS, Apple Silicon M3 Ultra
- Server: `mlx_vlm.server --model ... --host 0.0.0.0 --port 8080 --kv-bits 8 --max-kv-size 131072`
