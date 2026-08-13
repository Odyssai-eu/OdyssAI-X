# mlx-vlm `mlx_vlm.server`: `finish_reason="stop"` even when `tool_calls` are present

**Target:** https://github.com/Blaizzy/mlx-vlm/issues

## Summary

When the model emits tool calls, `mlx_vlm.server`'s OpenAI-compat chat
completion endpoint sets `choices[0].finish_reason = "stop"`. Per the
OpenAI spec it should be `"tool_calls"` whenever `message.tool_calls` /
`delta.tool_calls` is non-empty, otherwise agent loops in clients
(Companion, LangChain, etc.) never trigger tool execution.

## Reproduction

```bash
curl -X POST http://HOST:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "/path/to/Qwen3.6-35B-A3B-MLX-8bit",
    "messages": [{"role":"user","content":"What is the weather in Paris? Use the tool."}],
    "tools": [{
      "type":"function",
      "function":{
        "name":"get_weather",
        "description":"Get weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
      }
    }],
    "stream": false
  }'
```

### Actual response

```json
{
  "choices": [{
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "...",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
      }]
    }
  }]
}
```

### Expected response

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [...]
    }
  }]
}
```

## Reference

OpenAI Chat Completions API spec, `finish_reason` enum values:

- `stop`: stopped at a natural stop point or sequence
- `length`: max tokens reached
- `tool_calls`: model called a tool
- `content_filter`: omitted due to content filter

Source: https://platform.openai.com/docs/api-reference/chat/object#chat/object-choices

## Impact

Clients that implement an agent loop (Companion, LangChain, LangGraph,
autogen, OpenAI SDK auto-tool flows, …) gate the tool execution step on
`finish_reason === "tool_calls"`. With `mlx_vlm.server` returning
`"stop"`, these clients silently exit the loop after the first turn,
never invoking the tool, leaving the user with an empty assistant
response.

We work around this in our proxy (Odysseus) by post-processing every
response from `mlx_vlm.server` and rewriting `finish_reason` when
`tool_calls` is non-empty.

## Environment

- mlx-vlm: (your version)
- Model: `Qwen3.6-35B-A3B-MLX-8bit`
- Platform: macOS, Apple Silicon M3 Ultra
- Server: `mlx_vlm.server --model ... --host 0.0.0.0 --port 8080`

## Possible fix location

`mlx_vlm.server` chat completion handler, around the response building step.
Likely a 1-line check:

```python
finish_reason = "tool_calls" if tool_calls else stop_reason
```
