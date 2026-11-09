base = "VENV_PLACEHOLDER/lib/python3.12/site-packages/mlx_vlm/"
mark = "# OdyssAI-X #221"
atem_path, srv_path = base + "tools/parsers/atem.py", base + "server/openai.py"
with open(atem_path) as f:
    atem = f.read()
with open(srv_path) as f:
    srv = f.read()
a_old = 'tool_call_start = "to=self<|message|>"\n'
a_new = 'tool_call_start = "<atem:function_calls>"  # OdyssAI-X #221\n'
b_old = """                        # Clean thinking tags and control tokens from remaining text
                        _, clean_remaining = _split_thinking(
                            tc.remaining_text or "",
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
"""
b_new = """                        # OdyssAI-X #221: a model with a response template
                        # (Muse: to=self / to=user turns) says itself what is
                        # content, as in the stream; the routing headers
                        # around a call are not content.
                        _odx_state = make_response_stream_state(processor)
                        if type(_odx_state).__name__ == "ResponseTemplateStreamState":
                            clean_remaining = (_odx_state.feed(
                                tc.remaining_text or "", last=True).content or "").strip()
                        else:
                            _, clean_remaining = _split_thinking(
                                tc.remaining_text or "",
                                gen_args.thinking_start_token,
                                gen_args.thinking_end_token,
                            )
"""
if mark in atem and mark in srv:
    print("[install-mlx-vlm] ATEM call marker patch already applied, skipping")
elif atem.count(a_old) != 1 or srv.count(b_old) != 1:
    print("[install-mlx-vlm] WARNING: ATEM anchors not found (upstream file changed?) — skipping")
else:
    import shutil
    shutil.copy(atem_path, atem_path + ".bak-atem-marker")
    shutil.copy(srv_path, srv_path + ".bak-atem-marker")
    atem, srv = atem.replace(a_old, a_new), srv.replace(b_old, b_new)
    compile(atem, atem_path, "exec")
    compile(srv, srv_path, "exec")
    with open(atem_path, "w") as f:
        f.write(atem)
    with open(srv_path, "w") as f:
        f.write(srv)
    print("[install-mlx-vlm] ATEM call marker patch applied")
