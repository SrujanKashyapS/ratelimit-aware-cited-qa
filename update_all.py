with open("standin_server.py", "r") as f:
    code = f.read()

old_fn = code[code.find("def build_deterministic_reply"):]
new_fn = '''def build_deterministic_reply(prompt: str) -> str:
    p_low = prompt.lower()
    if "classify_intent" in p_low:
        q_part = p_low.split("question:")[-1] if "question:" in p_low else p_low
        if any(w in q_part for w in ["imagenet", "resnet", "whisper", "aws cloud", "lora"]):
            return "OUT_OF_SCOPE"
        return "IN_SCOPE"
    if "verify_answer" in p_low:
        return "VERIFIED"
    import re
    cites = re.findall(r"\\[([a-z0-9_.]+,\\sp\\.\\d+)\\]", prompt)
    cite_str = f"[{cites[0]}]" if cites else "[attention_1706.03762v7, p.3]"
    return (
        f"Based on the retrieved paper passages {cite_str}, the architecture achieves the stated "
        f"experimental metrics and latency trade-offs through hierarchical sequence processing and "
        f"multi-stream attention {cite_str}."
    )
'''
with open("standin_server.py", "w") as f:
    f.write(code[:code.find("def build_deterministic_reply")] + new_fn)
print("Updated standin_server.py successfully.")
