import time
import json
import random
import asyncio
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Calibrated Groq Stand-In Service")

# Calibrated directly from real Groq measurements on qwen/qwen3.8-27b & gpt-oss-20b
STATE = {
    "rpm_limit": 30,
    "tpm_limit": 8000,
    "otpm_limit": 1000,
    "rpd_limit": 1000,
    "in_flight": 0,
    "max_in_flight": 30,
    "requests_this_min": [],
    "tokens_this_min": [],
    "otokens_this_min": [],
    "requests_today": 0,
    "daily_exhausted_mode": False,
    "halved_limit_mode": False,
}

def clean_sliding_window(now: float):
    cutoff = now - 60.0
    STATE["requests_this_min"] = [t for t in STATE["requests_this_min"] if t > cutoff]
    STATE["tokens_this_min"] = [(t, tok) for t, tok in STATE["tokens_this_min"] if t > cutoff]
    STATE["otokens_this_min"] = [(t, tok) for t, tok in STATE["otokens_this_min"] if t > cutoff]

@app.post("/admin/config")
async def configure_standin(request: Request):
    body = await request.json()
    if body.get("reset"):
        STATE["requests_this_min"].clear()
        STATE["tokens_this_min"].clear()
        STATE["otokens_this_min"].clear()
        STATE["requests_today"] = 0
    for k in ["daily_exhausted_mode", "halved_limit_mode", "rpm_limit", "tpm_limit", "otpm_limit", "rpd_limit"]:
        if k in body:
            STATE[k] = body[k]
    return {"status": "configured", "state": {k: v for k, v in STATE.items() if not isinstance(v, list)}}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    now = time.time()
    clean_sliding_window(now)
    body = await request.json()

    model = body.get("model", "qwen/qwen3.8-27b")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = int(body.get("max_tokens", 150))

    prompt_text = " ".join(m.get("content", "") for m in messages)
    tokens_in = max(10, len(prompt_text) // 4)

    factor = 0.5 if STATE["halved_limit_mode"] else 1.0
    eff_rpm = int(STATE["rpm_limit"] * factor)
    eff_tpm = int(STATE["tpm_limit"] * factor)
    eff_otpm = int(STATE["otpm_limit"] * factor)

    cur_rpm = len(STATE["requests_this_min"])
    cur_tpm = sum(tok for _, tok in STATE["tokens_this_min"])
    cur_otpm = sum(tok for _, tok in STATE["otokens_this_min"])

    headers = {
        "x-ratelimit-limit-requests": str(STATE["rpd_limit"]),
        "x-ratelimit-limit-tokens": str(eff_tpm),
        "x-ratelimit-remaining-requests": str(max(0, STATE["rpd_limit"] - STATE["requests_today"])),
        "x-ratelimit-remaining-tokens": str(max(0, eff_tpm - cur_tpm)),
    }

    # 1. Check Daily Budget Exhausted (3pm scenario)
    if STATE["daily_exhausted_mode"] or STATE["requests_today"] >= STATE["rpd_limit"]:
        headers["retry-after"] = "3600"
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={"error": {"message": f"Rate limit reached for model `{model}` on requests per day (RPD): Limit {STATE['rpd_limit']}, Used {STATE['rpd_limit']}.", "type": "requests", "code": "rate_limit_exceeded"}}
        )

    # 2. Check Per-Minute Limits (OTPM, TPM, RPM, or Concurrency)
    if cur_otpm + max_tokens > eff_otpm:
        headers["retry-after"] = "1"
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={"error": {"message": f"Rate limit reached for model `{model}` on output tokens per minute (OTPM): Limit {eff_otpm}, Used {cur_otpm}, Requested {max_tokens}. Please try again in 420ms.", "type": "tokens", "code": "rate_limit_exceeded"}}
        )

    if cur_tpm + tokens_in > eff_tpm or cur_rpm + 1 > eff_rpm or STATE["in_flight"] >= STATE["max_in_flight"]:
        headers["retry-after"] = "2"
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={"error": {"message": f"Rate limit reached for model `{model}` on tokens/requests per minute (TPM/RPM).", "type": "tokens", "code": "rate_limit_exceeded"}}
        )

    # Record usage
    STATE["requests_this_min"].append(now)
    STATE["tokens_this_min"].append((now, tokens_in))
    STATE["otokens_this_min"].append((now, max_tokens))
    STATE["requests_today"] += 1
    STATE["in_flight"] += 1

    try:
        # Calibrated latency: 106.4ms baseline + ~18.8ms per concurrent request + jitter
        concurrency = STATE["in_flight"]
        base_ttft_s = 0.106 + (0.0188 * concurrency) + random.uniform(-0.015, 0.035)
        # 5% chance of "slow for no visible reason" (tail latency anomaly)
        if random.random() < 0.05:
            base_ttft_s += random.uniform(1.2, 2.5)
        # 2% chance of transient 503 tool/upstream failure
        if random.random() < 0.02:
            STATE["in_flight"] -= 1
            return JSONResponse(status_code=503, content={"error": {"message": "Upstream model engine overloaded"}})

        # Generate deterministic realistic reply matching the pipeline step
        reply_text = build_deterministic_reply(prompt_text)
        words = reply_text.split(" ")
        tokens_out = max(5, int(len(words) * 1.3))

        if not stream:
            await asyncio.sleep(base_ttft_s + (0.002 * len(words)))
            STATE["in_flight"] -= 1
            return JSONResponse(
                headers=headers,
                content={
                    "id": f"chatcmpl-{int(now*1000)}",
                    "choices": [{"message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
                }
            )

        async def token_generator():
            try:
                await asyncio.sleep(max(0.05, base_ttft_s))
                chunk_size = 3
                for i in range(0, len(words), chunk_size):
                    piece = " ".join(words[i:i+chunk_size]) + (" " if i + chunk_size < len(words) else "")
                    chunk = {
                        "id": f"chatcmpl-{int(now*1000)}",
                        "choices": [{"delta": {"content": piece}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # Calibrated inter-chunk delay (~97ms total generation for 60 tokens, slows under load)
                    await asyncio.sleep(0.006 * (1 + 0.08 * concurrency))
                yield "data: [DONE]\n\n"
            finally:
                STATE["in_flight"] = max(0, STATE["in_flight"] - 1)

        return StreamingResponse(token_generator(), media_type="text/event-stream", headers=headers)
    except Exception:
        STATE["in_flight"] = max(0, STATE["in_flight"] - 1)
        raise

def build_deterministic_reply(prompt: str) -> str:
    p_low = prompt.lower()
    if "classify_intent" in p_low:
        q_part = p_low.split("question:")[-1] if "question:" in p_low else p_low
        if any(w in q_part for w in ["imagenet", "resnet", "whisper", "aws cloud", "lora"]):
            return "OUT_OF_SCOPE"
        return "IN_SCOPE"
    if "verify_answer" in p_low:
        return "VERIFIED"
    import re
    cites = re.findall(r"\[([a-z0-9_.]+,\sp\.\d+)\]", prompt)
    cite_str = f"[{cites[0]}]" if cites else "[attention_1706.03762v7, p.3]"
    return (
        f"Based on the retrieved paper passages {cite_str}, the architecture achieves the stated "
        f"experimental metrics and latency trade-offs through hierarchical sequence processing and "
        f"multi-stream attention {cite_str}."
    )
