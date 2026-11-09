import os
import sys
import time
import asyncio
import httpx

API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not API_KEY or API_KEY == "your_groq_key_here":
    print("ERROR: GROQ_API_KEY is missing.")
    sys.exit(1)

BASE_URL = "https://api.groq.com/openai/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

async def pick_active_model(client):
    resp = await client.get(f"{BASE_URL}/models", headers=HEADERS, timeout=10.0)
    resp.raise_for_status()
    models = [m["id"] for m in resp.json().get("data", [])]
    print("Available Groq Models:", models)
    
    # Exclude audio/whisper/guard/embedding models
    chat_models = [
        m for m in models
        if not any(x in m.lower() for x in ["whisper", "guard", "tts", "embed", "vision", "playai"])
    ]
    # Prefer fast 8b/small llama or qwen/gemma models for high RPM/TPM probing
    for pref in ["8b", "llama", "qwen", "gemma", "mixtral", "deepseek"]:
        for m in chat_models:
            if pref in m.lower():
                return m
    return chat_models[0] if chat_models else models[0]

async def single_stream_probe(client, model, req_id):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain scaled dot-product attention in two sentences."}],
        "stream": True,
        "max_tokens": 60
    }
    start = time.perf_counter()
    first_token_ms = None
    rl_headers = {}

    try:
        async with client.stream("POST", f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=15.0) as resp:
            for k, v in resp.headers.items():
                if "ratelimit" in k.lower() or "retry-after" in k.lower():
                    rl_headers[k] = v

            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                return {
                    "id": req_id,
                    "status": "rate_limited" if resp.status_code == 429 else f"http_{resp.status_code}",
                    "code": resp.status_code,
                    "total_ms": round((time.perf_counter() - start) * 1000, 1),
                    "headers": rl_headers,
                    "error_body": body
                }

            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - start) * 1000, 1)

        total_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "id": req_id, "status": "ok", "code": 200,
            "first_token_ms": first_token_ms, "total_ms": total_ms, "headers": rl_headers
        }
    except Exception as e:
        return {
            "id": req_id, "status": f"exception_{type(e).__name__}",
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
            "headers": rl_headers, "error_body": str(e)
        }

async def main():
    async with httpx.AsyncClient() as client:
        model = await pick_active_model(client)
        print(f"\nSelected Model: {model}")

        print("\n--- 1. Single Request Header & Baseline Check ---")
        r0 = await single_stream_probe(client, model, 0)
        print("Status:", r0["status"], "| TTFT (ms):", r0.get("first_token_ms"), "| Total (ms):", r0["total_ms"])
        print("Rate Limit Headers:", r0["headers"])
        if r0["status"] != "ok":
            print("Error Details:", r0.get("error_body"))
            return

        print("\n--- 2. Burst Probe (35 rapid requests to trigger RPM 429 & measure recovery) ---")
        tasks = [single_stream_probe(client, model, i) for i in range(1, 36)]
        results = await asyncio.gather(*tasks)

        ok_runs = [r for r in results if r["status"] == "ok"]
        rl_runs = [r for r in results if r["status"] == "rate_limited"]
        other_runs = [r for r in results if r["status"] not in ("ok", "rate_limited")]
        print(f"Succeeded: {len(ok_runs)} | Rate-Limited (429): {len(rl_runs)} | Other: {len(other_runs)}")

        if ok_runs:
            ttfts = sorted([r["first_token_ms"] for r in ok_runs if r.get("first_token_ms")])
            totals = sorted([r["total_ms"] for r in ok_runs])
            print(f"OK TTFT p50: {ttfts[len(ttfts)//2]} ms | p95: {ttfts[int(len(ttfts)*0.95)]} ms | max: {ttfts[-1]} ms")
            print(f"OK Total p50: {totals[len(totals)//2]} ms | p95: {totals[int(len(totals)*0.95)]} ms | max: {totals[-1]} ms")

        if rl_runs:
            print("Sample 429 Headers:", rl_runs[0]["headers"])
            print("Sample 429 Body:", rl_runs[0]["error_body"][:350])

            t_429 = time.perf_counter()
            print("\nPolling every 0.5s to measure recovery time after 429...")
            while True:
                await asyncio.sleep(0.5)
                rec = await single_stream_probe(client, model, 999)
                if rec["status"] == "ok":
                    rec_ms = round((time.perf_counter() - t_429) * 1000, 1)
                    print(f"Recovered in {rec_ms} ms ({rec_ms/1000:.2f}s)!")
                    break

if __name__ == "__main__":
    asyncio.run(main())