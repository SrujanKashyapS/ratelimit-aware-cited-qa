import os, asyncio, httpx

API_KEY = os.environ["GROQ_API_KEY"].strip()
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

async def check():
    async with httpx.AsyncClient() as client:
        for m in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=HEADERS,
                json={"model": m, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}
            )
            rl = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower()}
            print(f"{m} ({r.status_code}): {rl}")

asyncio.run(check())