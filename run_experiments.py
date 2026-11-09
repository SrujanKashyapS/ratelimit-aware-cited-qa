import os
import csv
import time
import asyncio
import subprocess
import numpy as np
import httpx
from retrieval import PaperLibraryRetriever, EVAL_QUESTIONS
from pipeline import RelayPipeline

GROQ_BASE = "https://api.groq.com/openai/v1"
STANDIN_BASE = "http://127.0.0.1:8009/v1"
STANDIN_ADMIN = "http://127.0.0.1:8009/admin/config"
API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

CSV_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade"
]

def pct(arr, q):
    if not arr:
        return 0.0
    return round(float(np.percentile(arr, q)), 1)

async def run_scenario_batch(pipeline: RelayPipeline, client: httpx.AsyncClient, scenario_name: str, count: int, duration_target_s: float, setting: str = "S2_250w_k4_rerank"):
    interval = duration_target_s / max(1, count)
    tasks = []
    t0 = time.perf_counter()

    async def launch_one(idx):
        await asyncio.sleep(idx * interval)
        q_obj = EVAL_QUESTIONS[idx % len(EVAL_QUESTIONS)]
        req_id = f"{scenario_name}_{idx+1:03d}"
        t_out = 2.4 if scenario_name in ('burst', 'sustained_overload') else 10.0
        return await pipeline.answer_question(client, req_id, scenario_name, q_obj, retrieval_setting=setting, client_timeout_s=t_out)

    for i in range(count):
        tasks.append(asyncio.create_task(launch_one(i)))
    results = await asyncio.gather(*tasks)
    elapsed_s = round(time.perf_counter() - t0, 1)
    return results, elapsed_s, pipeline.peak_queue

async def main():
    print("1. Initializing 3-Paper Library Retriever...")
    retriever = PaperLibraryRetriever("data/corpus.json")

    print("2. Launching local Calibrated Stand-In Server on port 8009...")
    proc = subprocess.Popen(
        ["uvicorn", "standin_server:app", "--host", "127.0.0.1", "--port", "8009", "--log-level", "warning"]
    )
    await asyncio.sleep(2.0)

    all_rows = []
    scenario_meta = {}

    try:
        async with httpx.AsyncClient() as client:
            # --- PART A: REAL GROQ RUN (Steady scenario + S2 retrieval on all 15 questions) ---
            print("\n3. Running REAL Service Scenario: 'steady' (15 questions on Groq free tier)...")
            real_pipe = RelayPipeline(retriever, GROQ_BASE, API_KEY, source_label="real")
            # Paced over ~32s so we respect the 1,000 OTPM limit on qwen/qwen3.8-27b
            r_steady, dur_steady, pk_steady = await run_scenario_batch(
                real_pipe, client, "steady", count=15, duration_target_s=46.0, setting="S2_250w_k4_rerank"
            )
            all_rows.extend(r_steady)
            scenario_meta["steady"] = {"dur": dur_steady, "peak_q": pk_steady, "offered_rpm": round(15 / (dur_steady / 60.0), 1)}
            print(f"   Done 'steady' (real): {sum(1 for r in r_steady if r['status']=='ok')}/15 succeeded in {dur_steady}s.")

            # --- PART B: RETRIEVAL SETTINGS COMPARISON (S1 & S3 on 15 questions each) ---
            for set_name in ["S1_150w_k2_no_rerank", "S3_500w_k6_no_rerank"]:
                print(f"\n4. Evaluating Retrieval Setting: {set_name} (15 questions)...")
                await client.post(STANDIN_ADMIN, json={"reset": True, "daily_exhausted_mode": False, "halved_limit_mode": False})
                ret_pipe = RelayPipeline(retriever, STANDIN_BASE, "standin-key", source_label="stand-in")
                r_ret, dur_ret, _ = await run_scenario_batch(
                    ret_pipe, client, f"eval_{set_name}", count=15, duration_target_s=10.0, setting=set_name
                )
                all_rows.extend(r_ret)

            # --- PART C: LOAD SCENARIOS ON CALIBRATED STAND-IN ---
            # Scenario 2: Sudden Burst (45 requests in 3 seconds -> 900 RPM offered rate)
            print("\n5. Running Load Scenario: 'burst' (45 requests in 3s)...")
            await client.post(STANDIN_ADMIN, json={"reset": True, "daily_exhausted_mode": False, "halved_limit_mode": False})
            burst_pipe = RelayPipeline(retriever, STANDIN_BASE, "standin-key", source_label="stand-in")
            r_burst, dur_burst, pk_burst = await run_scenario_batch(
                burst_pipe, client, "burst", count=45, duration_target_s=3.0
            )
            all_rows.extend(r_burst)
            scenario_meta["burst"] = {"dur": dur_burst, "peak_q": pk_burst, "offered_rpm": 900.0}

            # Scenario 3: Sustained Overload (60 requests over 20 seconds -> 180 RPM offered rate)
            print("6. Running Load Scenario: 'sustained_overload' (60 requests over 20s)...")
            await client.post(STANDIN_ADMIN, json={"reset": True, "daily_exhausted_mode": False, "halved_limit_mode": False})
            over_pipe = RelayPipeline(retriever, STANDIN_BASE, "standin-key", source_label="stand-in")
            r_over, dur_over, pk_over = await run_scenario_batch(
                over_pipe, client, "sustained_overload", count=60, duration_target_s=20.0
            )
            all_rows.extend(r_over)
            scenario_meta["sustained_overload"] = {"dur": dur_over, "peak_q": pk_over, "offered_rpm": 180.0}

            # Scenario 4: Per-Minute Limit Halved (30 requests over 15 seconds with limits cut 50%)
            print("7. Running Load Scenario: 'limit_halved' (30 requests with RPM/TPM/OTPM halved)...")
            await client.post(STANDIN_ADMIN, json={"reset": True, "daily_exhausted_mode": False, "halved_limit_mode": True})
            half_pipe = RelayPipeline(retriever, STANDIN_BASE, "standin-key", source_label="stand-in")
            r_half, dur_half, pk_half = await run_scenario_batch(
                half_pipe, client, "limit_halved", count=30, duration_target_s=15.0
            )
            all_rows.extend(r_half)
            scenario_meta["limit_halved"] = {"dur": dur_half, "peak_q": pk_half, "offered_rpm": 120.0}

            # Scenario 5: Daily Budget Exhausted (20 requests when 3pm RPD limit is hit)
            print("8. Running Load Scenario: 'daily_exhausted' (20 requests at 0 remaining RPD)...")
            await client.post(STANDIN_ADMIN, json={"reset": True, "daily_exhausted_mode": True, "halved_limit_mode": False})
            exh_pipe = RelayPipeline(retriever, STANDIN_BASE, "standin-key", source_label="stand-in")
            r_exh, dur_exh, pk_exh = await run_scenario_batch(
                exh_pipe, client, "daily_exhausted", count=20, duration_target_s=4.0
            )
            all_rows.extend(r_exh)
            scenario_meta["daily_exhausted"] = {"dur": dur_exh, "peak_q": pk_exh, "offered_rpm": 300.0}

    finally:
        proc.terminate()
        proc.wait()

    # Save exact required results.csv
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results.csv!")

    # Print Computed Tables 2a, 2b, 3, and Section 4 Slow Request Trace
    print("\n================ TABLE 2a: LATENCY ================")
    print("Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50/p95/p99 (ms) | Total time p50/p95/p99 (ms)")
    for sc in ["steady", "burst", "sustained_overload", "limit_halved", "daily_exhausted"]:
        rows = [r for r in all_rows if r["scenario"] == sc]
        ttfts = [float(r["first_token_ms"]) for r in rows if r["first_token_ms"] != ""]
        ok_tots = [float(r["total_ms"]) for r in rows if r["first_token_ms"] != ""]
        totals = ok_tots if ok_tots else [float(r["total_ms"]) for r in rows]
        src = rows[0]["source"]
        m = scenario_meta[sc]
        ttft_str = f"{pct(ttfts,50)} / {pct(ttfts,95)} / {pct(ttfts,99)}" if ttfts else "not measured"
        tot_str = f"{pct(totals,50)} / {pct(totals,95)} / {pct(totals,99)}"
        print(f"{sc} | {src} | {len(rows)} | {m['dur']} | {m['offered_rpm']} | {ttft_str} | {tot_str}")

    print("\n================ TABLE 2b: OUTCOMES ================")
    print("Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers")
    for sc in ["steady", "burst", "sustained_overload", "limit_halved", "daily_exhausted"]:
        rows = [r for r in all_rows if r["scenario"] == sc]
        n = len(rows)
        ok_c = sum(1 for r in rows if r["status"] == "ok")
        rl_c = sum(1 for r in rows if r["status"] == "rate_limited")
        to_c = sum(1 for r in rows if r["status"] in ("timeout", "error"))
        avg_ret = round(sum(r["retries"] for r in rows) / n, 2)
        ok_rows = [r for r in rows if r["status"] == "ok"]
        tok_per_ans = round(sum(r["tokens_in"] + r["tokens_out"] for r in ok_rows) / len(ok_rows), 1) if ok_rows else 0.0
        # Estimated at $0.10 per 1M tokens ($0.0001 per 1k tokens) -> per 1,000 answers
        cost_1k = f"${round(tok_per_ans * 0.0001 * 1000, 2):.2f}"
        print(f"{sc} | {round(ok_c*100/n,1)}% | {round(rl_c*100/n,1)}% | {round(to_c*100/n,1)}% | {avg_ret} | {scenario_meta[sc]['peak_q']} | {tok_per_ans} | {cost_1k}")

    print("\n================ TABLE 3: RETRIEVAL SETTINGS ================")
    print("Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50/p95 (ms) | Total time p50/p95 (ms)")
    for sc_label, set_id in [("steady", "S2_250w_k4_rerank"), ("eval_S1_150w_k2_no_rerank", "S1_150w_k2_no_rerank"), ("eval_S3_500w_k6_no_rerank", "S3_500w_k6_no_rerank")]:
        rows = [r for r in all_rows if r["scenario"] == sc_label]
        n = len(rows)
        corr = sum(1 for r in rows if r["answer_grade"] == "correct")
        ref_ok = sum(1 for r in rows if r["answer_grade"] == "refused_correctly")
        wrong = sum(1 for r in rows if r["answer_grade"] in ("wrong", "refused_wrongly"))
        ok_rows = [r for r in rows if r["status"] == "ok"]
        tok_ans = round(sum(r["tokens_in"] + r["tokens_out"] for r in ok_rows) / max(1, len(ok_rows)), 1)
        ttfts = [float(r["first_token_ms"]) for r in ok_rows if r["first_token_ms"] != ""]
        totals = [float(r["total_ms"]) for r in ok_rows]
        print(f"{set_id} | {n} | {round(corr*100/n,1)}% | {round(wrong*100/n,1)}% | {round(ref_ok*100/n,1)}% | {tok_ans} | {pct(ttfts,50)} / {pct(ttfts,95)} | {pct(totals,50)} / {pct(totals,95)}")

    # Section 4: Slow request breakdown
    slow_candidates = [r for r in all_rows if r["status"] == "ok" and r["retries"] > 0]
    if not slow_candidates:
        slow_candidates = sorted([r for r in all_rows if r["status"] == "ok"], key=lambda x: float(x["total_ms"]), reverse=True)
    if slow_candidates:
        s = sorted(slow_candidates, key=lambda x: float(x["total_ms"]), reverse=True)[0]
        print("\n================ SECTION 4: SLOW REQUEST BREAKDOWN ================")
        print(f"Request ID: {s['request_id']} | Scenario: {s['scenario']} | Source: {s['source']}")
        print(f"First Token: {s['first_token_ms']} ms | Total Time: {s['total_ms']} ms | Retries: {s['retries']}")
        print(f"Split -> Waiting in queue: {s['_waiting_ms']} ms | Working (retrieval + LLM): {s['_working_ms']} ms | Retrying (backoff): {s['_retrying_ms']} ms")

if __name__ == "__main__":
    asyncio.run(main())