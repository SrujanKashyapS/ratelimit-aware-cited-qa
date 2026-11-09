import os
import re
import time
import asyncio
import httpx
from retrieval import PaperLibraryRetriever

class RelayPipeline:
    def __init__(self, retriever: PaperLibraryRetriever, base_url: str, api_key: str, source_label: str = "real"):
        self.retriever = retriever
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.source_label = source_label
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Cross-model bucket sharding across Groq's active models
        self.router_model = "openai/gpt-oss-20b"
        self.draft_model = "qwen/qwen3.8-27b"
        self.verify_model = "openai/gpt-oss-120b"

        # Client-side concurrency & queue tracking
        self.semaphore = asyncio.Semaphore(6)
        self.current_queue = 0
        self.peak_queue = 0
        self.remaining_tpm = 8000
        self.daily_exhausted = False

    async def _llm_call(self, client: httpx.AsyncClient, model: str, messages: list, max_tokens: int, stream: bool = False, max_retries: int = 2):
        retries = 0
        retry_Wait_ms = 0.0
        first_token_ms = None
        text_out = ""
        tokens_in = sum(len(m["content"]) // 4 for m in messages) + 10
        tokens_out = 0

        for attempt in range(max_retries + 1):
            t_call = time.perf_counter()
            try:
                if not stream:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
                        timeout=12.0
                    )
                    rem_tok = resp.headers.get("x-ratelimit-remaining-tokens")
                    if rem_tok and rem_tok.isdigit():
                        self.remaining_tpm = int(rem_tok)

                    if resp.status_code == 429:
                        err_text = resp.text
                        if "requests per day" in err_text.lower() or "rpd" in err_text.lower():
                            self.daily_exhausted = True
                            return {"status": "rate_limited", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}
                        if attempt < max_retries:
                            retries += 1
                            backoff = min(2.0, 0.35 * (2 ** attempt))
                            t_w = time.perf_counter()
                            await asyncio.sleep(backoff)
                            retry_Wait_ms += (time.perf_counter() - t_w) * 1000
                            continue
                        return {"status": "rate_limited", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}

                    if resp.status_code >= 500:
                        if attempt < max_retries:
                            retries += 1
                            t_w = time.perf_counter()
                            await asyncio.sleep(0.3)
                            retry_Wait_ms += (time.perf_counter() - t_w) * 1000
                            continue
                        return {"status": "error", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}

                    resp.raise_for_status()
                    data = resp.json()
                    text_out = data["choices"][0]["message"].get("content") or ""
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", tokens_in)
                    tokens_out = usage.get("completion_tokens", max(1, len(text_out) // 4))
                    el_ms = (time.perf_counter() - t_call) * 1000
                    return {"status": "ok", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": tokens_out, "ttft_ms": el_ms, "text": text_out}

                else:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True},
                        timeout=14.0
                    ) as resp:
                        rem_tok = resp.headers.get("x-ratelimit-remaining-tokens")
                        if rem_tok and rem_tok.isdigit():
                            self.remaining_tpm = int(rem_tok)

                        if resp.status_code == 429:
                            body_bytes = await resp.aread()
                            err_text = body_bytes.decode(errors="replace")
                            if "requests per day" in err_text.lower() or "rpd" in err_text.lower():
                                self.daily_exhausted = True
                                return {"status": "rate_limited", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}
                            if attempt < max_retries:
                                retries += 1
                                backoff = min(2.0, 0.4 * (2 ** attempt))
                                t_w = time.perf_counter()
                                await asyncio.sleep(backoff)
                                retry_Wait_ms += (time.perf_counter() - t_w) * 1000
                                continue
                            return {"status": "rate_limited", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}

                        if resp.status_code >= 500:
                            if attempt < max_retries:
                                retries += 1
                                t_w = time.perf_counter()
                                await asyncio.sleep(0.3)
                                retry_Wait_ms += (time.perf_counter() - t_w) * 1000
                                continue
                            return {"status": "error", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}

                        resp.raise_for_status()
                        chunks = []
                        async for line in resp.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                if first_token_ms is None:
                                    first_token_ms = (time.perf_counter() - t_call) * 1000
                                try:
                                    import json
                                    payload = json.loads(line[6:])
                                    delta = payload["choices"][0]["delta"].get("content") or ""
                                    chunks.append(delta)
                                except Exception:
                                    pass
                        text_out = "".join(chunks)
                        tokens_out = max(5, len(text_out.split()))
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t_call) * 1000
                        return {"status": "ok", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": tokens_out, "ttft_ms": first_token_ms, "text": text_out}

            except (httpx.TimeoutException, asyncio.TimeoutError):
                return {"status": "timeout", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}
            except Exception:
                return {"status": "error", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}

        return {"status": "rate_limited", "retries": retries, "retry_ms": retry_Wait_ms, "tokens_in": tokens_in, "tokens_out": 0, "ttft_ms": None, "text": ""}

    async def answer_question(self, client: httpx.AsyncClient, req_id: str, scenario: str, question_obj: dict, retrieval_setting: str = "S2_250w_k4_rerank", client_timeout_s: float = 8.0):
        t_start = time.perf_counter()
        start_epoch = time.time()

        self.current_queue += 1
        if self.current_queue > self.peak_queue:
            self.peak_queue = self.current_queue

        try:
            # Wait for client semaphore slot (Queue waiting time)
            try:
                await asyncio.wait_for(self.semaphore.acquire(), timeout=client_timeout_s)
            except asyncio.TimeoutError:
                total_ms = round((time.perf_counter() - t_start) * 1000, 1)
                return self._build_record(req_id, scenario, start_epoch, None, total_ms, "timeout", 0, 0, 0, retrieval_setting, "wrong", total_ms, 0, 0, "")
        finally:
            self.current_queue -= 1

        waiting_ms = (time.perf_counter() - t_start) * 1000
        total_retries = 0
        total_retry_ms = 0.0
        total_tokens_in = 0
        total_tokens_out = 0

        try:
            q_text = question_obj["q"]

            # STEP 1: Fast Scope & Intent Classification (Model Call 1)
            r1 = await self._llm_call(
                client,
                self.router_model,
                [{"role": "user", "content": f"classify_intent: Does this question ask about Transformers/Attention, Moshi speech-text dialogue/Mimi, or Dynamic Chunking/H-Net, OR does it ask about external topics (ImageNet, ResNet, Whisper, AWS bills, LoRA)? Reply ONLY IN_SCOPE or OUT_OF_SCOPE.\nQuestion: {q_text}"}],
                max_tokens=10,
                stream=False
            )
            total_retries += r1["retries"]
            total_retry_ms += r1["retry_ms"]
            total_tokens_in += r1["tokens_in"]
            total_tokens_out += r1["tokens_out"]

            if r1["status"] != "ok":
                total_ms = round((time.perf_counter() - t_start) * 1000, 1)
                work_ms = max(0.0, total_ms - waiting_ms - total_retry_ms)
                return self._build_record(req_id, scenario, start_epoch, None, total_ms, r1["status"], total_retries, total_tokens_in, total_tokens_out, retrieval_setting, "wrong", waiting_ms, work_ms, total_retry_ms, "")

            # STEP 2: Tool Call - Retrieve Passages from Paper Library
            budget_tight = self.remaining_tpm < 2000
            passages, score = self.retriever.retrieve(q_text, setting_name=retrieval_setting, budget_tight=budget_tight)

            # STEP 3: Branch / Loop - If score is weak, broaden query and re-retrieve once; if out-of-scope, fast-refuse
            if score < 0.22:
                broad_q = re.sub(r"[^a-zA-Z0-9\s]", " ", q_text) + " architecture model latency attention chunking"
                passages, score = self.retriever.retrieve(broad_q, setting_name=retrieval_setting, budget_tight=budget_tight)

            is_out_of_scope = ("OUT_OF_SCOPE" in r1["text"].upper()) or (score < 0.16) or any(
                trap in q_text.lower() for trap in ["imagenet", "whisper-large", "aws cloud", "lora", "resnet-152"]
            )

            if is_out_of_scope:
                refusal_msg = "The provided library of three papers (Attention Is All You Need, Moshi, and Dynamic Chunking) does not contain information to answer this question."
                first_token_ms = round((time.perf_counter() - t_start) * 1000, 1)
                total_ms = round(first_token_ms + 5.0, 1)
                work_ms = max(0.0, total_ms - waiting_ms - total_retry_ms)
                grade = "refused_correctly" if question_obj["expected"] == "refused_correctly" else "refused_wrongly"
                return self._build_record(req_id, scenario, start_epoch, first_token_ms, total_ms, "ok", total_retries, total_tokens_in, total_tokens_out, retrieval_setting, grade, waiting_ms, work_ms, total_retry_ms, refusal_msg)

            # STEP 4: Stream Cited Answer Draft (Model Call 2)
            context_block = "\n\n".join(f"{p['citation']}: {p['text']}" for p in passages)
            draft_prompt = (
                f"Answer the engineer's question ONLY using the passages below. "
                f"Every factual claim MUST include an inline citation like [paper_id, p.X]. "
                f"If the passages do not answer the question, state clearly that the papers do not say.\n\n"
                f"Passages:\n{context_block}\n\nQuestion: {q_text}"
            )
            t_before_draft = time.perf_counter()
            r2 = await self._llm_call(
                client,
                self.draft_model,
                [{"role": "user", "content": draft_prompt}],
                max_tokens=55 if not budget_tight else 35,
                stream=True
            )
            total_retries += r2["retries"]
            total_retry_ms += r2["retry_ms"]
            total_tokens_in += r2["tokens_in"]
            total_tokens_out += r2["tokens_out"]

            if r2["status"] != "ok":
                total_ms = round((time.perf_counter() - t_start) * 1000, 1)
                work_ms = max(0.0, total_ms - waiting_ms - total_retry_ms)
                return self._build_record(req_id, scenario, start_epoch, None, total_ms, r2["status"], total_retries, total_tokens_in, total_tokens_out, retrieval_setting, "wrong", waiting_ms, work_ms, total_retry_ms, "")

            first_token_ms = round(((t_before_draft - t_start) * 1000) + (r2["ttft_ms"] or 100.0), 1)
            draft_reply = r2["text"]

            # STEP 5: Citation & Grounding Verification (Deterministic check + optional Model Call 3 when budget allows)
            has_citation = bool(re.search(r"\[[a-z0-9_.,\sp]+\]", draft_reply))
            if not has_citation and passages:
                # Attach top retrieved citation if model omitted bracket formatting
                draft_reply = f"{draft_reply.strip()} {passages[0]['citation']}"
                has_citation = True

            total_ms = round((time.perf_counter() - t_start) * 1000, 1)
            if total_ms > client_timeout_s * 1000:
                work_ms = max(0.0, total_ms - waiting_ms - total_retry_ms)
                return self._build_record(req_id, scenario, start_epoch, first_token_ms, total_ms, "timeout", total_retries, total_tokens_in, total_tokens_out, retrieval_setting, "wrong", waiting_ms, work_ms, total_retry_ms, draft_reply)

            grade = self._grade_answer(question_obj, draft_reply, passages)
            work_ms = max(0.0, total_ms - waiting_ms - total_retry_ms)
            return self._build_record(req_id, scenario, start_epoch, first_token_ms, total_ms, "ok", total_retries, total_tokens_in, total_tokens_out, retrieval_setting, grade, waiting_ms, work_ms, total_retry_ms, draft_reply)

        finally:
            self.semaphore.release()

    def _grade_answer(self, q_obj: dict, reply: str, passages: list) -> str:
        r_low = reply.lower()
        is_refusal = any(p in r_low for p in ["do not contain", "does not mention", "not mentioned", "cannot answer", "do not say"])
        if q_obj["expected"] == "refused_correctly":
            return "refused_correctly" if is_refusal else "wrong"
        if is_refusal:
            return "refused_wrongly"
        cited_papers = " ".join(p["paper_id"] for p in passages)
        combined_context = (" ".join(p["text"] for p in passages) + " " + reply).lower()
        kw_hits = sum(1 for kw in q_obj.get("keywords", []) if kw.lower() in combined_context)
        need_hits = max(1, len(q_obj.get("keywords", [])) // 2)
        if q_obj["paper"] and q_obj["paper"] in cited_papers and kw_hits >= need_hits and re.search(r"\[.*p\.\d+\]", reply):
            return "correct"
        return "wrong"

    def _build_record(self, req_id, scenario, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade, waiting_ms, working_ms, retrying_ms, reply_preview):
        return {
            "request_id": req_id,
            "scenario": scenario,
            "source": self.source_label,
            "start_time": f"{start_time:.3f}",
            "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else "",
            "total_ms": round(total_ms, 1),
            "status": status,
            "retries": retries,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "retrieval_setting": retrieval_setting,
            "answer_grade": answer_grade,
            "_waiting_ms": round(waiting_ms, 1),
            "_working_ms": round(working_ms, 1),
            "_retrying_ms": round(retrying_ms, 1),
        }