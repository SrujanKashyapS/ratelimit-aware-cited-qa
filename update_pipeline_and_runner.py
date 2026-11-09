with open("pipeline.py", "r") as f:
    p_code = f.read()

# 1. Reduce max_tokens on draft call from 90 to 55 so real Groq OTPM (1000/min) allows all 15 steady requests
p_code = p_code.replace("max_tokens=90 if not budget_tight else 55", "max_tokens=55 if not budget_tight else 35")
# 2. Set semaphore to 6 so burst/overload build a real waiting queue and trigger realistic client give-ups
p_code = p_code.replace("self.semaphore = asyncio.Semaphore(12)", "self.semaphore = asyncio.Semaphore(6)")

# 3. Make _grade_answer verify that the retrieved passages actually captured the ground-truth keywords for Q1-Q10
old_grade = p_code[p_code.find("    def _grade_answer"):p_code.find("    def _build_record")]
new_grade = '''    def _grade_answer(self, q_obj: dict, reply: str, passages: list) -> str:
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
        if q_obj["paper"] and q_obj["paper"] in cited_papers and kw_hits >= need_hits and re.search(r"\\[.*p\\.\\d+\\]", reply):
            return "correct"
        return "wrong"

'''
p_code = p_code[:p_code.find("    def _grade_answer")] + new_grade + p_code[p_code.find("    def _build_record"):]

with open("pipeline.py", "w") as f:
    f.write(p_code)

with open("run_experiments.py", "r") as f:
    r_code = f.read()

# Pace real steady run over 46s so 1000 OTPM refreshes cleanly
r_code = r_code.replace("duration_target_s=32.0", "duration_target_s=46.0")
# Pass realistic client_timeout_s=2.2s on burst/overload so impatient users give up when queued too long
r_code = r_code.replace(
    "return await pipeline.answer_question(client, req_id, scenario_name, q_obj, retrieval_setting=setting)",
    "t_out = 2.4 if scenario_name in ('burst', 'sustained_overload') else 10.0\n        return await pipeline.answer_question(client, req_id, scenario_name, q_obj, retrieval_setting=setting, client_timeout_s=t_out)"
)
# Compute Table 2a total_ms percentiles over completed requests (or all if none succeeded) so TTFT <= Total Time always holds
r_code = r_code.replace(
    'totals = [float(r["total_ms"]) for r in rows]',
    'ok_tots = [float(r["total_ms"]) for r in rows if r["first_token_ms"] != ""]\n        totals = ok_tots if ok_tots else [float(r["total_ms"]) for r in rows]'
)

with open("run_experiments.py", "w") as f:
    f.write(r_code)

print("Updated pipeline.py and run_experiments.py successfully.")
