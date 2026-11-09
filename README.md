# Rate-Limit-Aware Cited QA

A compact research and systems project for evaluating a citation-grounded question-answering pipeline under realistic API rate limits. The system operates over a three-paper corpus and is designed to answer factual questions with inline citations while detecting out-of-scope prompts and degrading gracefully under quota exhaustion.

## Why this project exists

Large language models often fail in production when they are asked to answer under:

- strict RPM / TPM / OTPM limits,
- daily request budgets,
- burst traffic patterns,
- and retrieval-heavy tasks that require citation-aware outputs.

This repository implements a lightweight benchmark and pipeline that models those constraints and measures how the system behaves under each condition.

## Project summary

The core design is a 5-stage relay pipeline:

1. Intent classification
2. Retrieval of relevant paper passages
3. Scope check and refusal logic
4. Draft answer generation with citations
5. Verification / grading of the final answer

The project evaluates both:

- normal operational behavior against real Groq models, and
- stress behavior against a calibrated local stand-in service that mimics rate limits.

## Benchmark design

### Corpus
The library contains 3 primary papers:

- Attention Is All You Need
- Moshi
- Dynamic Chunking

These are stored in `data/corpus.json` and used as the knowledge base for retrieval and QA.

### Evaluation set
The benchmark includes:

- 15 total evaluation questions
- 10 answerable questions
- 5 deliberately out-of-scope / trap questions

The answer set is defined in `retrieval.py` via `EVAL_QUESTIONS`.

### Retrieval settings
The retriever implements 3 settings:

- `S1_150w_k2_no_rerank`
  - chunk size: 150 words
  - top-k: 2
  - no reranking

- `S2_250w_k4_rerank`
  - chunk size: 250 words
  - top-k: 4
  - BM25 + TF-IDF reranking

- `S3_500w_k6_no_rerank`
  - chunk size: 500 words
  - top-k: 6
  - no reranking

### Load scenarios
The experiment runner exercises 5 major scenarios:

- `steady`: 15 real requests over ~46 seconds
- `burst`: 45 requests in 3 seconds
- `sustained_overload`: 60 requests over 20 seconds
- `limit_halved`: 30 requests with limits halved to simulate constrained budgets
- `daily_exhausted`: 20 requests with a daily quota exhausted state

## System architecture

```text
User Question
    |
    v
Intent Router
    |
    +--> OUT_OF_SCOPE ? --> Refusal
    |
    v
Retriever (BM25 + rerank)
    |
    v
Draft Answer Generator (LLM)
    |
    v
Citation / Fact Verification
    |
    v
Final Answer + Citation Output
```

## Main components

### `retrieval.py`
Defines:

- the evaluation question bank,
- the corpus-based retrieval logic,
- BM25 indexing,
- TF-IDF reranking,
- and retrieval settings for different quality / latency trade-offs.

### `pipeline.py`
Contains the core `RelayPipeline` class that implements:

- request queue tracking,
- semaphore-based concurrency control,
- step-by-step LLM calls,
- rate-limit retry handling,
- answer grading,
- and output record generation for experiment CSVs.

### `standin_server.py`
A FastAPI service that simulates Groq-like rate limits and model behavior. It enforces:

- `RPM = 30`
- `TPM = 8000`
- `OTPM = 1000`
- `RPD = 1000`

and supports toggles for:

- daily quota exhaustion,
- halved limit mode,
- and burst overload behavior.

### `run_experiments.py`
Runs the benchmark against:

- the real Groq API,
- and the local calibrated stand-in service.

It exports results to `results.csv` and prints summary tables for latency, outcomes, retrieval comparison, and slow-request diagnostics.

## Runtime workflow

Each `answer_question(...)` call follows this execution path:

1. Acquire a semaphore slot to model queue waiting time.
2. Classify the prompt as in scope or out of scope.
3. Retrieve passages from the paper library.
4. If the score is low or the prompt is off-topic, trigger a broader retrieval or a refusal.
5. Generate a draft answer using the retrieved evidence.
6. Check for citations and grade the response.
7. Record timing, retries, token usage, queue latency, and outcome.

## Key algorithmic behavior

### Queue-aware request relay
The pipeline tracks:

- current in-flight queue length,
- peak queue depth,
- waiting time before acquisition,
- retry delay time,
- and working time spent in retrieval + LLM generation.

This matters because a real rate-limited service is not just a `429` issue; it is a queueing and latency problem as well.

### Rate-limit handling
`_llm_call()` handles:

- HTTP 429 responses,
- daily limit exhaustion detection,
- transient server errors,
- timeout retries,
- and stream-mode responses.

### Citation groundedness
The answer is graded as correct when:

- the expected paper is cited,
- key keywords appear in the final answer or context,
- and the answer contains paper-page-style citation syntax such as `[paper_id, p.3]`.

The refusal logic is also deliberately strict so that trap questions are rejected correctly rather than hallucinated.

## Setup

### Prerequisites

- Python 3.10+
- pip
- a Groq API key

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn httpx numpy rank-bm25 scikit-learn
```

### Environment variables

```bash
export GROQ_API_KEY="your_key_here"
```

## Run the project

### 1. Check model availability

```bash
python check_models.py
```

### 2. Probe Groq directly

```bash
python probe_groq.py
```

### 3. Run the full benchmark

```bash
python run_experiments.py
```

This launches the local stand-in server, runs the benchmark scenarios, and saves results to `results.csv`.

## Expected outputs

The experiment runner produces:

- `results.csv` with per-request metrics,
- latency summaries,
- outcome breakdowns,
- retrieval-setting comparison,
- and slow-request trace information.

The CSV schema includes:

- `request_id`
- `scenario`
- `source`
- `start_time`
- `first_token_ms`
- `total_ms`
- `status`
- `retries`
- `tokens_in`
- `tokens_out`
- `retrieval_setting`
- `answer_grade`

## Repository structure

```text
.
├── README.md
├── check_models.py
├── fetch_papers.py
├── pipeline.py
├── probe_groq.py
├── retrieval.py
├── run_experiments.py
├── standin_server.py
├── update_all.py
├── update_pipeline_and_runner.py
├── data/
│   └── corpus.json
└── results.csv
```

## Design strengths

This project is useful for interviews and technical discussions because it combines multiple systems concerns:

- LLM orchestration
- retrieval quality vs latency trade-offs
- queueing under rate limits
- API quota modeling
- citation-aware answer validation
- failure-mode testing for production-grade AI systems

It demonstrates practical engineering thinking, not just prompt usage.

## Limitations

This is a research-style, compact benchmark rather than a production-grade distributed service. It intentionally focuses on:

- controlled evaluation,
- reproducible scenarios,
- and clear observability of the model/tool bottlenecks.

For production use, one would extend this with:

- persistent storage,
- proper logging and tracing,
- dedicated model routing and fallback policies,
- queue workers,
- and deployment monitoring.

## License

This project is provided as a research / experimentation repository. Use it as a base for learning, benchmarking, and extension.

## Contact / project status

This repository was built to investigate the interaction between retrieval, answer citation, and API rate-limit constraints in an LLM-driven QA system.
