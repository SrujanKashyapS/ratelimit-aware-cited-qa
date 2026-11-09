import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 15 Engineered Evaluation Questions (10 Answerable across the 3 papers, 5 Unanswerable / Trap questions)
EVAL_QUESTIONS = [
    # Paper 1: Attention Is All You Need (1706.03762v7)
    {"id": "Q1", "q": "What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-German translation task?", "expected": "correct", "paper": "attention_1706.03762v7", "keywords": ["28.4", "bleu"]},
    {"id": "Q2", "q": "In the base Transformer model, what are the values of d_model, d_ff, and the number of attention heads h?", "expected": "correct", "paper": "attention_1706.03762v7", "keywords": ["512", "2048", "8"]},
    {"id": "Q3", "q": "Why do the authors scale the dot products by 1/sqrt(d_k) in Scaled Dot-Product Attention?", "expected": "correct", "paper": "attention_1706.03762v7", "keywords": ["large", "magnitude", "softmax", "gradients"]},
    {"id": "Q4", "q": "How many GPUs and for how many days was the big Transformer model trained?", "expected": "correct", "paper": "attention_1706.03762v7", "keywords": ["8", "p100", "3.5"]},
    # Paper 2: Moshi (2410.00037v1)
    {"id": "Q5", "q": "What is Moshi's theoretical latency and what is its practical overall latency in real-time dialogue?", "expected": "correct", "paper": "moshi_2410.00037v1", "keywords": ["160", "200"]},
    {"id": "Q6", "q": "What is the role of the Inner Monologue mechanism in Moshi and what tokens does it predict?", "expected": "correct", "paper": "moshi_2410.00037v1", "keywords": ["text", "audio", "time-aligned", "monologue"]},
    {"id": "Q7", "q": "What is Mimi in the Moshi paper, and what frame rate or bitrate does its neural audio codec operate at?", "expected": "correct", "paper": "moshi_2410.00037v1", "keywords": ["mimi", "12.5", "codec"]},
    # Paper 3: Dynamic Chunking (2507.07955v2)
    {"id": "Q8", "q": "How does Dynamic Chunking determine chunk boundaries without fixed heuristics or external tokenizers?", "expected": "correct", "paper": "dynamic_chunking_2507.07955v2", "keywords": ["routing", "boundary", "similarity", "chunk"]},
    {"id": "Q9", "q": "In Dynamic Chunking (H-Net), what happens to the sequence resolution across hierarchical stages?", "expected": "correct", "paper": "dynamic_chunking_2507.07955v2", "keywords": ["compress", "hierarchical", "stage", "byte"]},
    {"id": "Q10", "q": "How does Dynamic Chunking handle byte-level or character-level inputs compared to standard subword tokenizers like BPE?", "expected": "correct", "paper": "dynamic_chunking_2507.07955v2", "keywords": ["byte", "bpe", "tokenizer"]},
    # 5 Unanswerable / Out-of-Scope Trap Questions (Must be refused!)
    {"id": "Q11", "q": "What top-1 accuracy did the Transformer achieve on ImageNet-1K classification in Attention Is All You Need?", "expected": "refused_correctly", "paper": None, "keywords": []},
    {"id": "Q12", "q": "How many parameters does OpenAI's Whisper-large-v3 encoder use inside Moshi's speech recognition pipeline?", "expected": "refused_correctly", "paper": None, "keywords": []},
    {"id": "Q13", "q": "What were the exact AWS cloud hosting bills in USD for training the Dynamic Chunking H-Net models?", "expected": "refused_correctly", "paper": None, "keywords": []},
    {"id": "Q14", "q": "How does LoRA (Low-Rank Adaptation) fine-tuning affect Moshi's German-to-French translation BLEU score?", "expected": "refused_correctly", "paper": None, "keywords": []},
    {"id": "Q15", "q": "What learning rate schedule is recommended for training ResNet-152 with AdamW in the Attention paper?", "expected": "refused_correctly", "paper": None, "keywords": []},
]

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

class PaperLibraryRetriever:
    def __init__(self, corpus_path="data/corpus.json"):
        with open(corpus_path, "r") as f:
            self.pages = json.load(f)
        self.indices = {}
        # Pre-build the 3 required retrieval settings
        for name, cfg in self.SETTINGS.items():
            self._build_index(name, cfg["chunk_words"])

    SETTINGS = {
        "S1_150w_k2_no_rerank": {"chunk_words": 150, "top_k": 2, "rerank": False},
        "S2_250w_k4_rerank":    {"chunk_words": 250, "top_k": 4, "rerank": True},
        "S3_500w_k6_no_rerank": {"chunk_words": 500, "top_k": 6, "rerank": False},
    }

    def _build_index(self, setting_name: str, chunk_words: int):
        chunks = []
        for p in self.pages:
            words = p["text"].split()
            step = max(50, chunk_words - 30)
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i + chunk_words])
                if len(piece.strip()) < 40:
                    continue
                chunks.append({
                    "paper_id": p["paper_id"],
                    "page": p["page"],
                    "text": piece,
                    "citation": f"[{p['paper_id']}, p.{p['page']}]"
                })
        tokenized = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
        tfidf_matrix = tfidf.fit_transform([c["text"] for c in chunks])
        self.indices[setting_name] = {
            "chunks": chunks,
            "bm25": bm25,
            "tfidf": tfidf,
            "tfidf_matrix": tfidf_matrix
        }

    def retrieve(self, query: str, setting_name="S2_250w_k4_rerank", budget_tight=False):
        cfg = self.SETTINGS[setting_name]
        idx = self.indices[setting_name]
        top_k = cfg["top_k"]
        # When near rate limit, halve retrieved chunks to save input tokens
        if budget_tight:
            top_k = max(1, top_k // 2)

        q_tok = tokenize(query)
        bm25_scores = np.array(idx["bm25"].get_scores(q_tok))

        if cfg["rerank"]:
            # Fetch 3x candidates with BM25, then re-rank with TF-IDF bigram cosine similarity
            cand_Count = min(len(idx["chunks"]), top_k * 3)
            cand_indices = np.argsort(bm25_scores)[::-1][:cand_Count]
            q_vec = idx["tfidf"].transform([query])
            cos_scores = cosine_similarity(q_vec, idx["tfidf_matrix"][cand_indices]).flatten()
            # Normalize & combine
            b_sub = bm25_scores[cand_indices]
            b_norm = b_sub / (b_sub.max() + 1e-9)
            combined = 0.45 * b_norm + 0.55 * cos_scores
            order = np.argsort(combined)[::-1][:top_k]
            chosen_idx = cand_indices[order]
            best_score = float(combined[order[0]]) if len(order) > 0 else 0.0
        else:
            chosen_idx = np.argsort(bm25_scores)[::-1][:top_k]
            max_b = float(bm25_scores[chosen_idx[0]]) if len(chosen_idx) > 0 else 0.0
            best_score = min(1.0, max_b / 25.0)

        results = [idx["chunks"][i] for i in chosen_idx]
        return results, best_score