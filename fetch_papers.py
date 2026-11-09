import os
import httpx
import fitz  # PyMuPDF
import json

PAPERS = {
    "attention_1706.03762v7": "https://arxiv.org/pdf/1706.03762v7.pdf",
    "moshi_2410.00037v1": "https://arxiv.org/pdf/2410.00037v1.pdf",
    "dynamic_chunking_2507.07955v2": "https://arxiv.org/pdf/2507.07955v2.pdf",
}

os.makedirs("data/pdfs", exist_ok=True)

def download_and_extract():
    corpus = []
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        for paper_id, url in PAPERS.items():
            pdf_path = f"data/pdfs/{paper_id}.pdf"
            if not os.path.exists(pdf_path):
                print(f"Downloading {paper_id}...")
                resp = client.get(url)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
            
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                corpus.append({
                    "paper_id": paper_id,
                    "page": page_num + 1,
                    "text": text
                })
    with open("data/corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Extracted {len(corpus)} total pages across 3 papers into data/corpus.json")

if __name__ == "__main__":
    download_and_extract()