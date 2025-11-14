"""
Ingest script for clinic FAQ knowledge base.
- Reads `data/clinic_faqs.csv` (or local copy)
- Chunks document text, computes embeddings, and stores a FAISS index + metadata
- Output: `faiss_index.pkl` (index saved via faiss.write_index) and `meta.pkl` (list of dicts)

Usage:
    python ingest.py --csv clinic_faqs.csv --index-path faiss.index --meta-path meta.pkl

This script uses sentence-transformers for embeddings.
"""
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    print("Important: required packages missing. Install with: pip install -r requirements.txt")
    raise


def chunk_text(text, chunk_size=400, overlap=50):
    """Simple character-based chunker."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks


def build_index(csv_path, index_path, meta_path, model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(csv_path)
    texts = []
    metadatas = []
    ids = []
    for _, row in df.iterrows():
        doc_id = row["id"]
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        full = f"{title}\n\n{content}"
        chunks = chunk_text(full)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source_id": int(doc_id), "title": title, "chunk_index": i})
            ids.append(f"{doc_id}_{i}")

    print(f"Computing embeddings for {len(texts)} chunks using model {model_name}...")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    # Save index and metadata
    os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"metadatas": metadatas, "texts": texts, "ids": ids, "model_name": model_name}, f)

    print(f"Saved FAISS index to {index_path} and metadata to {meta_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="clinic_faqs.csv")
    parser.add_argument("--index-path", default="faiss.index")
    parser.add_argument("--meta-path", default="meta.pkl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    build_index(args.csv, args.index_path, args.meta_path, model_name=args.model)
