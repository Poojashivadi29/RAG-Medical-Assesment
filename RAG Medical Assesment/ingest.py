"""
Ingest script for clinic FAQ knowledge base.
- Reads `clinic_faqs.csv`
- Chunks document text, computes embeddings, and stores vectors+metadata.

Behavior:
- If environment variable OPENAI_API_KEY is set, uses OpenAI embeddings and stores vectors in a Chroma-compatible pickle (`meta.pkl` with embeddings).
- Otherwise falls back to sentence-transformers + FAISS (local).

Usage:
    python ingest.py --csv clinic_faqs.csv --index-path faiss.index --meta-path meta.pkl

The script writes a `meta.pkl` file containing texts, metadatas, ids, embeddings and model_name.
"""
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd


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


def build_index_openai(csv_path, meta_path, model_name="text-embedding-3-small"):
    """Use OpenAI embeddings and store metadata+embeddings to meta_path (pickle)."""
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package is required for OpenAI embeddings. Install with pip install openai")

    # Read CSV and chunk
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

    print(f"Computing OpenAI embeddings for {len(texts)} chunks using model {model_name}...")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # batch request
    embeddings = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = openai.Embedding.create(model=model_name, input=batch)
        for d in resp["data"]:
            embeddings.append(d["embedding"])  # list of floats

    embeddings = np.array(embeddings, dtype=np.float32)

    # Save embeddings + metadata
    os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
    with open(meta_path, "wb") as f:
        pickle.dump({"metadatas": metadatas, "texts": texts, "ids": ids, "embeddings": embeddings, "model_name": model_name}, f)

    print(f"Saved metadata+embeddings to {meta_path} (OpenAI+Chroma compatible).")


def build_index_faiss(csv_path, index_path, meta_path, model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except Exception:
        raise RuntimeError("sentence-transformers and faiss are required for the local faiss pipeline. Install with pip install sentence-transformers faiss-cpu")

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
        pickle.dump({"metadatas": metadatas, "texts": texts, "ids": ids, "embeddings": embeddings, "model_name": model_name}, f)

    print(f"Saved FAISS index to {index_path} and metadata to {meta_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="clinic_faqs.csv")
    parser.add_argument("--index-path", default="faiss.index")
    parser.add_argument("--meta-path", default="meta.pkl")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # Decide pipeline based on environment
    if os.getenv("OPENAI_API_KEY"):
        model = args.model or "text-embedding-3-small"
        build_index_openai(args.csv, args.meta_path, model_name=model)
    else:
        model = args.model or "all-MiniLM-L6-v2"
        build_index_faiss(args.csv, args.index_path, args.meta_path, model_name=model)
