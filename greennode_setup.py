"""
Setup: Load models và data (không dùng MTEB)
Vietnamese Legal Text Retrieval with GreenNode Embedding + Jina Reranker

Requirements:
    pip install sentence-transformers datasets torch numpy

Cách dùng:
    python greennode_setup.py    # Load models & data, cache lại
    python greennode_evaluate.py # Chạy evaluation
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
import torch
import os
import json


# ============================================================
# CONFIGURATION
# ============================================================
EMBEDDING_MODEL_NAME = "GreenNode/GreenNode-Embedding-Large-VN-V1"
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
DATASET_PATH = "GreenNode/zalo-ai-legal-text-retrieval-vn"
OUTPUT_DIR = "D:/DA/results"
DATA_CACHE_FILE = os.path.join(OUTPUT_DIR, "data_cache.json")


# ============================================================
# PART 1: Load Models
# ============================================================
def load_models():
    """
    Load embedding model và reranker.
    Returns: (embedding_model, reranker, device)
    """
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load embedding model
    print(f"\n[1] Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_model.to(device)
    print(f"    ✅ Loaded! Type: {type(embedding_model).__name__}")

    # Load reranker
    print(f"\n[2] Loading Reranker: {RERANKER_MODEL_NAME}")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device,trust_remote_code=True)
    print(f"    ✅ Loaded! Type: {type(reranker).__name__}")

    return embedding_model, reranker, device


# ============================================================
# PART 2: Load Dataset
# ============================================================
def load_data():
    """Load dataset từ HuggingFace."""
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    print(f"\nLoading: {DATASET_PATH}")

    ds = load_dataset(DATASET_PATH, split="test")
    print(f"    ✅ Loaded! Total samples: {len(ds)}")
    print(f"    Columns: {ds.column_names}")

    corpus = {}
    queries = {}
    qrels = {}

    for row in ds:
        doc_id = row['doc_id']
        query = row['query']
        positive_docs = row['positive_docs']

        corpus[doc_id] = row['text']
        queries[doc_id] = query
        qrels[doc_id] = {d: 1 for d in positive_docs}

    print(f"\n    - Corpus: {len(corpus)} documents")
    print(f"    - Queries: {len(queries)} queries")
    print(f"    - Qrels: {sum(len(v) for v in qrels.values())} relevant pairs")

    return corpus, queries, qrels, ds


# ============================================================
# PART 3: Save/Load Cache
# ============================================================
def save_cache(corpus, queries, qrels):
    """Lưu data cache."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cache = {
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "dataset": DATASET_PATH,
        },
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }

    with open(DATA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Data cache saved to: {DATA_CACHE_FILE}")


def load_cache():
    """Load data từ cache."""
    if not os.path.exists(DATA_CACHE_FILE):
        return None

    with open(DATA_CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    if cache["config"]["embedding_model"] != EMBEDDING_MODEL_NAME:
        print("  Cache config mismatch, reloading data...")
        return None

    print(f"\n Loaded data from cache: {DATA_CACHE_FILE}")
    return cache["corpus"], cache["queries"], cache["qrels"]


# ============================================================
# MAIN
# ============================================================
def main():
 
    # Step 1: Load models
    embedding_model, reranker, device = load_models()

    # Step 2: Load data (hoặc từ cache)
    cached = load_cache()
    if cached:
        corpus, queries, qrels = cached[0], cached[1], cached[2]
    else:
        corpus, queries, qrels, ds = load_data()
        save_cache(corpus, queries, qrels)

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\n Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f" Reranker Model: {RERANKER_MODEL_NAME}")
    print(f" Dataset: {DATASET_PATH}")
    print(f" Corpus: {len(corpus)} documents")
    print(f" Queries: {len(queries)} queries")

    print("\n" + "-" * 60)
    print(" Ready to run evaluation!")
    print("   python greennode_evaluate.py")

    return {
        "embedding_model": embedding_model,
        "reranker": reranker,
        "device": device,
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }


if __name__ == "__main__":
    main()
