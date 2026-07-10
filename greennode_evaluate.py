"""
Evaluate: Run retrieval and reranking evaluation (không dùng MTEB)
Vietnamese Legal Text Retrieval with GreenNode Embedding + Jina Reranker

Requirements:
    pip install sentence-transformers datasets torch numpy

Cách dùng:
    python greennode_setup.py    # Load models & data trước
    python greennode_evaluate.py # Chạy evaluation
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
import os
import json

# Import từ setup file
from greennode_setup import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    OUTPUT_DIR,
    DATA_CACHE_FILE,
    load_models,
)


# ============================================================
# CONFIGURATION
# ============================================================
TOP_K_RETRIEVAL = 10   # Số docs lấy sau retrieval ban đầu
TOP_K_RERANK = 10       # Số docs sau khi rerank
RERANK_ALPHA = 0.3      # Trọng số: alpha * retrieval + (1-alpha) * reranker


# ============================================================
# PART 1: Bi-Encoder Retriever
# ============================================================
class BiEncoderRetriever:
    """Bi-encoder retriever sử dụng sentence embeddings."""

    def __init__(self, model):
        self.model = model
        self.corpus_ids = None
        self.corpus_embeddings = None

    def encode_corpus(self, corpus: dict, batch_size: int = 32):
        """Encode tất cả documents trong corpus."""
        self.corpus_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in self.corpus_ids]

        print(f"    Encoding {len(texts)} documents...")
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        print(f"    ✅ Corpus encoded! Shape: {self.corpus_embeddings.shape}")

    def encode_queries(self, queries: list, batch_size: int = 32):
        """Encode queries."""
        return self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

    def retrieve(self, query_embeddings, top_k: int = 100):
        """Retrieve top-k documents cho mỗi query."""
        similarities = self.model.similarity(query_embeddings, self.corpus_embeddings)

        results = {}
        for i in range(len(query_embeddings)):
            top_indices = torch.argsort(similarities[i], descending=True)[:top_k]
            results[i] = [
                (self.corpus_ids[idx], float(similarities[i, idx]))
                for idx in top_indices
            ]

        return results


# ============================================================
# PART 2: Cross-Encoder Reranker
# ============================================================
class CrossEncoderReranker:
    """Cross-encoder reranker."""

    def __init__(self, model):
        self.model = model

    def rerank(self, queries: list, candidates: list, alpha: float = 0.3):
        """
        Rerank candidates sử dụng cross-encoder scores.

        Args:
            queries: List of query strings
            candidates: List of list of (doc_id, doc_text, original_score)
            alpha: Weight for original scores (1-alpha for reranker)

        Returns:
            List of reranked (doc_id, combined_score) tuples
        """
        reranked_results = []

        for query, candidate_list in zip(queries, candidates):
            if len(candidate_list) == 0:
                reranked_results.append([])
                continue

            doc_ids = [doc_id for doc_id, _, _ in candidate_list]
            doc_texts = [text for _, text, _ in candidate_list]
            original_scores = [score for _, _, score in candidate_list]

            # Prepare pairs cho reranker
            pairs = [[query, doc_text] for doc_text in doc_texts]

            # Compute reranker scores
            rerank_scores = self.model.predict(pairs, show_progress_bar=False)

            # Normalize rerank scores nếu cần
            if isinstance(rerank_scores, np.ndarray) and len(rerank_scores.shape) > 1:
                rerank_scores = rerank_scores.flatten()

            # Normalize về [0, 1] nếu scores lớn hơn
            if np.max(rerank_scores) > 1:
                rerank_scores = (rerank_scores - np.min(rerank_scores)) / (
                    np.max(rerank_scores) - np.min(rerank_scores) + 1e-8
                )

            # Combine scores: alpha * retrieval + (1-alpha) * reranker
            combined = [
                (doc_id, alpha * orig_score + (1 - alpha) * rerank_score)
                for (doc_id, _, orig_score), rerank_score
                in zip(candidate_list, rerank_scores)
            ]

            # Sort
            combined.sort(key=lambda x: x[1], reverse=True)
            reranked_results.append(combined)

        return reranked_results


# ============================================================
# PART 3: Compute NDCG Metrics
# ============================================================
def compute_ndcg(results: dict, qrels: dict, k_values: list = [1, 3, 5, 10]):
    """Compute NDCG metrics at k=1,3,5,10."""

    def dcg_at_k(relevances, k):
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += rel / np.log2(i + 2)
        return dcg

    def ndcg_at_k(retrieved_relevances, ideal_relevances, k):
        dcg = dcg_at_k(retrieved_relevances, k)
        idcg = dcg_at_k(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0

    metrics = {f"NDCG@{k}": [] for k in k_values}

    for qid in qrels.keys():
        relevant_docs = set(qrels[qid].keys())
        retrieved = results.get(qid, [])
        retrieved_ids = [doc_id for doc_id, _ in retrieved]

        retrieved_relevances = [qrels[qid].get(doc_id, 0) for doc_id in retrieved_ids]
        ideal_relevances = sorted(qrels[qid].values(), reverse=True)

        for k in k_values:
            metrics[f"NDCG@{k}"].append(
                ndcg_at_k(retrieved_relevances, ideal_relevances, k)
            )

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ============================================================
# PART 4: Load Data from Cache
# ============================================================
def load_data_from_cache():
    """Load data từ cache file."""
    if not os.path.exists(DATA_CACHE_FILE):
        print(f"❌ Cache file not found: {DATA_CACHE_FILE}")
        print("   Please run greennode_setup.py first!")
        return None

    with open(DATA_CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    print(f"✅ Loaded data from cache")
    return cache["corpus"], cache["queries"], cache["qrels"]


# ============================================================
# PART 5: Run Evaluation
# ============================================================
def run_evaluation(embedding_model, reranker, corpus, queries, qrels):
    """Chạy retrieval + reranking evaluation."""
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    # Bi-encoder retrieval
    print("\n[Step 1] BI-ENCODER RETRIEVAL")
    print("-" * 40)

    retriever = BiEncoderRetriever(embedding_model)
    retriever.encode_corpus(corpus, batch_size=32)

    query_list = [queries[qid] for qid in sorted(queries.keys())]
    print("\nEncoding queries...")
    query_embeddings = retriever.encode_queries(query_list, batch_size=32)

    print("\nRetrieving documents...")
    retrieval_results = retriever.retrieve(query_embeddings, top_k=TOP_K_RETRIEVAL)

    retrieval_results_dict = {
        qid: retrieval_results[i]
        for i, qid in enumerate(sorted(queries.keys()))
    }
    print(f"    ✅ Retrieved top-{TOP_K_RETRIEVAL} for {len(queries)} queries")

    # Cross-encoder reranking
    print("\n[Step 2] CROSS-ENCODER RERANKING")
    print("-" * 40)

    reranker_model = CrossEncoderReranker(reranker)

    candidates_list = []
    for qid in sorted(queries.keys()):
        retrieved = retrieval_results_dict[qid]
        candidates = [
            (doc_id, corpus[doc_id], score)
            for doc_id, score in retrieved[:TOP_K_RERANK]
        ]
        candidates_list.append(candidates)

    print(f"Reranking top-{TOP_K_RERANK} candidates...")
    reranked_results = reranker_model.rerank(query_list, candidates_list, alpha=RERANK_ALPHA)

    reranked_results_dict = {
        qid: reranked_results[i]
        for i, qid in enumerate(sorted(queries.keys()))
    }
    print(f"    ✅ Reranked {len(queries)} queries")

    # Compute metrics
    print("\n[Step 3] COMPUTING METRICS")
    print("-" * 40)

    metrics_retrieval_only = compute_ndcg(retrieval_results_dict, qrels)
    metrics_with_reranker = compute_ndcg(reranked_results_dict, qrels)

    return {
        "retrieval_only": metrics_retrieval_only,
        "with_reranker": metrics_with_reranker,
    }


# ============================================================
# PART 6: Print Results
# ============================================================
def print_results(metrics_retrieval_only, metrics_with_reranker):
    """In kết quả evaluation."""
    print("\n" + "-" * 50)
    print("📈 RETRIEVAL ONLY (Bi-Encoder):")
    print("-" * 50)
    for metric in ["NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]:
        value = metrics_retrieval_only.get(metric, 0)
        print(f"    {metric:15s}: {value:.4f}")

    print("\n" + "-" * 50)
    print("📈 RETRIEVAL + RERANKING (Bi-Encoder + Cross-Encoder):")
    print("-" * 50)
    for metric in ["NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]:
        value = metrics_with_reranker.get(metric, 0)
        print(f"    {metric:15s}: {value:.4f}")

    # Calculate improvement
    print("\n" + "-" * 50)
    print("📈 IMPROVEMENT (with Reranker):")
    print("-" * 50)
    for metric in ["NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]:
        old = metrics_retrieval_only.get(metric, 0)
        new = metrics_with_reranker.get(metric, 0)
        if old > 0:
            imp = ((new - old) / old) * 100
            sign = "+" if imp > 0 else ""
            print(f"    {metric:15s}: {old:.4f} → {new:.4f} ({sign}{imp:.2f}%)")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "EVALUATION" + " " * 26 + "║")
    print("║" + " " * 5 + "GreenNode Embedding + Jina Reranker" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")

    # Step 1: Load data from cache
    print("\n" + "=" * 60)
    print("LOADING DATA FROM CACHE")
    print("=" * 60)

    data = load_data_from_cache()
    if not data:
        return

    corpus, queries, qrels = data

    print(f"📦 Corpus: {len(corpus)} documents")
    print(f"📦 Queries: {len(queries)} queries")

    # Step 2: Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    embedding_model, reranker, device = load_models()

    # Step 3: Run evaluation
    results = run_evaluation(embedding_model, reranker, corpus, queries, qrels)

    # Step 4: Print results
    print_results(
        results["retrieval_only"],
        results["with_reranker"]
    )

    # Step 5: Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "top_k_rerank": TOP_K_RERANK,
            "rerank_alpha": RERANK_ALPHA,
        },
        "results": results,
    }

    output_file = os.path.join(OUTPUT_DIR, "results_greennode_reranker.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
