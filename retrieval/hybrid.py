from typing import List, Dict
from collections import defaultdict


def _min_max_normalize(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1].
    """
    if not scores:
        return scores

    min_s = min(scores)
    max_s = max(scores)

    if max_s - min_s < 1e-6:
        return [1.0 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_merge(
    sparse_results: List[Dict],
    dense_results: List[Dict],
    k: int = 20,
) -> List[Dict]:
    """
    Merge sparse and dense retrieval results.

    Args:
        sparse_results: output of SparseRetriever.search
        dense_results: output of DenseRetriever.search
        k: number of merged results to return

    Returns:
        List of merged documents sorted by combined score.
    """

    merged = defaultdict(lambda: {"score": 0.0, "text": None})

    # Normalize scores independently
    sparse_scores = _min_max_normalize(
        [r["score"] for r in sparse_results]
    )
    dense_scores = _min_max_normalize(
        [r["score"] for r in dense_results]
    )

    # Add sparse contributions
    for r, s in zip(sparse_results, sparse_scores):
        merged[r["doc_id"]]["score"] += s
        merged[r["doc_id"]]["text"] = r["text"]

    # Add dense contributions
    for r, s in zip(dense_results, dense_scores):
        merged[r["doc_id"]]["score"] += s
        merged[r["doc_id"]]["text"] = r["text"]

    # Sort by combined score
    ranked = sorted(
        merged.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    return [
        {
            "doc_id": doc_id,
            "text": data["text"],
            "score": data["score"],
        }
        for doc_id, data in ranked[:k]
    ]
