from typing import List, Dict
from collections import defaultdict


def recall_at_k(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved_doc_ids: ranked list of retrieved document IDs
        relevant_doc_ids: ground-truth relevant document IDs
        k: cutoff

    Returns:
        Recall@K value.
    """
    if not relevant_doc_ids:
        return 0.0

    retrieved_k = set(retrieved_doc_ids[:k])
    relevant = set(relevant_doc_ids)

    hits = retrieved_k.intersection(relevant)
    return len(hits) / len(relevant)


def evaluate_retriever(
    retriever,
    queries: List[str],
    ground_truth: Dict[str, List[str]],
    k: int = 20,
) -> float:
    """
    Evaluate a retriever over a set of queries.

    Args:
        retriever: object with .search(query, k) -> List[Dict]
        queries: list of query strings
        ground_truth: mapping from query -> list of relevant doc_ids
        k: cutoff

    Returns:
        Average Recall@K over all queries.
    """
    recalls = []

    for query in queries:
        results = retriever.search(query, k=k)
        retrieved_ids = [r["doc_id"] for r in results]
        relevant_ids = ground_truth.get(query, [])

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        )
        recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0.0


def compare_retrievers(
    retrievers: Dict[str, any],
    queries: List[str],
    ground_truth: Dict[str, List[str]],
    k: int = 20,
) -> Dict[str, float]:
    """
    Compare multiple retrievers (sparse, dense, hybrid).

    Args:
        retrievers: name -> retriever instance
        queries: list of queries
        ground_truth: query -> relevant doc_ids
        k: cutoff

    Returns:
        Mapping from retriever name to Recall@K.
    """
    scores = {}

    for name, retriever in retrievers.items():
        score = evaluate_retriever(
            retriever,
            queries,
            ground_truth,
            k,
        )
        scores[name] = score

    return scores
