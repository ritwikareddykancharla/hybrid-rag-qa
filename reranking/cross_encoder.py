from typing import List, Dict
import torch


class CrossEncoderReranker:
    """
    Cross-encoder reranker for query-document relevance scoring.

    Uses a pretrained Transformer that jointly encodes
    (query, document) pairs and outputs a relevance score.
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: sentence-transformers CrossEncoder model
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank retrieved documents using cross-encoder scoring.

        Args:
            query: user query
            documents: list of retrieved docs
            top_k: number of documents to return after reranking

        Returns:
            Reranked list of documents with updated scores.
        """
        if not documents:
            return []

        # Build (query, doc_text) pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Cross-encoder scoring
        with torch.no_grad():
            scores = self.model.predict(pairs)

        # Attach new scores
        for doc, score in zip(documents, scores):
            doc["score"] = float(score)

        # Sort by relevance score
        reranked = sorted(
            documents,
            key=lambda x: x["score"],
            reverse=True,
        )

        return reranked[:top_k]
