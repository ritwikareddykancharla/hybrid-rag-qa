from typing import List, Dict
import numpy as np
import faiss


class DenseRetriever:
    """
    Dense semantic retriever using embeddings + FAISS.
    """

    def __init__(
        self,
        embedder,
        index: faiss.Index,
        documents: List[str],
    ):
        """
        Args:
            embedder: sentence-transformers style model with .encode()
            index: FAISS index built over document embeddings
            documents: list of raw document texts (aligned with index)
        """
        self.embedder = embedder
        self.index = index
        self.documents = documents

    def search(self, query: str, k: int = 20) -> List[Dict]:
        """
        Perform dense semantic retrieval.

        Args:
            query: user query string
            k: number of nearest neighbors

        Returns:
            List of dicts:
            [
              {
                "doc_id": int,
                "text": str,
                "score": float
              }
            ]
        """
        # Encode query → shape (1, d)
        query_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        # FAISS search
        scores, indices = self.index.search(
            np.array(query_vec, dtype="float32"),
            k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append(
                {
                    "doc_id": int(idx),
                    "text": self.documents[idx],
                    "score": float(score),
                }
            )

        return results
