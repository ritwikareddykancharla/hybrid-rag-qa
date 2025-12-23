from typing import List, Dict, Any


class SparseRetriever:
    """
    Sparse retriever using BM25-style lexical matching.

    In production, this is backed by OpenSearch / Elasticsearch.
    Here, we keep the interface clean and minimal.
    """

    def __init__(self, client: Any, index_name: str):
        """
        Args:
            client: OpenSearch / Elasticsearch client
            index_name: name of the text index
        """
        self.client = client
        self.index_name = index_name

    def search(self, query: str, k: int = 20) -> List[Dict]:
        """
        Perform sparse retrieval.

        Args:
            query: user query string
            k: number of documents to retrieve

        Returns:
            List of dicts:
            [
              {
                "doc_id": str,
                "text": str,
                "score": float
              }
            ]
        """
        response = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "text": {
                            "query": query
                        }
                    }
                }
            },
            size=k
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "doc_id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "score": float(hit["_score"]),
                }
            )

        return results
