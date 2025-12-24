from typing import Dict, Any, List

from retrieval.hybrid import hybrid_merge


class RAGFlow:
    """
    End-to-end RAG orchestration:
    sparse + dense retrieval -> hybrid merge -> rerank -> grounded generation
    """

    def __init__(
        self,
        sparse_retriever,
        dense_retriever,
        reranker,
        generator,
        retrieve_k: int = 20,
        rerank_k: int = 10,
    ):
        """
        Args:
            sparse_retriever: SparseRetriever
            dense_retriever: DenseRetriever
            reranker: CrossEncoderReranker
            generator: GroundedQAGenerator
            retrieve_k: top-K docs to retrieve from each retriever
            rerank_k: top-K docs after reranking
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.generator = generator
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.

        Args:
            question: user question

        Returns:
            {
              "answer": str,
              "sources": List[Dict]
            }
        """

        # 1. Retrieve (recall-focused)
        sparse_results = self.sparse_retriever.search(
            question, k=self.retrieve_k
        )
        dense_results = self.dense_retriever.search(
            question, k=self.retrieve_k
        )

        # 2. Hybrid merge
        merged_results = hybrid_merge(
            sparse_results, dense_results, k=self.retrieve_k
        )

        # 3. Rerank (precision-focused)
        reranked_results = self.reranker.rerank(
            question, merged_results, top_k=self.rerank_k
        )

        # 4. Grounded generation
        output = self.generator.generate(
            question, reranked_results
        )

        return output
