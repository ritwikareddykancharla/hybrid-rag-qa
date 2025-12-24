from typing import List, Dict


class GroundedQAGenerator:
    """
    LLM-based grounded question answering.

    The model is explicitly instructed to answer
    ONLY using the provided documents.
    """

    def __init__(self, llm):
        """
        Args:
            llm: language model with a generate(prompt: str) method
        """
        self.llm = llm

    def generate(
        self,
        question: str,
        documents: List[Dict],
    ) -> Dict:
        """
        Generate a grounded answer with source citations.

        Args:
            question: user question
            documents: top-ranked retrieved documents

        Returns:
            {
              "answer": str,
              "sources": List[{source_id, snippet, score}]
            }
        """

        if not documents:
            return {
                "answer": "I could not find sufficient evidence to answer this question.",
                "sources": [],
            }

        # Build evidence context
        context_blocks = []
        for i, doc in enumerate(documents):
            context_blocks.append(
                f"[Source {i}] {doc['text']}"
            )

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a question answering system.
Answer the question using ONLY the information in the sources below.
If the answer is not present, say you do not know.

Sources:
{context}

Question:
{question}

Answer (include source numbers):
"""

        answer = self.llm.generate(prompt)

        sources = []
        for i, doc in enumerate(documents):
            sources.append(
                {
                    "source_id": doc["doc_id"],
                    "snippet": doc["text"][:300],
                    "score": doc["score"],
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }
