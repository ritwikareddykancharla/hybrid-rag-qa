from typing import Dict, List


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    Lowercasing keeps it robust.
    """
    return text.lower().split()


def faithfulness_score(
    answer: str,
    sources: List[Dict],
) -> float:
    """
    Compute a simple faithfulness score.

    Measures what fraction of answer tokens
    appear in the retrieved source texts.

    Args:
        answer: generated answer string
        sources: list of source dicts with "snippet" or "text"

    Returns:
        Faithfulness score in [0, 1]
    """
    if not answer or not sources:
        return 0.0

    answer_tokens = tokenize(answer)

    source_text = " ".join(
        src.get("snippet", "") for src in sources
    )
    source_tokens = set(tokenize(source_text))

    if not answer_tokens:
        return 0.0

    supported = [
        tok for tok in answer_tokens if tok in source_tokens
    ]

    return len(supported) / len(answer_tokens)


def evaluate_faithfulness(
    outputs: List[Dict],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate faithfulness over a dataset.

    Args:
        outputs: list of model outputs
                 each containing "answer" and "sources"
        threshold: minimum faithfulness to count as grounded

    Returns:
        {
          "average_faithfulness": float,
          "hallucination_rate": float
        }
    """
    scores = []
    hallucinations = 0

    for out in outputs:
        score = faithfulness_score(
            out.get("answer", ""),
            out.get("sources", []),
        )
        scores.append(score)

        if score < threshold:
            hallucinations += 1

    avg_faithfulness = sum(scores) / len(scores) if scores else 0.0
    hallucination_rate = (
        hallucinations / len(scores) if scores else 0.0
    )

    return {
        "average_faithfulness": avg_faithfulness,
        "hallucination_rate": hallucination_rate,
    }
