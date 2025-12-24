import argparse
from typing import List


def load_raw_documents(path: str) -> List[str]:
    """
    Load raw documents from a text file.
    Each line is treated as one document.
    """
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs


def chunk_document(
    text: str,
    max_tokens: int = 200,
) -> List[str]:
    """
    Naive document chunking by whitespace length.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i : i + max_tokens])
        chunks.append(chunk)

    return chunks


def ingest(
    input_path: str,
    output_path: str,
    chunk: bool = False,
    max_tokens: int = 200,
):
    """
    Ingest raw documents and write cleaned corpus.
    """
    raw_docs = load_raw_documents(input_path)

    processed_docs = []
    for doc in raw_docs:
        if chunk:
            processed_docs.extend(
                chunk_document(doc, max_tokens=max_tokens)
            )
        else:
            processed_docs.append(doc)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in processed_docs:
            f.write(doc + "\n")

    print(f"Ingested {len(processed_docs)} documents → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--chunk", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)

    args = parser.parse_args()

    ingest(
        input_path=args.input,
        output_path=args.output,
        chunk=args.chunk,
        max_tokens=args.max_tokens,
    )
