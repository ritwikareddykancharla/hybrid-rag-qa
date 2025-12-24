import argparse
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_documents(path: str) -> List[str]:
    """
    Load documents from a text file (one document per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs


def build_faiss_index(
    documents: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Build a FAISS index from document embeddings.
    """
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings


def main(args):
    documents = load_documents(args.input)

    index, _ = build_faiss_index(
        documents,
        model_name=args.model,
    )

    # Save FAISS index
    faiss.write_index(index, args.index_out)

    # Save raw documents (aligned with index order)
    with open(args.docs_out, "wb") as f:
        pickle.dump(documents, f)

    print(f"Saved FAISS index to {args.index_out}")
    print(f"Saved documents to {args.docs_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--index-out", type=str, default="faiss.index")
    parser.add_argument("--docs-out", type=str, default="documents.pkl")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    args = parser.parse_args()
    main(args)
