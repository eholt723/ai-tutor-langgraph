

from __future__ import annotations

import argparse

from ai_tutor.config import Config
from ai_tutor.rag.ingest import ingest_reference_corpus
from ai_tutor.rag.store import save_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="If set, rebuild index from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Build RAG Index ===")
    print(f"RAG index path: {Config.rag_index_path}")
    print(f"Rebuild:        {args.rebuild}")
    print()

    docs = ingest_reference_corpus()
    vector_store = save_vector_store(docs, rebuild=args.rebuild)

    print("RAG index built.")
    print(f"Stored at: {Config.rag_index_path}")
    print(f"Number of documents: {len(docs)}")
    print(f"Vector store summary: {vector_store}")


if __name__ == "__main__":
    main()
