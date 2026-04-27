from __future__ import annotations

import argparse

from ragbot.chunking import chunk_pages
from ragbot.config import Settings
from ragbot.document_loader import load_documents
from ragbot.embeddings import HashingEmbedder
from ragbot.vector_store import JsonVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local documents into the JSON vector store.")
    parser.add_argument("--reset", action="store_true", help="Clear the old vector store before indexing")
    args = parser.parse_args()

    settings = Settings()
    pages = load_documents(settings.data_dir)
    chunks = chunk_pages(pages, settings.chunk_size_words, settings.chunk_overlap_words)

    print(f"Loaded {len(pages)} pages/sections from {settings.data_dir}")
    print(f"Created {len(chunks)} chunks")

    embedder = HashingEmbedder(settings.vector_size)
    store = JsonVectorStore(settings.persist_dir, settings.store_file)
    if args.reset:
        store.reset()
    store.add_chunks(chunks, embedder)
    print(f"Saved vector store to {settings.persist_dir / settings.store_file}")


if __name__ == "__main__":
    main()
