from __future__ import annotations

import argparse

from ragbot.config import Settings
from ragbot.embeddings import HashingEmbedder
from ragbot.llm import generate_answer
from ragbot.vector_store import JsonVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against indexed documents.")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    top_k = args.top_k or settings.top_k
    embedder = HashingEmbedder(settings.vector_size)
    store = JsonVectorStore(settings.persist_dir, settings.store_file)

    print("Document Q&A bot. Type 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        chunks = store.query(question, embedder, top_k)
        answer = generate_answer(question, chunks, settings)

        print("\nAnswer:")
        print(answer)
        print("\nSources used:")
        for index, chunk in enumerate(chunks, start=1):
            preview = " ".join(chunk["text"].split())[:260]
            print(f"{index}. {chunk['citation']} (score={chunk['score']:.3f})")
            print(f"   {preview}...")


if __name__ == "__main__":
    main()
