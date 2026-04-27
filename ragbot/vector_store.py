from __future__ import annotations

from pathlib import Path
import json

from ragbot.chunking import Chunk
from ragbot.embeddings import HashingEmbedder, cosine


class JsonVectorStore:
    def __init__(self, persist_dir: Path, store_file: str) -> None:
        self.persist_dir = persist_dir
        self.path = persist_dir / store_file
        self.items: list[dict] = []
        if self.path.exists():
            self.items = json.loads(self.path.read_text(encoding="utf-8"))

    def reset(self) -> None:
        self.items = []
        if self.path.exists():
            self.path.unlink()

    def add_chunks(self, chunks: list[Chunk], embedder: HashingEmbedder, batch_size: int = 32) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = embedder.embed_documents([chunk.text for chunk in batch])
            for chunk, embedding in zip(batch, embeddings, strict=True):
                self.items.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "embedding": embedding,
                    }
                )
        self.persist()

    def persist(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=2), encoding="utf-8")

    def query(self, question: str, embedder: HashingEmbedder, top_k: int) -> list[dict]:
        if not self.items:
            raise RuntimeError("Vector store is empty. Run index.py --reset first.")

        query_embedding = embedder.embed_query(question)
        ranked = []
        for item in self.items:
            score = cosine(query_embedding, item["embedding"])
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)

        matches = []
        for score, item in ranked[:top_k]:
            metadata = item["metadata"]
            matches.append(
                {
                    "text": item["text"],
                    "metadata": metadata,
                    "score": score,
                    "citation": format_citation(metadata),
                }
            )
        return matches


def format_citation(metadata: dict) -> str:
    source = metadata.get("source", "unknown source")
    locator = metadata.get("locator")
    return f"{source}, {locator}" if locator else str(source)
