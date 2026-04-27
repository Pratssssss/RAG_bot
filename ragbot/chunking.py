from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import uuid

from ragbot.document_loader import DocumentPage


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def chunk_pages(pages: list[DocumentPage], chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    if overlap_words >= chunk_size_words:
        raise ValueError("Chunk overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n+", page.text) if p.strip()]
        current: list[str] = []
        current_words = 0
        chunk_index = 0

        for paragraph in paragraphs:
            count = len(paragraph.split())
            if current and current_words + count > chunk_size_words:
                chunks.append(_make_chunk(page, current, chunk_index))
                chunk_index += 1
                current = _overlap(current, overlap_words)
                current_words = len(" ".join(current).split())
            current.append(paragraph)
            current_words += count

        if current:
            chunks.append(_make_chunk(page, current, chunk_index))
    return chunks


def _overlap(parts: list[str], words_to_keep: int) -> list[str]:
    words = " ".join(parts).split()
    return [" ".join(words[-words_to_keep:])] if words and words_to_keep > 0 else []


def _make_chunk(page: DocumentPage, parts: list[str], chunk_index: int) -> Chunk:
    text = "\n".join(parts)
    locator = f"page {page.page}" if page.page is not None else page.section or "document"
    raw = f"{page.source}:{locator}:{chunk_index}:{text[:120]}"
    chunk_id = str(uuid.UUID(hashlib.md5(raw.encode("utf-8")).hexdigest()))
    metadata: dict[str, str | int] = {"source": page.source, "locator": locator, "chunk": chunk_index}
    if page.page is not None:
        metadata["page"] = page.page
    return Chunk(id=chunk_id, text=text, metadata=metadata)
