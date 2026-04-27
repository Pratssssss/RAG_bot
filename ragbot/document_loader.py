from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class DocumentPage:
    text: str
    source: str
    page: int | None = None
    section: str | None = None


def load_documents(data_dir: Path) -> list[DocumentPage]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data folder: {data_dir}")

    pages: list[DocumentPage] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            pages.extend(load_document(path))

    if not pages:
        raise ValueError(f"No supported documents found in {data_dir}")
    return pages


def load_document(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = _clean_text(path.read_text(encoding="utf-8"))
        return [DocumentPage(text=text, source=path.name, section="document")] if text else []
    if suffix == ".pdf":
        return _load_simple_pdf(path)
    raise ValueError(f"Unsupported document type: {path}")


def _load_simple_pdf(path: Path) -> list[DocumentPage]:
    raw = path.read_text(encoding="latin-1", errors="ignore")
    page_parts = re.split(r"/Type\s*/Page\b", raw)
    pages: list[DocumentPage] = []
    candidates = page_parts[1:] if len(page_parts) > 1 else [raw]
    for index, part in enumerate(candidates, start=1):
        strings = re.findall(r"\((.*?)\)\s*Tj", part, flags=re.S)
        text = _clean_text("\n".join(_unescape_pdf(s) for s in strings))
        if text:
            pages.append(DocumentPage(text=text, source=path.name, page=index))
    return pages


def _unescape_pdf(text: str) -> str:
    return (
        text.replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace(r"\\", "\\")
        .replace(r"\n", "\n")
    )


def _clean_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if re.fullmatch(r"(page\s*)?\d+", line.lower()):
            continue
        lines.append(line)
    return "\n".join(lines)
