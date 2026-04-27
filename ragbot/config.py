from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    persist_dir: Path = Path(os.getenv("RAG_PERSIST_DIR", "vector_db"))
    store_file: str = os.getenv("RAG_STORE_FILE", "store.json")
    chunk_size_words: int = int(os.getenv("CHUNK_SIZE_WORDS", "180"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "35"))
    vector_size: int = int(os.getenv("VECTOR_SIZE", "768"))
    top_k: int = int(os.getenv("TOP_K", "2"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "smollm2:135m")
