# Lightweight Document Q&A RAG Bot

This is a lightweight Retrieval-Augmented Generation document Q&A bot built for the AI intern assignment. It indexes local documents, chunks them with overlap, stores local vector embeddings on disk, retrieves relevant chunks for a question, and asks an Ollama model to answer with source citations.

This version is designed for low-disk-space Windows laptops: it uses only Python standard library code and Ollama. There is no ChromaDB, PyTorch, SentenceTransformers, or Visual C++ build requirement.

## Tech Stack

- Python 3.11 or 3.12
- Ollama for local answer generation
- Streamlit 1.39.0 for the optional web UI
- Local hashing-vector embeddings implemented in `ragbot/embeddings.py`
- JSON vector store persisted in `vector_db/store.json`
- Standard-library document parsing for TXT, MD, and simple text-layer PDF files

## Project Structure

- `index.py`: loads documents, chunks them, embeds chunks in batches, and writes the vector store
- `chat.py`: interactive command-line Q&A bot
- `app.py`: Streamlit web interface
- `ragbot/`: RAG pipeline code
- `data/`: sample document collection with 5 documents, including one PDF
- `.env.example`: environment variable template

## Setup

Open PowerShell or Command Prompt and run:

```powershell
cd C:\Users\Admin\.conda\Desktop\RAG-bot
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

You do not need to activate the virtual environment. Run Python directly from `.venv`.

Install Ollama from:

```text
https://ollama.com/download
```

Pull the small model:

```powershell
ollama pull smollm2:135m
```

Make sure `.env` contains:

```env
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=smollm2:135m
TOP_K=2
```

## Run

Index the documents:

```powershell
.\.venv\Scripts\python.exe index.py --reset
```

Start the chatbot:

```powershell
.\.venv\Scripts\python.exe chat.py --top-k 2
```

Type `exit` to quit.

Start the Streamlit UI:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Architecture Overview

1. Document ingestion scans `data/` for `.txt`, `.md`, and `.pdf` files.
2. Text is cleaned and split into paragraph-aware overlapping chunks.
3. Chunks are embedded in batches with a local hashing-vector embedding model.
4. Embeddings and metadata are persisted to `vector_db/store.json`.
5. A user question is embedded and compared with stored vectors using cosine similarity.
6. The top-k chunks are passed to Ollama with a grounding prompt.
7. The answer and source chunks are displayed in the terminal or Streamlit UI.

## Chunking Strategy

The bot uses paragraph-aware fixed-size word chunks. The default chunk size is 180 words with 35 words of overlap. Paragraph boundaries make chunks readable, while overlap reduces context loss at chunk boundaries.

Each chunk stores source filename, page or section, and chunk index.

## Embedding Model and Vector Database

The embedding model is a local hashing-vector embedder. It lowercases text, extracts word tokens, hashes terms into a fixed-size vector, applies term-frequency weighting, and normalizes vectors for cosine search. It is much smaller than neural embedding models and works without internet or large downloads.

The vector database is a local JSON vector store. It persists embeddings, text, and metadata to disk and keeps indexing separate from querying. This is intentionally simple so the full RAG pipeline is easy to explain.

## Example Queries

- What are the main components of a practical AI governance program?
- How can cities reduce heat risk?
- What remote work habits reduce ambiguity?
- Why is public transit frequency important?
- What should public health teams do during emergencies?
- What does the collection say about medieval shipbuilding? This should be answered as unsupported by the documents.

## Screen Recording Checklist

For the 3 to 8 minute demo:

1. Show the folder structure.
2. Run `.\.venv\Scripts\python.exe index.py --reset`.
3. Run `.\.venv\Scripts\python.exe chat.py --top-k 2` or `.\.venv\Scripts\streamlit.exe run app.py`.
4. Ask at least five questions across at least two documents.
5. Show the retrieved source chunks and citations.
6. Ask one unsupported question.
7. Explain the lightweight hashing embeddings and local JSON vector store choice.

## Known Limitations

- The PDF extractor is basic and works best with text-layer PDFs.
- Hashing embeddings are lightweight but less semantically powerful than neural embeddings.
- Answer quality depends on the small Ollama model available on the laptop.
- There is no user access-control layer.
