# SmartDocs

An AI-powered document interaction system that runs **fully locally** — no API costs, no cloud dependencies. Upload a document and interact with it through conversational RAG-powered Q&A (with multi-turn memory) and summarization.

---

## Features

- **Ask questions with Conversation Memory** — retrieval-augmented generation (RAG) grounds every answer strictly in your document, with multi-turn memory so the LLM remembers previous messages for context-aware follow-ups.
- **Summarization** — query-focused summarization for small documents; map-reduce for files larger than 2MB.
- **REST API** — FastAPI backend with endpoints for file ingestion and querying.
- **Clean UI** — minimal dark frontend, no framework dependencies.

---

## How it works

### Ingestion phase
1. User uploads a `.pdf` or `.txt` file via the API
2. Text is extracted from the file
3. Text is split into overlapping 200-word chunks
4. Each chunk is converted to a vector embedding using `nomic-embed-text`
5. Embeddings are stored persistently in ChromaDB

### Query phase with Conversational Memory
1. User submits a question along with prior conversation history
2. Question is embedded using `nomic-embed-text`
3. ChromaDB performs cosine similarity search and returns the top 3 most relevant chunks
4. A grounded system prompt is built containing the retrieved document chunks
5. Conversation turns (`user` and `assistant`) are assembled in sequence with the new question
6. `phi3:mini` generates a context-aware answer strictly grounded in the document

### Summarization
- Files **under 2MB** — query-focused summarization: top 8–10 chunks are retrieved and summarized in one pass
- Files **over 2MB** — map-reduce: each chunk is summarized individually, then summaries are combined into a final summary

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM | `phi3:mini` via Ollama (local) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB (persistent, local) |
| API | FastAPI + Uvicorn |
| PDF parsing | PyMuPDF (fitz) |
| Language | Python 3.11+ |

---

## Getting started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone the repo
```bash
git clone https://github.com/your-username/smart_docs.git
cd smart_docs
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### 4. Start Ollama
```bash
ollama serve
```

### 5. Start the API
```bash
uvicorn api:app --reload
```

### 6. Open the UI
Open `index.html` in your browser. The status indicator will turn green when the API is online.

---

## API endpoints

### `POST /uploadfile/`
Upload and ingest a document.

**Request:** multipart/form-data with a `.pdf` or `.txt` file

**Response:**
```json
{ "message": "Document 'sample.txt' ready for questions!" }
```

---

### `POST /userquery/`
Ask a question about the ingested document with optional conversation history.

**Request:**
```json
{
  "question": "What are its key findings?",
  "history": [
    { "role": "user", "content": "What is this document about?" },
    { "role": "assistant", "content": "This document discusses..." }
  ]
}
```

**Response:**
```json
{ "answer": "The key findings include..." }
```

---

## Further reading

- [RAG — Retrieval Augmented Generation](https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c)
- [ChromaDB docs](https://docs.trychroma.com)
- [FastAPI docs](https://fastapi.tiangolo.com)
- [Ollama model library](https://ollama.com/library)

---

## Hardware notes

Built and tested on 16GB RAM with an NVIDIA GPU. Designed to run comfortably on consumer hardware:
- Models are unloaded from RAM immediately after use (`keep_alive=0`)
- Ollama is configured to load only one model at a time
- Chunk size is tuned to 200 words to avoid memory spikes during embedding

---

## License

MIT
