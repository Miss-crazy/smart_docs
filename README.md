# SmartDocs

An AI-powered document interaction system that runs **fully locally** — no API costs, no cloud dependencies. Upload a document and interact with it through RAG-powered Q&A, sentiment analysis, and summarization.

---

## Features

- **Ask questions** — retrieval-augmented generation (RAG) grounds every answer strictly in your document. No hallucinations.
- **Sentiment analysis** — emotion detection on any text using a fine-tuned transformer model.
- **Summarization** — query-focused summarization for small documents; map-reduce for files larger than 2MB.
- **REST API** — FastAPI backend with endpoints for file ingestion and querying.
- **Clean UI** — minimal dark frontend, no framework dependencies.

---

## Demo

<img width="1909" height="937" alt="image" src="https://github.com/user-attachments/assets/16a64736-bbe0-4caa-a988-539905e466ce" />
<img width="309" height="641" alt="image" src="https://github.com/user-attachments/assets/7d610684-4b1e-4815-98d3-ffb82e42db5c" />
<img width="1619" height="527" alt="image" src="https://github.com/user-attachments/assets/000f1aec-dfa1-467f-9a34-58efb59ef670" />
<img width="1892" height="558" alt="image" src="https://github.com/user-attachments/assets/6385db61-03a9-4c44-aef5-fab2b923e709" />
<img width="1583" height="383" alt="image" src="https://github.com/user-attachments/assets/ba3f05a6-ed75-49d7-b60f-db5378a7ae1c" />
<img width="1594" height="388" alt="image" src="https://github.com/user-attachments/assets/6b1cea66-c59f-46be-84a9-dad0244af4ce" />


---

## How it works

### Ingestion phase
1. User uploads a `.pdf` or `.txt` file via the API
2. Text is extracted from the file
3. Text is split into overlapping 200-word chunks
4. Each chunk is converted to a vector embedding using `nomic-embed-text`
5. Embeddings are stored persistently in ChromaDB

### Query phase
1. User submits a question
2. Question is embedded using the same model
3. ChromaDB performs cosine similarity search and returns the top 3 most relevant chunks
4. A grounded prompt is built: `answer ONLY from the context below`
5. `phi3:mini` generates an answer strictly from retrieved context

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
| Sentiment model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| PDF parsing | PyMuPDF (fitz) |
| Language | Python 3.11+ |


---
##Work Flow
<img width="1336" height="1218" alt="image" src="https://github.com/user-attachments/assets/5fd095db-520e-4612-a2cd-1401cce05254" />

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
{ "message": "Document ready for questions!" }
```

---

### `POST /userquery/`
Ask a question about the ingested document.

**Request:**
```json
{ "question": "What is this document about?" }
```

**Response:**
```json
{ "answer": "..." }
```

---

### `POST /sentiment/`
Analyse the emotional tone of any text.

**Request:**
```json
{ "text": "I'm so excited about this!" }
```

**Response:**
```json
{ "label": "joy", "score": 0.94 }
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
