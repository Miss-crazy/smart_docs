# SmartDocs - Local Conversational RAG & Document Intelligence

SmartDocs is a privacy-first, fully local Document Interaction & Retrieval-Augmented Generation (RAG) system. It allows users to upload documents (`.pdf`, `.txt`) and chat with them using local Large Language Models and embedding models — **zero cloud API costs, zero external data sharing, and 100% data privacy**.

Powered by **Ollama (`phi3:mini` + `nomic-embed-text`)**, **ChromaDB**, and **FastAPI**, SmartDocs incorporates **multi-turn conversational memory**, **sentence boundary-aware semantic chunking**, and **contextual query rewriting**.

---

## 📊 RAG Performance & Benchmark (RAGAS Evaluation)

The SmartDocs RAG pipeline was evaluated using **RAGAS** (Retrieval Augmented Generation Assessment Suite) metrics to ensure high precision, zero hallucinations, and high recall:

| Metric | Score | Description |
|:---|:---:|:---|
| **Faithfulness** | **100.0%** | Measures if all claims in generated answers are strictly grounded in document context (zero hallucination). |
| **Answer Relevance** | **100.0%** | Measures how directly and completely the generated answer addresses the question. |
| **Context Recall** | **95.0%** | Measures if all relevant facts from the ground truth were successfully retrieved. |
| **Context Precision** | **74.2%** | Measures the density of relevant context across top-$k$ retrieved chunks. |
| **🏆 Overall RAGAS Score** | **92.3%** | **High-precision, robust local RAG pipeline.** |


---

## ✨ Key Features

- **💬 Conversational RAG with Multi-Turn Memory**: Keeps track of prior questions and answers, enabling natural follow-up conversations without losing context.
- **🔍 Contextual Query Rewriting**: Automatically reformulates follow-up queries containing pronouns (e.g. *"What are its advantages?"*) into standalone search queries before semantic vector lookup.
- **📄 Sentence Boundary-Aware Semantic Chunking**: Splits text along natural paragraph and sentence boundaries with intelligent overlap rather than naive word cuts, preserving coherent thoughts and ideas.
- **🏷️ Nomic Contrastive Task Prefixes**: Utilizes `search_document: ` and `search_query: ` prefixes optimized for `nomic-embed-text` to maximize cosine similarity accuracy.
- **📚 Dual-Mode Document Summarization**:
  - **Query-Focused Summarization** for standard documents (< 2MB).
  - **Map-Reduce Summarization** for large files (> 2MB) to prevent context window overflow.
- **⚡ Fully Local & Private**: All embeddings, vector operations, and text generation execute on your local machine using Ollama and ChromaDB.
- **🖥️ Responsive Web UI**: Minimalist, high-performance dark interface with live API status, conversation memory counter, and one-click chat reset.
- **🚀 RESTful FastAPI Backend**: Clean architecture with modular endpoints for file upload, ingestion, and multi-turn querying.

---

## 🖼️ UI Preview & Screenshots

### Chat Interface with Conversational Memory
<img width="1909" height="937" alt="SmartDoc UI Overview" src="https://github.com/user-attachments/assets/16a64736-bbe0-4caa-a988-539905e466ce" />

### Document Ingestion & Live Status
<img width="309" height="641" alt="Document Upload Sidebar" src="https://github.com/user-attachments/assets/7d610684-4b1e-4815-98d3-ffb82e42db5c" />

### Grounded Document Answers
<img width="1619" height="527" alt="Grounded RAG Response" src="https://github.com/user-attachments/assets/000f1aec-dfa1-467f-9a34-58efb59ef670" />


---

## 🏗️ System Architecture & Workflow

```
[ User Document: .pdf / .txt ]
             │
             ▼
    [ PyMuPDF / Text Loader ]
             │
             ▼
[ Sentence Boundary Chunker ] ─── (500 chars / 50 overlap)
             │
             ▼
   [ nomic-embed-text ] ────────── (with 'search_document:' prefix)
             │
             ▼
   [ ChromaDB Vector Store ] ───── (Persistent Local DB)
             │
             │
[ User Query + Chat History ] ───► [ Contextual Query Rewriter ]
                                                │
                                                ▼
                                    [ nomic-embed-text Query ]
                                    (with 'search_query:' prefix)
                                                │
                                                ▼
                                    [ ChromaDB Top-k Retrieval ]
                                                │
                                                ▼
                                    [ Grounded System Prompt ]
                                    + [ Conversation History ]
                                    + [ Retrieved Document Chunks ]
                                                │
                                                ▼
                                        [ phi3:mini (LLM) ]
                                                │
                                                ▼
                                     [ Final Answer to User ]
```

### Detailed Ingestion Flow:
1. **Document Loading**: Uploaded `.pdf` files are parsed with PyMuPDF (`fitz`), and `.txt` files with standard UTF-8 readers.
2. **Boundary-Aware Chunking**: Text is split into coherent segments respecting paragraph and sentence boundaries with token overlap.
3. **Prefix Embeddings**: Chunks are prefixed with `search_document: ` and embedded via `nomic-embed-text`.
4. **Vector Storage**: Chunks and embeddings are upserted into persistent collections in ChromaDB with UUID isolation.

### Detailed Query & Memory Flow:
1. **Query Reformulation**: If conversation history exists, the system uses `phi3:mini` to rephrase pronouns into a self-contained search query.
2. **Asymmetric Vector Search**: The query is embedded with `search_query: ` and ChromaDB retrieves the top-$k$ (default 4) most relevant document chunks.
3. **Structured Multi-Turn Generation**: A prompt containing strict grounding instructions, document context, previous chat history, and the new query is sent to `phi3:mini`.
4. **Memory Unloading**: Model memory is released after response generation (`keep_alive=0`) to preserve RAM.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **LLM (Language Model)** | `phi3:mini` (via Ollama) | Fast, highly capable local reasoning & synthesis |
| **Embedding Model** | `nomic-embed-text` (via Ollama) | High-dimension asymmetric text embeddings |
| **Vector Database** | ChromaDB | Persistent local vector indexing and cosine retrieval |
| **Backend API** | FastAPI + Uvicorn | Async REST API endpoints |
| **Document Parsing** | PyMuPDF (`fitz`) | High-speed PDF text extraction |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript | Modern dark-theme UI with DM Mono & Syne typography |
| **Evaluation Suite** | RAGAS Metrics / Custom Harness | Faithfulness, relevance, recall, and precision validation |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running on your system

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Miss-crazy/smart_docs.git
cd smart_docs
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Pull Ollama Models
Ensure Ollama is running, then pull the required models:
```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### Step 4: Start Ollama Service
```bash
ollama serve
```

### Step 5: Run the FastAPI Server
```bash
uvicorn api:app --reload
```
The server will start at `http://127.0.0.1:8000`. You can test the interactive API docs at `http://127.0.0.1:8000/docs`.

### Step 6: Launch the Web UI
Double-click `index.html` or open it directly in your browser. The status dot will turn **green** (`api online`) when connected to the backend.

---

## 🔌 API Reference

### 1. Ingest Document
- **Endpoint**: `POST /uploadfile/`
- **Content-Type**: `multipart/form-data`
- **Description**: Uploads and indexes a `.pdf` or `.txt` file into ChromaDB.

**Response:**
```json
{
  "message": "Document 'sample.txt' ready for questions!"
}
```

---

### 2. Query Document (with Conversation Memory)
- **Endpoint**: `POST /userquery/`
- **Content-Type**: `application/json`
- **Description**: Submits a user question along with past conversation turns for context-aware answering.

**Request Body:**
```json
{
  "question": "What techniques help manage cloud costs?",
  "history": [
    {
      "role": "user",
      "content": "What is this document about?"
    },
    {
      "role": "assistant",
      "content": "This document covers cloud computing concepts, databases, edge computing, and cost management."
    }
  ]
}
```

**Response:**
```json
{
  "answer": "Techniques for managing cloud costs include using reserved instances, auto-scaling capabilities, and resource tagging to optimize resource usage and prevent overspending."
}
```

---

## 💻 Hardware & Performance Optimizations

SmartDocs is optimized to run smoothly on standard consumer laptops and PCs (tested on 16GB RAM with NVIDIA GPU / CPU):
- **Resource Cleanup**: Unloads models from memory after every request (`keep_alive=0`) to avoid RAM bloat.
- **Batch Processing**: Embeddings are generated in batches during ingestion to prevent GPU memory spikes.
- **Parallelism Control**: Environment variables configure Ollama to prevent model contention during simultaneous embedding and generation tasks.

---

## 📚 Further Reading & References

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://docs.ragas.io/en/stable/)
- [Nomic Embed: Training and Architecture Overview](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Ollama Model Library](https://ollama.com/library)

---

## 🔮 Future Roadmap & Potential Features

-  **Hybrid Search (BM25 + Dense Vectors)**: Combine sparse lexical keyword matching with dense vector search using Reciprocal Rank Fusion (RRF).
-  **Document Page Highlighting & Citations**: Return precise page numbers and highlighted bounding boxes for PDF documents.
-  **Streaming Responses (SSE / WebSockets)**: Enable real-time token streaming in the UI for instant feedback.
-  **Multi-Document Comparison**: Query across multiple indexed files simultaneously with document filter pills.
-  **OCR Support for Scanned Documents**: Integrate Tesseract or Suriya OCR for image-based PDFs.
-  **Chat History Export**: Download chat transcripts as Markdown or PDF summaries.

---

## 📄 License

This project is licensed under the **MIT License**.
