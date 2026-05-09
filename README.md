# SmartDoc 📄

An AI-powered document interaction system that lets you upload 
documents and ask questions, get summaries, and analyse sentiment 
— all running 100% locally with no API costs.

## Features
- Ask questions about uploaded documents (RAG pipeline)
- Grounded answers — refuses to hallucinate outside document context
- Emotion/sentiment analysis on any text
- Fully local — no OpenAI key, no cloud costs

## Tech Stack
| Layer | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | phi3:mini via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB |
| Sentiment model | j-hartmann/emotion-english-distilroberta-base |
| File support | PDF (PyMuPDF), plain text |

## Architecture
<img width="1440" height="1432" alt="image" src="https://github.com/user-attachments/assets/e7622ebd-ebc8-490a-952f-d379dfe3fa5e" />


## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed

### Installation
git clone https://github.com/your-username/smart_docs
cd smart_docs
pip install -r requirements.txt
ollama pull phi3:mini
ollama pull nomic-embed-text

### Run
ollama serve          # terminal 1
uvicorn api:app --reload  # terminal 2
# open index.html in browser

## How it works
1. Upload a PDF or txt file via the UI
2. The file is chunked into 200-word overlapping segments
3. Each chunk is embedded using nomic-embed-text and stored in ChromaDB
4. When you ask a question, it's embedded and matched against stored chunks
5. Top 3 matching chunks are passed as context to phi3:mini
6. The LLM answers strictly from context — preventing hallucinations

## Screenshots
[upload your UI screenshots here]

## Resources
- [FastAPI docs](https://fastapi.tiangolo.com)
- [Ollama](https://ollama.com)
- [ChromaDB docs](https://docs.trychroma.com)
- [RAG explained](https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c)
