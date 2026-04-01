# main.py
from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed_and_store
from rag.retriver import query

def ingest(file_path: str):
    print(f"Loading {file_path}...")
    text = load_file(file_path)
    
    print("Chunking text...")
    chunks = chunk_text(text)
    
    print("Embedding and storing in ChromaDB...")
    embed_and_store(chunks)
    
    print("Document ready for questions!")

def ask(question: str):
    print(f"\n Question: {question}")
    answer = query(question)
    print(f" Answer: {answer}\n")

if __name__ == "__main__":
    # Step 1: ingest a document
    ingest("sample.pdf")  # swap with your file
    
    # Step 2: ask questions
    ask("What is this document about?")
    ask("Summarize the key points.")