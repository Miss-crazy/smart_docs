import os
os.environ["OLLAMA_NUM_PARALLEL"]="1"
os.environ["OLLAMA_MAX_LOADED_MODELS"]="1"

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed_and_store
from rag.summarization.chunk_summary import map_reduce_summary

LARGE_FILE_THRESHOLD_BYTES = 2 * 1024 * 1024  # 2 MB, adjust as needed

def ingest(file_path: str):
    print(f"Loading {file_path}...")
    text = load_file(file_path)

    file_size = os.path.getsize(file_path)
    is_large = file_size >= LARGE_FILE_THRESHOLD_BYTES

    if is_large:
        print("Large file detected. Running map-reduce summarization...")
        chunks = chunk_text(text, chunk_size=1200, overlap=100)
        map_summaries, final_summary = map_reduce_summary(chunks)

        print("Embedding summarized chunks...")
        embed_and_store(map_summaries, collection_name="smart_docs")

        print("Embedding final reduced summary...")
        embed_and_store([final_summary], collection_name="smart_docs_summary")
    else:
        print("Small file detected. Using normal chunking...")
        chunks = chunk_text(text)
        embed_and_store(chunks, collection_name="smart_docs")
    
    print("Document ready for questions!")





