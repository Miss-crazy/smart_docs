import uuid
import ollama
import chromadb 

client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(collection_name: str = "smart_docs"):
    return client.get_or_create_collection(name=collection_name)

def reset_collection(collection_name: str = "smart_docs"):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(name=collection_name)

def embed_and_store(
    chunks: list[str],
    collection_name: str = "smart_docs",
    clear_existing: bool = True
):
    if not chunks:
        return
    
    collection = reset_collection(collection_name) if clear_existing else get_or_create_collection(collection_name)
    
    ids = []
    embeddings = []
    documents = []
    
    for i, chunk in enumerate(chunks):
        # nomic-embed-text requires 'search_document: ' prefix for stored chunks
        prompt = f"search_document: {chunk}"
        response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
        embedding = response['embedding']
        
        chunk_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  -> {i + 1}/{len(chunks)} chunks embedded")

            
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents
    )
    
    # unload model from ram after embedding
    ollama.generate(model="nomic-embed-text", prompt="", keep_alive=0)
    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'")

