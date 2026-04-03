import ollama
import chromadb 

client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(collection_name : str):
    return client.get_or_create_collection(name=collection_name)

def embed_and_store(chunks : list[str] , collection_name : str ="smart_docs"):
    collection = get_or_create_collection(collection_name)
    
    for i , chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt= chunk)
        embedding = response['embedding']
        collection.add(
            ids=[f"chunk_{i}"] ,
            embeddings = [embedding] ,
            documents = [chunk]
        )
        if i%10 ==0:
            print(f"  → {i}/{len(chunks)} chunks embedded")
    #unload model from ram after embedding to free up resources
    ollama.generate(model="nomic-embed-text" , prompt="" , keep_alive=0)
    
    print(f"stored {len(chunks)} chunks in collection '{collection_name}'")
