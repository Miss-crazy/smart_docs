# rag/retriver.py
import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

def query(question: str, collection_name: str = "smart_doc", top_k: int = 3) -> str:
    collection = client.get_or_create_collection(name=collection_name)
    
    # embed the question
    response = ollama.embeddings(model="nomic-embed-text", prompt=question)
    question_embedding = response["embedding"]
    
    # find most relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    relevant_chunks = "\n\n".join(results["documents"][0])
    
    # build prompt — this is what prevents hallucinations
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
      If the answer is not in the context, say "I don't know based on the provided document."

      Context:
              {relevant_chunks}

      Question: {question}
              Answer:"""
    
    # ask Ollama
    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}]
    )
    #unload model from ram after generating response to free up resources
    ollama.generate(model="phi3:mini" , prompt="" , keep_alive=0) 
    return response["message"]["content"]