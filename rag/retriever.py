from typing import List, Dict, Optional
import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

def query(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    collection_name: str = "smart_docs",
    top_k: int = 3
) -> str:
    collection = client.get_or_create_collection(name=collection_name)
    
    # embed the question
    response = ollama.embeddings(model="nomic-embed-text", prompt=question)
    question_embedding = response["embedding"]
    
    # find most relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    relevant_chunks = "\n\n".join(results["documents"][0]) if results["documents"] and results["documents"][0] else ""
    
    # build system message with context and instructions
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer questions based on the provided document context "
            "and the ongoing conversation history.\n"
            "If the answer cannot be found or deduced from the context, say "
            "\"I don't know based on the provided document.\"\n\n"
            f"Document Context:\n{relevant_chunks}"
        )
    }
    
    messages = [system_message]
    
    # Append prior conversation history if provided
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
    
    # Append current user question
    messages.append({"role": "user", "content": question})
    
    # ask Ollama
    chat_response = ollama.chat(
        model="phi3:mini",
        messages=messages
    )
    
    # unload model from ram after generating response to free up resources
    ollama.generate(model="phi3:mini", prompt="", keep_alive=0)
    
    return chat_response["message"]["content"]