from typing import List, Dict, Optional
import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

def reformulate_query(question: str, history: Optional[List[Dict[str, str]]]) -> str:
    """
    If conversation history exists, reformulate follow-up questions into
    a standalone search query for vector retrieval.
    """
    if not history or len(history) == 0:
        return question
    
    # Extract last 2-4 turns of history
    recent_history = history[-4:]
    history_text = "\n".join([f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in recent_history])
    
    prompt = f"""Given the following conversation history and follow-up question, rephrase the follow-up question into a standalone search query that contains all necessary context for document retrieval. Do NOT answer the question, just output the standalone search query.

Conversation History:
{history_text}

Follow-up Question: {question}
Standalone Search Query:"""

    try:
        res = ollama.chat(
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        standalone = res["message"]["content"].strip().strip('"')
        return standalone if standalone else question
    except Exception:
        return question

def query(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    collection_name: str = "smart_docs",
    top_k: int = 4
) -> str:
    collection = client.get_or_create_collection(name=collection_name)
    
    # 1. Reformulate question if history exists
    search_term = reformulate_query(question, history) if history else question
    
    # 2. Embed the question using nomic's search_query prefix
    embed_prompt = f"search_query: {search_term}"
    response = ollama.embeddings(model="nomic-embed-text", prompt=embed_prompt)
    question_embedding = response["embedding"]
    
    # 3. Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0] if results.get("documents") and results["documents"] else []
    relevant_chunks = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."
    
    # 4. Build prompt
    system_instruction = (
        "You are SmartDoc, an accurate and helpful AI assistant for document analysis.\n"
        "Instructions:\n"
        "1. Base your answer strictly and factually on the provided Document Context and conversation history.\n"
        "2. Synthesize clear, well-structured, and complete answers.\n"
        "3. If the answer cannot be determined or inferred from the context, state honestly: "
        "\"I don't know based on the provided document.\"\n"
        "4. Do not speculate or invent facts not grounded in the context.\n\n"
        f"Document Context:\n{relevant_chunks}"
    )
    
    messages = [{"role": "system", "content": system_instruction}]
    
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": question})
    
    chat_response = ollama.chat(
        model="phi3:mini",
        messages=messages,
        options={"temperature": 0.2}
    )
    
    # unload model from ram
    ollama.generate(model="phi3:mini", prompt="", keep_alive=0)
    
    return chat_response["message"]["content"]
