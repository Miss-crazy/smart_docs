from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from rag.retriever import query 
from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed_and_store
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="SmartDoc API", description="Document QA with RAG and Conversational Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

@app.post("/uploadfile/")
def upload_file(file: UploadFile):
    contents = file.file.read()

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    text = load_file(tmp_path)
    chunks = chunk_text(text)
    embed_and_store(chunks)

    os.remove(tmp_path)

    return {"message": f"Document '{file.filename}' ready for questions!"}

@app.post("/userquery/")
def user_query(request: QuestionRequest):
    question = request.question
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history] if request.history else []
    answer = query(question=question, history=history_dicts)
    return {"answer": answer}

