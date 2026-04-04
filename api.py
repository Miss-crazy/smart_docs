from pydantic import BaseModel
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from rag.retriever import query 
from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed_and_store
import os
import tempfile

app = FastAPI()
class QuestionRequest(BaseModel):
    question:str

@app.post("/uploadfile/")
def upload_file(file : UploadFile):
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
def user_query(request : QuestionRequest):
    question = request.question
    answer = query(question)
    return {"answer": answer}
