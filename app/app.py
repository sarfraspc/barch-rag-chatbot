from fastapi import FastAPI
from pydantic import BaseModel
from app.support import load_vectorstore,qa_chain
from dotenv import load_dotenv

load_dotenv()

vectorestore = load_vectorstore()

qa = qa_chain(vectorestore)

app = FastAPI(title="Archi", version="1.0")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
 
    result = qa_chain({"query": request.query})
    return {
        "answer": result.get("result"),
        "sources": [doc.metadata for doc in result.get("source_documents", [])]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
