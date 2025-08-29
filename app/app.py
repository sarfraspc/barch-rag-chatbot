import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.support import load_vectorstore, qa_chain
from fastapi import HTTPException

load_dotenv()

vectorstore = load_vectorstore()
qa = qa_chain(vectorstore)  

app = FastAPI(title="Archie - BArch Chatbot")

app.mount("/static", StaticFiles(directory="app/template"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open(os.path.join("app", "template", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok"}

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(payload: dict):
    query = payload.get("query", "")
    if not query:
        return {"error": "No query provided"}

    try:
        response = qa.invoke(query)

        if isinstance(response, dict):
            answer_text = response.get("result") or response.get("answer") or str(response)
   
        else:
            answer_text = str(response)
        

     
        return {"answer": answer_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
