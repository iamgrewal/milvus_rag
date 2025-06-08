# graphrag/rag_system/main.py
from fastapi import FastAPI, HTTPException
from graphrag.rag_system.service import RAGService

app = FastAPI()
rag_service = RAGService()

@app.get("/query")
async def query_rag(q: str):
    try:
        return await rag_service.query_with_retry(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
