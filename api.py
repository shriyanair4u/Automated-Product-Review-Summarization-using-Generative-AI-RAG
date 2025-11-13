# api.py
from fastapi import FastAPI, HTTPException
from query_rag import rag_summary  # your RAG model

app = FastAPI(
    title="Simple Skincare RAG API",
    version="1.0"
)

# ✅ Single endpoint: https://localhost:5000/?question=hello
@app.get("/")
def root(question: str):
    try:
        summary = rag_summary(question, k=6)
        return {
            "question": question,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
