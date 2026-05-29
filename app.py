from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chain import build_qa_chain
from src.retriever import get_retriever
from src.monitor import log_query, get_stats

app = FastAPI(
    title="Finance RAG Assistant",
    description="Ask questions about SEC 10-K filings",
    version="1.0.0"
)

chain = build_qa_chain()
retriever = get_retriever()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/stats")
def get_monitoring_stats():
    return get_stats()


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    docs = retriever.invoke(request.question)
    answer = chain.invoke(request.question)

    log_query(
        question=request.question,
        answer=answer,
        num_chunks=len(docs)
    )

    return AnswerResponse(question=request.question, answer=answer)