from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chain import build_qa_chain

app = FastAPI(
    title="Finance RAG Assistant",
    description="Ask questions about SEC 10-K filings",
    version="1.0.0"
)

chain = build_qa_chain()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = chain.invoke(request.question)
    return AnswerResponse(question=request.question, answer=answer)