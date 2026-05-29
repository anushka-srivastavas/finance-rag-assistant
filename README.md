# Finance RAG Assistant

A RAG-powered Q&A system that answers questions about SEC 10-K filings using local LLMs — built with LangChain, FAISS, Mistral, and FastAPI.

---

## What it does

- Loads and processes SEC 10-K annual report PDFs
- Converts document chunks into vector embeddings using HuggingFace
- Retrieves relevant context using FAISS similarity search
- Generates accurate, grounded answers using Mistral LLM via Ollama
- Logs all queries with a `/stats` monitoring endpoint for tracking usage and answer quality
- Exposes a REST API via FastAPI
- Includes a custom evaluation pipeline measuring retrieval relevancy and answer faithfulness

---

## Monitoring

Every query is logged to `data/query_log.json` with:
- Timestamp
- Question asked
- Answer generated
- Number of chunks retrieved

The `/stats` endpoint exposes aggregate metrics:

```json
{
  "total_queries": 5,
  "avg_answer_length_words": 22.4,
  "recent_questions": ["What was Apple revenue in 2025?"]
}
```

Designed to track model usage and detect answer quality degradation over time.

---

## Tech Stack

| Component | Tool |
|---|---|
| RAG orchestration | LangChain |
| Vector store | FAISS |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 |
| LLM | Mistral via Ollama |
| API | FastAPI |
| Frontend | Streamlit |
| Monitoring | Custom logging + /stats endpoint |

---

## Project Structure

finance-rag-assistant/
├── data/                  # 10-K PDF storage, FAISS index, query logs
├── src/
│   ├── loader.py          # PDF loading and chunking
│   ├── embeddings.py      # Vector store creation and loading
│   ├── retriever.py       # Similarity search retrieval
│   ├── chain.py           # RAG chain combining retriever + LLM
│   ├── monitor.py         # Query logging and stats
│   └── evaluate.py        # Evaluation pipeline
├── app.py                 # FastAPI application
├── streamlit_app.py       # Streamlit frontend
└── requirements.txt

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/anushka-srivastavas/finance-rag-assistant.git
cd finance-rag-assistant
```

**2. Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Install and start Ollama**

```bash
ollama pull mistral
ollama serve
```

**4. Add your 10-K PDF to the `data/` folder**

**5. Build the vector store**

```bash
python -c "from src.loader import load_and_chunk_pdf; from src.embeddings import create_vector_store; create_vector_store(load_and_chunk_pdf('data/your_file.pdf'))"
```

**6. Run the API**

```bash
uvicorn app:app --reload
```

**7. Run the frontend**

```bash
streamlit run streamlit_app.py
```

---

## API Usage

**Ask a question:**

```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What was Apple total revenue in 2025?"}'
```

**Check monitoring stats:**

```bash
curl http://localhost:8000/stats
```

---

## Evaluation

```bash
python -m src.evaluate
```

---

## Author

Anushka Srivastava — [LinkedIn](https://linkedin.com/in/anushka-srivastava1633) · [GitHub](https://github.com/anushka-srivastavas)