# Finance RAG Assistant

A retrieval-augmented generation (RAG) system that answers questions about SEC 10-K filings using local LLMs.

## What it does
- Loads and processes SEC 10-K annual report PDFs
- Converts document chunks into vector embeddings using HuggingFace
- Retrieves relevant context using FAISS similarity search
- Generates accurate, grounded answers using Mistral LLM via Ollama
- Exposes a REST API via FastAPI
- Includes an evaluation pipeline measuring relevancy and faithfulness scores

## Tech Stack
- LangChain — RAG orchestration
- FAISS — vector similarity search
- HuggingFace Embeddings — sentence-transformers/all-MiniLM-L6-v2
- Mistral via Ollama — local LLM inference
- FastAPI — REST API serving
- Streamlit — frontend interface

## Project Structure

```
finance-rag-assistant/
├── data/                  # 10-K PDF storage and FAISS index
├── src/
│   ├── loader.py          # PDF loading and chunking
│   ├── embeddings.py      # Vector store creation and loading
│   ├── retriever.py       # Similarity search retrieval
│   ├── chain.py           # RAG chain combining retriever + LLM
│   └── evaluate.py        # Evaluation pipeline
├── app.py                 # FastAPI application
├── streamlit_app.py       # Streamlit frontend
└── requirements.txt
```

## Setup

1. Clone the repo
2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Install and start Ollama

```bash
ollama pull mistral
ollama serve
```

4. Add your 10-K PDF to the `data/` folder

5. Build the vector store

```bash
python -c "from src.loader import load_and_chunk_pdf; from src.embeddings import create_vector_store; create_vector_store(load_and_chunk_pdf('data/your_file.pdf'))"
```

6. Run the API

```bash
uvicorn app:app --reload
```

7. Run the frontend

```bash
streamlit run streamlit_app.py
```

## API Usage

```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What was Apple total revenue in 2025?"}'
```

## Evaluation

```bash
python -m src.evaluate
```