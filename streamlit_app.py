import streamlit as st
import requests

st.set_page_config(
    page_title="Finance RAG Assistant",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Finance RAG Assistant")
st.markdown("Ask questions about Apple's 2025 SEC 10-K filing")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Apple's financials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching 10-K and generating answer..."):
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": prompt}
                )
                answer = response.json()["answer"]
            except Exception as e:
                answer = f"Error: Make sure the FastAPI server is running. {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.title("About")
st.sidebar.markdown("""
This Website uses a RAG pipeline to answer questions from Apple's 2025 10-K filing.

**How it works:**
1. Your question is embedded into a vector
2. FAISS retrieves the 4 most relevant chunks
3. Mistral generates an answer using those chunks

**Evaluation scores:**
- Relevancy: 0.56
- Faithfulness: 0.06
""")