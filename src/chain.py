from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever


def build_qa_chain():
    llm = OllamaLLM(model="mistral")

    prompt_template = PromptTemplate.from_template("""You are a financial analyst assistant.
Use the following context from an SEC 10-K filing to answer the question.
Be specific and cite numbers where available.
If you don't know the answer from the context, say so clearly.

Context:
{context}

Question: {question}

Answer:""")

    retriever = get_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x))),
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("QA chain ready")
    return chain