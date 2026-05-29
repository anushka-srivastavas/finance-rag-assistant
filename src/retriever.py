from src.embeddings import load_vector_store


def get_retriever(k: int = 4):
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print(f"Retriever ready — will fetch top {k} chunks per query")
    return retriever