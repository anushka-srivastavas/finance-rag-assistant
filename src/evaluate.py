from src.chain import build_qa_chain
from src.retriever import get_retriever


def evaluate_pipeline():
    test_cases = [
        {
            "question": "What was Apple total revenue in 2025?",
            "expected_keywords": ["416", "billion", "million", "revenue", "sales"]
        },
        {
            "question": "What was Apple net income in 2025?",
            "expected_keywords": ["net income", "profit", "billion", "million"]
        },
        {
            "question": "What are the main risk factors?",
            "expected_keywords": ["risk", "competition", "market", "economic"]
        },
        {
            "question": "How many employees does Apple have?",
            "expected_keywords": ["employees", "full-time", "workforce"]
        },
    ]

    chain = build_qa_chain()
    retriever = get_retriever()

    results = []
    print("Running evaluation...\n")

    for test in test_cases:
        question = test["question"]
        answer = chain.invoke(question)
        docs = retriever.invoke(question)

        answer_lower = answer.lower()
        keyword_hits = sum(
            1 for kw in test["expected_keywords"] if kw.lower() in answer_lower
        )
        relevancy_score = round(keyword_hits / len(test["expected_keywords"]), 2)

        context_text = " ".join(doc.page_content for doc in docs).lower()
        answer_words = [w for w in answer_lower.split() if len(w) > 5]
        faithful_hits = sum(1 for w in answer_words if w in context_text)
        faithfulness_score = round(
            min(faithful_hits / max(len(answer_words), 1), 1.0), 2
        )

        results.append({
            "question": question,
            "answer": answer,
            "relevancy_score": relevancy_score,
            "faithfulness_score": faithfulness_score,
        })

        print(f"Q: {question}")
        print(f"A: {answer[:150]}...")
        print(f"Relevancy Score: {relevancy_score}")
        print(f"Faithfulness Score: {faithfulness_score}")
        print("-" * 60)

    avg_relevancy = round(
        sum(r["relevancy_score"] for r in results) / len(results), 2
    )
    avg_faithfulness = round(
        sum(r["faithfulness_score"] for r in results) / len(results), 2
    )

    print(f"\nAverage Relevancy Score: {avg_relevancy}")
    print(f"Average Faithfulness Score: {avg_faithfulness}")
    return results


if __name__ == "__main__":
    evaluate_pipeline()