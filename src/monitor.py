import json
import os
from datetime import datetime


LOG_FILE = "data/query_log.json"


def log_query(question: str, answer: str, num_chunks: int):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "answer_length": len(answer.split()),
        "chunks_retrieved": num_chunks
    }

    logs = load_logs()
    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


def load_logs() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        return json.load(f)


def get_stats() -> dict:
    logs = load_logs()

    if not logs:
        return {"total_queries": 0, "message": "No queries logged yet"}

    avg_answer_length = round(
        sum(l["answer_length"] for l in logs) / len(logs), 2
    )

    return {
        "total_queries": len(logs),
        "avg_answer_length_words": avg_answer_length,
        "recent_questions": [l["question"] for l in logs[-5:]]
    }