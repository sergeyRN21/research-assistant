# main.py
from graph import app

# Запуск графа
result = app.invoke({
    "question": "Как уменьшить KV-cache в LLM?",
    "retry_count": 0,
    "papers": [],
    "retrieved_texts": [],
    "hypotheses": [],
    "evidence": [],
    "final_answer": "",
    "error": ""
})

print("\n🎯 Финальный результат:")
print(result["final_answer"])