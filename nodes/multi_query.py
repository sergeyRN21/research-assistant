# nodes/multi_query.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

# --- Исправленный base_url ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1", # <- Убраны пробелы
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-flash-001",
    timeout=30,
    max_retries=2
)

template = """Ты — научный помощник по машинному обучению. Твоя задача — сгенерировать 5 различных формулировок одного и того же вопроса, чтобы улучшить поиск по научным статьям.

Каждый вариант должен:
- Сохранять смысл оригинального вопроса,
- Использовать разные научные термины (например, "KV cache" / "key-value cache", "compression" / "reduction"),
- Подходить для поиска в векторной базе (FAISS).

Верни только 5 вариантов, по одному на строке, без нумерации.

Оригинальный вопрос: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries_chain = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:5])
)

def multi_query(state):
    """
    Узел: Генерирует 5 альтернативных версий вопроса для поиска.
    """
    print("🔁 Узел: Multi-Query — генерация альтернативных запросов...")
    
    question = state.get("question")
    if not question:
        return {"queries": []}

    try:
        queries = generate_queries_chain.invoke({"question": question})
        print(f"✅ Сгенерировано {len(queries)} альтернативных запросов")
        return {"queries": queries}
    
    except Exception as e:
        print(f"❌ Ошибка при генерации запросов: {e}")
        # Возвращаем хотя бы оригинальный вопрос в случае ошибки
        return {"queries": [question]}