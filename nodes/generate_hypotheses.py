# nodes/generate_hypotheses.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-flash-001",
    timeout=30,
    max_retries=2
)

def generate_hypotheses(state):
    """
    Узел 3: Генерирует 3 проверяемые гипотезы из вопроса.
    Вход: state["question"]
    Выход: state["hypotheses"] (список строк)
    """
    print("💡 Узел: Генерация гипотез...")
    
    question = state.get("question")
    if not question:
        print("⚠️ Нет вопроса для генерации гипотез.")
        return {"hypotheses": []}
    
    # 🔧 Промпт: чёткая инструкция для LLM
    prompt = ChatPromptTemplate.from_template("""
Ты — научный ассистент. Твоя задача — разбить следующий вопрос в области машинного обучения на 3 конкретные, проверяемые гипотезы.

Каждая гипотеза должна быть:
- Основана на существующих исследованиях,
- Конкретной (не общая фраза),
- Поддающейся проверке по тексту научной статьи.

Не выдумывай. Если не уверен — сформулируй осторожно.

Вопрос: {question}

Верни только список из 3 гипотез, каждая на новой строке, без нумерации и без пояснений.
""")
    
    try:
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        # Разбиваем ответ на строки → список гипотез
        hypotheses = [line.strip() for line in result.split("\n") if line.strip()]
        hypotheses = hypotheses[:3]  # берём максимум 3
        
        print(f"✅ Сгенерировано гипотез: {len(hypotheses)}")
        return {"hypotheses": hypotheses}
    
    except Exception as e:
        print(f"❌ Ошибка при генерации гипотез: {e}")
        return {"hypotheses": [], "error": str(e)}