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
    Узел 3: Генерирует проверяемые гипотезы из вопроса.
    Вход = state['question']
    Выход = state['hypotheses']
    """
    print("💡 Узел: Генерация гипотез")

    question = state.get("question")
    
    return {"hypotheses": ["hypothetical hypothesis"]}