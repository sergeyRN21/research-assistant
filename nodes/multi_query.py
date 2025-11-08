# nodes/multi_query.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

# Настройка LLM через OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-flash-001",
    temperature=0.0  # детерминированность
)

# Промпт: генерация 5 научных версий запроса
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector database. 

By generating multiple perspectives on the user question, your goal is to help overcome some of the limitations 
of distance-based similarity search. Focus on scientific terminology used in machine learning papers.

Provide these alternative questions separated by newlines. 
Do not number or add prefixes.

Original question: {question}
"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# Цепочка: промпт → LLM → парсинг → список запросов
generate_queries_chain = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

def multi_query(state):
    """
    Узел: Генерирует 5 научно-ориентированных версий вопроса.
    Цель — повысить recall в FAISS.
    """
    print("🔁 Узел: Multi-Query — Query Translation...")
    
    question = state.get("question")
    if not question:
        return {"queries": []}

    try:
        queries = generate_queries_chain.invoke({"question": question})
        print(f"✅ Сгенерировано {len(queries)} альтернативных запросов:")
        for q in queries:
            print(f"   • '{q}'")
        
        return {"queries": queries}
    
    except Exception as e:
        print(f"❌ Ошибка при генерации запросов: {e}")
        return {"queries": [question]}