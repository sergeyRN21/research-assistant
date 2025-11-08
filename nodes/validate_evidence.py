# nodes/validate_evidence.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import re

# --- Исправленный base_url ---
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1", # <- Убраны пробелы
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-flash-001"
)

def validate_evidence(state):
    """
    Узел: Проверяет, подтверждают ли фрагменты гипотезы.
    Использует LLM-as-a-Judge с учётом метаданных.
    """
    print("✅ Узел: Валидация доказательств (LLM-as-a-Judge)...")

    evidence_list = state.get("evidence", [])

    if not evidence_list:
        print("⚠️ Нет доказательств для валидации.")
        return {"evidence": [], "retry_count": state.get("retry_count", 0)}

    prompt = ChatPromptTemplate.from_template("""
Ты — эксперт по научным статьям. Оцени, насколько следующий фрагмент текста **подтверждает** гипотезу.

Формат оценки:
- Подтверждено полностью: если в тексте прямо или косвенно говорится то же самое.
- Частично подтверждено: если есть намёк, но не хватает деталей.
- Не подтверждено: если текст не относится к гипотезе или противоречит ей.

Гипотеза: {hypothesis}
Фрагмент текста: {chunk}
Метаданные: {metadata}

Верни ТОЛЬКО JSON:
{{"confirmed": true|false, "partial": true|false, "confidence": 0.0–1.0, "reason": "одно предложение объяснения"}}
Без дополнительного текста!
""")

    validated_evidence = []

    for item in evidence_list:
        hypothesis = item["hypothesis"]
        validated_chunks = []

        for chunk_data in item["chunks"]:
            chunk_text = chunk_data["text"]
            chunk_metadata = chunk_data["metadata"]

            try:
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({
                    "hypothesis": hypothesis,
                    "chunk": chunk_text,
                    "metadata": chunk_metadata
                })

                # Пытаемся найти JSON внутри ответа
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    judgment = json.loads(json_match.group(0))
                else:
                    print(f"⚠️ Не удалось найти JSON в ответе LLM: {result}")
                    judgment = {
                        "confirmed": False,
                        "partial": False,
                        "confidence": 0.1,
                        "reason": "parse error"
                    }

            except json.JSONDecodeError:
                print(f"⚠️ Ошибка парсинга JSON: {json_match.group(0) if json_match else 'No JSON found'}")
                judgment = {
                    "confirmed": False,
                    "partial": False,
                    "confidence": 0.1,
                    "reason": "json decode error"
                }
            except Exception as e:
                print(f"⚠️ Ошибка при вызове LLM: {e}")
                judgment = {
                    "confirmed": False,
                    "partial": False,
                    "confidence": 0.1,
                    "reason": "llm call error"
                }

            validated_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "judgment": judgment
            })

        validated_evidence.append({
            "hypothesis": hypothesis,
            "validated_chunks": validated_chunks
        })

    print("✅ Все доказательства проверены.")
    
    current_retry = state.get("retry_count", 0)
    # Увеличиваем retry_count при первом проходе через этот узел, если условие для повтора сработает
    new_retry = current_retry + 1 if current_retry == 0 else current_retry

    return {
        "evidence": validated_evidence,
        "retry_count": new_retry
    }