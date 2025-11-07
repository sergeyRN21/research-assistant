# nodes/validate_evidence.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import re

# Используем тот же LLM через OpenRouter
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="google/gemini-2.0-flash-001"
)


def validate_evidence(state):
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

Верни ТОЛЬКО JSON в следующем формате:
{{"confirmed": true|false, "partial": true|false, "confidence": 0.0–1.0, "reason": "одно предложение объяснения"}}
Без дополнительного текста!
""")

    validated_evidence = []

    for item in evidence_list:
        hypothesis = item["hypothesis"]
        validated_chunks = []

        for chunk_data in item["chunks"]:
            chunk_text = chunk_data["text"]

            try:
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({
                    "hypothesis": hypothesis,
                    "chunk": chunk_text
                })

                # Извлекаем JSON из ответа
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    judgment = json.loads(json_str)
                else:
                    print(f"⚠️ Не найден JSON в ответе: {result}")
                    judgment = {
                        "confirmed": False,
                        "partial": False,
                        "confidence": 0.1,
                        "reason": "parse error"
                    }

            except json.JSONDecodeError:
                print(f"⚠️ Ошибка парсинга JSON: {result}")
                judgment = {
                    "confirmed": False,
                    "partial": False,
                    "confidence": 0.1,
                    "reason": "parse error"
                }
            except Exception as e:
                print(f"❌ Ошибка при валидации: {e}")
                judgment = {
                    "confirmed": False,
                    "partial": False,
                    "confidence": 0.1,
                    "reason": "parse error"
                }

            validated_chunks.append({
                "text": chunk_text,
                "judgment": judgment
            })

        validated_evidence.append({
            "hypothesis": hypothesis,
            "validated_chunks": validated_chunks
        })

    print("✅ Все доказательства проверены.")

    # 🔥 Обновляем retry_count: если это первый проход — увеличиваем
    current_retry = state.get("retry_count", 0)
    new_retry = current_retry + 1 if current_retry == 0 else current_retry

    return {
        "evidence": validated_evidence,
        "retry_count": new_retry
    }