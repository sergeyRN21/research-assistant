# nodes/extract_text.py
import requests
from pypdf import PdfReader
import io
import re

def extract_text(state):
    """
    Узел 2: Извлекает текст из PDF.
    Отсеивает PDF с метаданными, грантами, лицензиями.
    Оставляет только статьи с научным содержанием.
    """
    print("📄 Узел: Извлечение текста из PDF...")
    
    papers = state.get("papers", [])
    if not papers:
        print("⚠️ Нет статей для извлечения.")
        return {"retrieved_texts": []}
    
    retrieved_texts = []
    for i, paper in enumerate(papers):
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            continue

        print(f"📥 Скачиваем PDF [{i+1}/{len(papers)}]: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()

            pdf = PdfReader(io.BytesIO(response.content))
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            
            # 🔍 Чистим и анализируем текст
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # ❌ Индикаторы "мусорного" PDF
            bad_indicators = [
                len(clean_text) < 1000,
                "Personal use of this material is permitted" in clean_text,
                "©" in clean_text[:300] and "All rights reserved" in clean_text,
                "This research was supported by" in clean_text,
                "grant" in clean_text[:500] and "funded" in clean_text[:500],
                "IEEE" in clean_text[:200] and "Proceedings" in clean_text[:300],
            ]
            if any(bad_indicators):
                print("⚠️ Пропускаем: это не полный текст статьи")
                retrieved_texts.append("")
                continue
            
            # 🔍 Проверяем наличие научной структуры
            structure_keywords = ["abstract", "introduction", "method", "experiment", "results", "conclusion"]
            if not any(kw in clean_text.lower()[:1000] for kw in structure_keywords):
                print("⚠️ Пропускаем: нет структуры научной статьи")
                retrieved_texts.append("")
                continue
            
            # 🔍 Проверяем наличие ML-терминов
            tech_terms = ["attention", "kv cache", "quantization", "layer", "embedding", "model", "inference"]
            if not any(term in clean_text.lower() for term in tech_terms):
                print("⚠️ Пропускаем: нет технического содержания")
                retrieved_texts.append("")
                continue
            
            # ✅ Сохраняем только первые 10K символов
            text = full_text
            retrieved_texts.append(text)
            print(f"✅ Текст извлечён ({len(text)} символов)")
        
        except Exception as e:
            print(f"❌ Ошибка при обработке {pdf_url}: {e}")
            retrieved_texts.append("")
    
    print(f"✅ Извлечено текстов: {len(retrieved_texts)}")
    return {"retrieved_texts": retrieved_texts}