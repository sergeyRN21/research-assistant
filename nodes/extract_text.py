# nodes/extract_text.py
import requests
from pypdf import PdfReader
import io
import re

def extract_text(state):
    """
    Узел 2: Извлекает текст из PDF статей.
    Скачивает PDF в память, проверяет, есть ли в нём реальный научный текст.
    
    Вход: state["papers"] (список с pdf_url)
    Выход: state["retrieved_texts"] (список строк текста)
    """
    print("📄 Узел: Извлечение текста из PDF...")
    
    papers = state.get("papers", [])
    if not papers:
        print("⚠️ Нет статей для извлечения текста.")
        return {"retrieved_texts": []}
    
    retrieved_texts = []
    for i, paper in enumerate(papers):
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            print(f"⚠️ У статьи '{paper.get('title', 'Unknown')}' нет PDF-ссылки — пропускаем.")
            continue

        print(f"📥 Скачиваем PDF [{i+1}/{len(papers)}]: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()

            pdf = PdfReader(io.BytesIO(response.content))
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            
            # 🔥 Проверяем, что текст — не только метаданные
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Ключевые индикаторы "мусорного" PDF
            low_quality_indicators = [
                "IEEE" in clean_text[:500] and "grant" in clean_text[:800],
                "Personal use of this material is permitted" in clean_text,
                "This research was supported by" in clean_text,
                "©" in clean_text[:200] and "All rights reserved" in clean_text,
                len(clean_text) < 1000,
                "Abstract" not in clean_text[:500],  # если нет Abstract — вероятно, не статья
            ]
            
            if any(low_quality_indicators):
                print("⚠️ Пропускаем PDF: только метаданные или слишком короткий")
                retrieved_texts.append("")
                continue
            
            # Проверяем, есть ли в тексте научные ключевые слова
            scientific_keywords = ["method", "model", "experiment", "attention", "layer", "network", "dataset", "result"]
            if not any(kw in clean_text.lower() for kw in scientific_keywords):
                print("⚠️ Пропускаем PDF: нет научного содержания")
                retrieved_texts.append("")
                continue
            
            # Ограничиваем длину текста
            text = full_text[:10_000]  # первые 10 000 символов
            retrieved_texts.append(text)
            print(f"✅ Текст извлечён ({len(text)} символов)")
        
        except Exception as e:
            print(f"❌ Ошибка при обработке {pdf_url}: {e}")
            retrieved_texts.append("")  # добавим пустую строку, чтобы сохранить порядок
    
    print(f"✅ Извлечено текстов: {len(retrieved_texts)}")
    
    return {"retrieved_texts": retrieved_texts}