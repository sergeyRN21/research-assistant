# nodes/extract_text.py
import requests
from pypdf import PdfReader  # 🔥 Заменили PyPDF2 на pypdf
import io

def extract_text(state):
    """
    Узел 2: Извлекает текст из PDF статей.
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
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            
            # 🔥 Ограничим длину текста (чтобы не перегружать LLM)
            text = text[:10_000]  # первые 10 000 символов
            retrieved_texts.append(text)
            print(f"✅ Текст извлечён ({len(text)} символов)")
        
        except Exception as e:
            print(f"❌ Ошибка при обработке {pdf_url}: {e}")
            retrieved_texts.append("")  # добавим пустую строку, чтобы сохранить порядок
    
    print(f"✅ Извлечено текстов: {len(retrieved_texts)}")
    
    return {"retrieved_texts": retrieved_texts}