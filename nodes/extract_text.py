# nodes/extract_text.py
import requests
from pypdf import PdfReader
import io
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(state):
    """
    Узел 2: Извлекает текст из PDF и разбивает на чанки с метаданными.
    """
    print("📄 Узел: Извлечение текста из PDF + chunking с метаданными...")
    
    papers = state.get("papers", [])
    if not papers:
        print("⚠️ Нет статей для извлечения.")
        return {"retrieved_texts": []}
    
    all_chunks_with_metadata = []
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
            
            # 🔍 Проверяем, что это реальная статья
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            structure_keywords = ["abstract", "introduction", "method", "experiment", "results", "conclusion"]
            if not any(kw in clean_text.lower()[:1000] for kw in structure_keywords):
                print("⚠️ Пропускаем: нет структуры научной статьи")
                continue

            tech_terms = ["attention", "kv cache", "quantization", "layer", "embedding", "model", "inference"]
            if not any(term in clean_text.lower() for term in tech_terms):
                print("⚠️ Пропускаем: нет технического содержания")
                continue

            # 🔥 Разбиваем на чанки с метаданными
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " ", ""],
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_text(full_text)

            # Добавляем метаданные к каждому чанку
            for chunk in chunks:
                metadata = {
                    "source_title": paper["title"],
                    "contains_method": any(kw in chunk.lower() for kw in ["method", "algorithm", "approach"]),
                    "contains_results": any(kw in chunk.lower() for kw in ["result", "accuracy", "throughput", "memory", "table", "figure"]),
                    "contains_experiment": any(kw in chunk.lower() for kw in ["experiment", "benchmark", "evaluation", "dataset"]),
                    "contains_figures": "figure" in chunk.lower() or "table" in chunk.lower(),
                    "page_estimate": len(full_text[:full_text.find(chunk)]) // 2000  # грубая оценка страницы
                }
                all_chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": metadata
                })

            print(f"✅ Разбито на {len(chunks)} чанков")
        
        except Exception as e:
            print(f"❌ Ошибка при обработке {pdf_url}: {e}")
    
    print(f"✅ Всего чанков с метаданными: {len(all_chunks_with_metadata)}")
    return {"chunks_with_metadata": all_chunks_with_metadata}