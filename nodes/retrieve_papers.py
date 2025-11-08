# nodes/retrieve_papers.py
import arxiv
import re

def retrieve_papers(state):
    """
    Узел 1: Находит статьи через arXiv API.
    Гарантирует, что pdf_url ведёт на актуальную версию (без v1/v2).
    
    Вход: state["question"]
    Выход: state["papers"]
    """
    print("🔍 Узел: Поиск статей через arXiv API...")
    
    question = state["question"]
    
    search = arxiv.Search(
        query=question,
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    try:
        for result in search.results():
            # 🔧 Формируем чистый URL без версии
            base_id = result.entry_id.split("/")[-1].split("v")[0]
            # --- Исправленный URL ---
            pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf" # <- Убраны пробелы
            
            # Проверяем формат (теперь корректный)
            if not re.fullmatch(r"https://arxiv\.org/pdf/\d+\.\d+\.pdf", pdf_url):
                 print(f"⚠️ Некорректный URL: {pdf_url}")
                 continue
                
            papers.append({
                "entry_id": result.entry_id,
                "title": result.title,
                "summary": result.summary,
                "pdf_url": pdf_url,
                "published": result.published.strftime("%Y-%m-%d"),
                "authors": [author.name for author in result.authors]
            })
    except Exception as e:
        print(f"❌ Ошибка при поиске статей: {e}")
        return {"papers": [], "error": str(e)}
    
    print(f"✅ Найдено {len(papers)} статей с PDF.")
    return {"papers": papers}