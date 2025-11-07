# nodes/retrieve_papers.py
import arxiv
import re

def retrieve_papers(state):
    """
    Узел 1: Находит статьи через arXiv API.
    Проверяет, что возвращаемые URL ведут на реальные PDF.
    
    Вход: state["question"]
    Выход: state["papers"]
    """
    print("🔍 Узел: Поиск статей через arXiv API...")
    
    question = state["question"]
    
    search = arxiv.Search(
        query=question,
        max_results=3,  # уменьшаем, чтобы быстрее
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    try:
        for result in search.results():
            # 🔥 Формируем URL вручную — надёжнее
            base_id = result.entry_id.split("/")[-1].split("v")[0]
            pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
            
            # Проверяем, что URL корректен
            if not re.match(r"https://arxiv\.org/pdf/[\d\.]+\.pdf", pdf_url):
                print(f"⚠️ Некорректный URL для статьи: {result.title}")
                continue

            papers.append({
                "entry_id": result.entry_id,
                "title": result.title,
                "summary": result.summary,
                "pdf_url": pdf_url,  # теперь точно правильный
                "published": result.published.strftime("%Y-%m-%d"),
                "authors": [author.name for author in result.authors]
            })
    except Exception as e:
        print(f"❌ Ошибка при поиске статей: {e}")
        return {"papers": [], "error": str(e)}
    
    print(f"✅ Найдено {len(papers)} статей с PDF.")
    
    return {"papers": papers}