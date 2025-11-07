# nodes/retrieve_papers.py
import arxiv
from datetime import datetime


def retrieve_papers(state):
    """
    Узел 1: Находит статьи через arXiv API.
    Пока — заглушка.
    """
    print("🔍 Узел: Поиск статей через arXiv API")

    question = state["question"]

    # Ищем по заголовку и аннотации
    search = arxiv.Search(
        query=question,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    try:
        for result in search.results():
            papers.append({
                "entry_id" : result.entry_id,
                "title" : result.title,
                "pdf_url": result.pdf_url,
                "summary" : result.summary,
                "published" : result.published,
                "authors" : [author.name for author in result.authors]
            })
    except Exception as e:
        print(f"Ошибка при поиске статей: {e}")
        return {"papers": [], "error":str(e)}
    
    print(f"Найдено {len(papers)} статей.")

    return {"papers": papers}