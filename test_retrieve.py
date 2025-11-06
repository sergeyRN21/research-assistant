# test_retrieve.py
from nodes.retrieve_papers import retrieve_papers

state = {
    "question": "KV cache reduction",
    "papers": []
}

result = retrieve_papers(state)

print("\n🔍 Результаты поиска:")
for i, paper in enumerate(result["papers"][:3], 1):  # первые 3
    print(f"{i}. {paper['title']}")
    print(f"   URL: {paper['pdf_url']}")
    print(f"   Дата: {paper['published']}")
    print()