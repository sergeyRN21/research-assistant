# graph.py
from langgraph.graph import StateGraph, END
from state import GraphState

# Импортируем заглушки узлов
from nodes.retrieve_papers import retrieve_papers
from nodes.extract_text import extract_text
from nodes.generate_hypotheses import generate_hypotheses
from nodes.retrieve_evidence import retrieve_evidence
from nodes.validate_evidence import validate_evidence
from nodes.synthesize_answer import synthesize_answer

# === Определяем условный переход ===
def should_retry(state):
    """
    Условный переход после валидации.
    Если мало доказательств и ещё есть попытки — повторить поиск.
    """
    evidence = state.get("evidence", [])
    confirmed = [e for e in evidence if e.get("judgment", {}).get("confirmed", False)]
    
    if len(confirmed) < 2 and state.get("retry_count", 0) < 1:
        print("🔄 Мало подтверждённых гипотез — повторный поиск")
        return "retrieve_papers"
    else:
        print("✅ Достаточно доказательств — завершаем")
        return "synthesize_answer"

# === Создаём граф ===
workflow = StateGraph(GraphState)

# Добавляем узлы
workflow.add_node("retrieve_papers", retrieve_papers)
workflow.add_node("extract_text", extract_text)
workflow.add_node("generate_hypotheses", generate_hypotheses)
workflow.add_node("retrieve_evidence", retrieve_evidence)
workflow.add_node("validate_evidence", validate_evidence)
workflow.add_node("synthesize_answer", synthesize_answer)

# === Задаём переходы ===
workflow.set_entry_point("retrieve_papers")

workflow.add_edge("retrieve_papers", "extract_text")
workflow.add_edge("extract_text", "generate_hypotheses")
workflow.add_edge("generate_hypotheses", "retrieve_evidence")
workflow.add_edge("retrieve_evidence", "validate_evidence")

# # Условный переход: повтор или завершение
# workflow.add_conditional_edges(
#     "validate_evidence",
#     should_retry,
#     {
#         "retrieve_papers": "retrieve_papers",
#         "synthesize_answer": "synthesize_answer"
#     }
# )

workflow.add_edge("validate_evidence", "synthesize_answer")
workflow.add_edge("synthesize_answer", END)

# === Компилируем граф ===
app = workflow.compile()