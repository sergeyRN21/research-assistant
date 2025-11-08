# graph.py
from langgraph.graph import StateGraph, END
from state import GraphState

from nodes.retrieve_papers import retrieve_papers
from nodes.extract_text import extract_text
from nodes.generate_hypotheses import generate_hypotheses
from nodes.multi_query import multi_query
from nodes.retrieve_evidence import retrieve_evidence
from nodes.validate_evidence import validate_evidence
from nodes.synthesize_answer import synthesize_answer

def should_retry(state):
    """Условие повтора: если мало подтверждённых гипотез и retry_count = 0."""
    evidence_list = state.get("evidence", [])
    confirmed = [
        vc for item in evidence_list
        for vc in item.get("validated_chunks", [])
        if vc["judgment"]["confirmed"]
    ]
    if len(confirmed) < 2 and state.get("retry_count", 0) == 0:
        return "retrieve_papers"
    else:
        return "synthesize_answer"

workflow = StateGraph(GraphState)

# Добавляем узлы
workflow.add_node("retrieve_papers", retrieve_papers)
workflow.add_node("extract_text", extract_text)
workflow.add_node("generate_hypotheses", generate_hypotheses)
workflow.add_node("multi_query", multi_query)
workflow.add_node("retrieve_evidence", retrieve_evidence)
workflow.add_node("validate_evidence", validate_evidence)
workflow.add_node("synthesize_answer", synthesize_answer)

# Начинаем с retrieve_papers
workflow.set_entry_point("retrieve_papers")

# --- Обновлённые связи ---
# retrieve_papers -> extract_text
workflow.add_edge("retrieve_papers", "extract_text")
# extract_text -> generate_hypotheses
workflow.add_edge("extract_text", "generate_hypotheses")
# extract_text -> multi_query (теперь они могут работать параллельно после extract_text)
# multi_query -> retrieve_evidence (multi_query использует question)
workflow.add_edge("multi_query", "retrieve_evidence")
# generate_hypotheses -> retrieve_evidence (retrieve_evidence теперь использует и hypotheses)
# Мы не можем просто так соединить 2 узла к одному входу.
# Нужно убедиться, что retrieve_evidence запускается ПОСЛЕ и extract_text, и multi_query, и generate_hypotheses.
# LangGraph не поддерживает параллельное выполнение до одного узла напрямую так легко.
# Вместо этого, можно использовать Conditional Edge от generate_hypotheses или multi_query,
# но это усложнит логику. Проще всего, если generate_hypotheses и multi_query зависят от extract_text,
# а retrieve_evidence зависит от extract_text (для chunks) и от multi_query (для queries) и от generate_hypotheses (для hypotheses).
# Тогда можно сделать так:
# extract_text -> multi_query
# extract_text -> generate_hypotheses
# multi_query -> retrieve_evidence
# generate_hypotheses -> retrieve_evidence (но это не работает напрямую)
# Лучший способ - сделать один "join" узел, который дожидается и гипотез, и queries.
# Но для простоты, можно сделать так: extract_text -> generate_hypotheses -> multi_query -> retrieve_evidence
# Или: extract_text -> multi_query -> generate_hypotheses -> retrieve_evidence
# Или: extract_text -> generate_hypotheses (параллельно) -> retrieve_evidence
# extract_text -> multi_query (параллельно) -> retrieve_evidence
# Но LangGraph не очень удобен для параллелизма без специальных узлов или Conditional Edges.
# В текущей реализации retrieve_evidence ожидает, что и hypotheses, и queries, и chunks_with_metadata уже в state.
# Значит, мы должны убедиться, что они все вычислены до вызова retrieve_evidence.
# Мы можем заставить retrieve_evidence быть следующим после multi_query, но он также должен дождаться generate_hypotheses.
# В реальности, если multi_query и generate_hypotheses зависят только от extract_text, и не зависят друг от друга,
# то можно выполнить их параллельно (в коде LangGraph это означает, что они оба следуют за extract_text).
# Затем, нужен способ сказать, что retrieve_evidence запускается, когда *оба* предыдущих узла завершены.
# LangGraph не делает это автоматически. Он последовательный.
# В текущей архитектуре, если мы делаем:
# extract_text -> multi_query
# multi_query -> retrieve_evidence
# extract_text -> generate_hypotheses (НЕ СВЯЗАН с retrieve_evidence напрямую!)
# Тогда retrieve_evidence не будет знать о hypotheses!
# Нужно изменить порядок или использовать промежуточный узел или заставить retrieve_evidence ожидать state, в который добавлены и hypotheses, и queries.
# LangGraph неявно передаёт state между узлами.
# Значит, если мы сначала выполним generate_hypotheses, затем multi_query, затем retrieve_evidence, всё будет в state.
# Но нам нужно, чтобы retrieve_evidence ждал *и* гипотез, *и* queries.
# Простой способ: выполнить generate_hypotheses -> multi_query -> retrieve_evidence
# Или multi_query -> generate_hypotheses -> retrieve_evidence
# Давайте выберем: extract_text -> generate_hypotheses -> multi_query -> retrieve_evidence
# Это логично: сначала генерируем гипотезы, затем запросы для поиска, затем ищем доказательства для гипотез с помощью запросов.

workflow.add_edge("extract_text", "generate_hypotheses")
workflow.add_edge("generate_hypotheses", "multi_query") # Теперь multi_query после генерации гипотез
workflow.add_edge("multi_query", "retrieve_evidence") # retrieve_evidence теперь после обоих
workflow.add_edge("retrieve_evidence", "validate_evidence")

# Условный переход
workflow.add_conditional_edges(
    "validate_evidence",
    should_retry,
    {
        "retrieve_papers": "retrieve_papers",
        "synthesize_answer": "synthesize_answer"
    }
)

workflow.add_edge("synthesize_answer", END)

# Компилируем
app = workflow.compile()