#state.py
from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """
    Состояние workflow - единое пространство данных между этапами.
    Каждый узел читает и дополняет этот словарь.
    """
    question: str                            # Вопрос пользователя
    papers: List[Dict[str, Any]]             # Найденные статьи из arXiv
    retrieved_texts: List[str]               # Извлечённый текст из PDF
    hypotheses: List[str]                    # Сгенерированные гипотезы
    evidence: List[Dict[str, Any]]           # Доказательства для каждой гипотезы
    final_answer: str                        # Итоговый ответ
    retry_count: int                         # Счётчик повторных попыток
    error: str                               # Ошибка, если возникла