# state.py
from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    question: str
    papers: List[Dict[str, Any]]
    chunks_with_metadata: List[Dict[str, Any]]
    hypotheses: List[str]
    evidence: List[Dict[str, Any]]
    final_answer: str
    retry_count: int
    error: str