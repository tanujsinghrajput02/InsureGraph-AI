from typing import TypedDict, Any


class InsuranceAgentState(TypedDict, total=False):
    question: str
    retrieved_context: dict[str, Any]
    answer: str
    answer_source: str
    sources: list[dict]
    graph_evidence: list[dict]
    status: str

    answer_valid: bool
    validation_error: str
    retry_count: int

    retrieval_seconds: float
    answer_generation_seconds: float
    vector_search_calls: int
    recovery_attempts: int