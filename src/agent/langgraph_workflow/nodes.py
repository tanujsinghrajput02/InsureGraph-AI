
import re
import time

from src.agent.rag_agent import RAGAgent


STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with",
    "that", "this", "from", "has", "have", "had", "its",
    "their", "they", "them", "his", "her", "you", "your",
    "who", "what", "when", "where", "which", "while",
    "will", "would", "could", "should", "can", "may",
    "not", "but", "all", "any", "one", "two", "into",
    "about", "under", "over", "per", "is", "in", "on",
    "of", "to", "a", "an", "by", "as", "it", "be",
    "or", "if", "at", "we", "he", "she"
}


def validate_summary_grounding(answer, documents):
    """
    Basic lexical grounding check for LLM-generated summaries.
    """

    if not answer or not answer.strip() or not documents:
        return False

    evidence = " ".join(
        document.get("content", "")
        for document in documents
    ).casefold()

    answer_lower = answer.casefold()

    answer_numbers = re.findall(
        r"\b\d+(?:,\d+)*(?:\.\d+)?\b",
        answer
    )

    evidence_numbers = set(
        re.findall(
            r"\b\d+(?:,\d+)*(?:\.\d+)?\b",
            evidence
        )
    )

    if any(
        number not in evidence_numbers
        for number in answer_numbers
    ):
        return False

    answer_words = {
        word
        for word in re.findall(r"[a-zA-Z]{3,}", answer_lower)
        if word not in STOPWORDS
    }

    if not answer_words:
        return False

    grounded_words = {
        word
        for word in answer_words
        if word in evidence
    }

    grounding_ratio = len(grounded_words) / len(answer_words)

    return grounding_ratio >= 0.50


def create_retrieval_node(agent: RAGAgent):

    def retrieval_node(state: dict):

        question = state["question"]

        start_time = time.perf_counter()
        context = agent.retrieve(question)
        retrieval_seconds = time.perf_counter() - start_time

        metrics = {
            "retrieval_seconds": round(retrieval_seconds, 6),
            "vector_search_calls": 1,
            "answer_generation_seconds": 0.0,
            "recovery_attempts": 0
        }

        if context["clarification_required"]:

            return {
                "retrieved_context": context,
                "answer": (
                    "Multiple insurance policies match this question. "
                    "Please specify the customer name or policy number."
                ),
                "answer_source": "clarification",
                "sources": [
                    item["metadata"]
                    for item in context["documents"]
                ],
                "graph_evidence": context["graph_results"],
                "status": "complete",
                **metrics
            }

        if not context["documents"]:

            return {
                "retrieved_context": context,
                "answer": (
                    "I could not find a matching insurance document."
                ),
                "answer_source": "no_matching_document",
                "sources": [],
                "graph_evidence": [],
                "status": "complete",
                **metrics
            }

        return {
            "retrieved_context": context,
            "status": "ready",
            **metrics
        }

    return retrieval_node


def create_answer_node(agent, model, tokenizer):

    def answer_node(state: dict):

        start_time = time.perf_counter()

        response = agent.answer_from_context(
            state["question"],
            state["retrieved_context"],
            model,
            tokenizer
        )

        generation_seconds = time.perf_counter() - start_time

        print("\nDraft answer before validation:")
        print(response["answer"])

        return {
            "answer": response["answer"],
            "answer_source": response["answer_source"],
            "sources": response["sources"],
            "graph_evidence": response["graph_evidence"],
            "answer_generation_seconds": round(
                generation_seconds,
                6
            ),
            "retry_count": state.get("retry_count", 0)
        }

    return answer_node


def create_validation_node(agent: RAGAgent):

    def validation_node(state: dict):

        question = state["question"]
        answer = state.get("answer", "")
        context = state.get("retrieved_context", {})

        relationships = context.get("graph_results", [])
        answer_source = state.get("answer_source")

        sources_available = bool(state.get("sources"))

        # PATH 1: Verify multi-policy answers against all selected PDFs.
        if answer_source == "knowledge_graph_multi_policy":

            expected_answer = agent.get_verified_multi_policy_answer(
                question,
                context
            )

            multi_policy_valid = (
                expected_answer is not None
                and answer.strip().casefold()
                == expected_answer.strip().casefold()
            )

            answer_valid = (
                multi_policy_valid
                and sources_available
            )

            if answer_valid:
                error = ""
            elif not sources_available:
                error = "No source documents were attached."
            else:
                error = (
                    "The multi-policy answer does not match "
                    "the verified policy records."
                )

            return {
                "answer_valid": answer_valid,
                "validation_error": error
            }

        # PATH 2: Verify deterministic policy summaries.
        if answer_source == "knowledge_graph_summary":

            expected_summary = agent.get_verified_policy_summary(
                context
            )

            summary_valid = (
                expected_summary is not None
                and answer.strip().casefold()
                == expected_summary.strip().casefold()
            )

            answer_valid = (
                summary_valid
                and sources_available
            )

            if answer_valid:
                error = ""
            elif not sources_available:
                error = "No source document was attached."
            else:
                error = (
                    "The policy summary does not match the "
                    "verified knowledge-graph summary."
                )

            return {
                "answer_valid": answer_valid,
                "validation_error": error
            }

        # PATH 3: Verify direct graph answers.
        if answer_source == "knowledge_graph_direct":

            relevant_relationship = agent.get_relevant_relationship(
                question,
                relationships
            )

            if not relevant_relationship:

                return {
                    "answer_valid": False,
                    "validation_error": (
                        "No matching graph relationship was found."
                    )
                }

            if relevant_relationship["relation"] == "OWNS":
                expected_value = relevant_relationship["source"]
            else:
                expected_value = relevant_relationship["target"]

            answer_supported = (
                expected_value.casefold() in answer.casefold()
            )

            answer_valid = (
                answer_supported
                and sources_available
            )

            if answer_valid:
                error = ""
            elif not answer_supported:
                error = (
                    "The answer does not contain the expected "
                    "value from graph evidence."
                )
            else:
                error = "No source document was attached."

            return {
                "answer_valid": answer_valid,
                "validation_error": error
            }

        # PATH 4: Verify LLM-generated answers.
        if answer_source == "local_llm":

            summary_valid = validate_summary_grounding(
                answer,
                context.get("documents", [])
            )

            answer_valid = (
                summary_valid
                and sources_available
            )

            if answer_valid:
                error = ""
            elif not sources_available:
                error = "No source document was attached."
            else:
                error = (
                    "The generated answer could not be "
                    "sufficiently grounded in document evidence."
                )

            return {
                "answer_valid": answer_valid,
                "validation_error": error
            }

        return {
            "answer_valid": False,
            "validation_error": (
                "No matching verified answer type was found."
            )
        }

    return validation_node


def create_recovery_node(agent: RAGAgent):

    def recovery_node(state: dict):

        context = state.get("retrieved_context", {})

        fallback_answer = agent.grounded_fallback(
            state["question"],
            context
        )

        return {
            "answer": fallback_answer,
            "answer_source": "knowledge_graph_recovery",
            "retry_count": state.get("retry_count", 0) + 1,
            "recovery_attempts": (
                state.get("recovery_attempts", 0) + 1
            )
        }

    return recovery_node


def create_failure_node():

    def failure_node(state: dict):

        return {
            "answer": (
                "I could not verify a reliable answer from the "
                "available insurance document evidence."
            ),
            "answer_source": "validation_failed",
            "status": "failed"
        }

    return failure_node


def route_after_retrieval(state: dict):

    if state.get("status") == "complete":
        return "end"

    return "generate_answer"


def route_after_validation(state: dict):

    if state.get("answer_valid"):
        return "end"

    if state.get("retry_count", 0) < 1:
        return "recover"

    return "fail"
