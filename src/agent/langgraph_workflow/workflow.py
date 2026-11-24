from langgraph.graph import StateGraph, START, END

from src.agent.langgraph_workflow.state import InsuranceAgentState

from src.agent.langgraph_workflow.nodes import (
    create_retrieval_node,
    create_answer_node,
    create_validation_node,
    create_recovery_node,
    create_failure_node,
    route_after_retrieval,
    route_after_validation
)


def build_insurance_workflow(agent, model, tokenizer):

    workflow = StateGraph(InsuranceAgentState)

    # Register nodes
    workflow.add_node(
        "retrieve",
        create_retrieval_node(agent)
    )

    workflow.add_node(
        "generate_answer",
        create_answer_node(agent, model, tokenizer)
    )

    workflow.add_node(
        "validate_answer",
        create_validation_node(agent)
    )

    workflow.add_node(
        "recover_answer",
        create_recovery_node(agent)
    )

    workflow.add_node(
        "validation_failure",
        create_failure_node()
    )

    # Initial connection
    workflow.add_edge(START, "retrieve")

    # Retrieval routing
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "end": END
        }
    )

    # Generate then validate
    workflow.add_edge(
        "generate_answer",
        "validate_answer"
    )

    # Validation routing
    workflow.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "end": END,
            "recover": "recover_answer",
            "fail": "validation_failure"
        }
    )

    # Recovery gets validated again
    workflow.add_edge(
        "recover_answer",
        "validate_answer"
    )

    workflow.add_edge(
        "validation_failure",
        END
    )

    return workflow.compile()