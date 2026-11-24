
import re
import torch

from src.retrieval.vector_search import VectorSearch
from src.knowledge_graph.graph_query import GraphQuery


class RAGAgent:

    def __init__(
        self,
        vector_search: VectorSearch,
        graph_query: GraphQuery
    ):
        self.vector_search = vector_search
        self.graph_query = graph_query

    # --------------------------------------------------
    # 1. DETECT MULTI-POLICY QUESTIONS
    # --------------------------------------------------

    def is_multi_policy_question(self, question: str) -> bool:
        question_lower = question.casefold()

        aggregate_keywords = [
            "compare",
            "comparison",
            "highest",
            "lowest",
            "maximum",
            "minimum",
            "rank",
            "ranking",
            "all policies",
            "every policy",
            "across policies",
            "which policy has",
            "which customer has",
            "list policies",
            "list customers",
            "list all customers",
            "all customers",
            "customer list",
            "list insurers",
            "list all insurers",
            "all insurers",
            "list all insurance providers",
            "all insurance providers",
            "insurance providers"
        ]

        return any(
            keyword in question_lower
            for keyword in aggregate_keywords
        )

    # --------------------------------------------------
    # 2. RETRIEVAL
    # --------------------------------------------------

    def retrieve(self, question: str):

        documents = self.vector_search.search(
            question,
            top_k=50
        )

        question_lower = question.casefold()
        graph_entities = list(self.graph_query.graph.nodes)

        policy_match = re.search(
            r"\bPOL-\d+\b",
            question,
            re.IGNORECASE
        )

        if policy_match:

            requested_policy = policy_match.group(0).upper()

            known_policies = {
                str(entity).upper()
                for entity in graph_entities
                if re.fullmatch(
                    r"POL-\d+",
                    str(entity),
                    re.IGNORECASE
                )
            }

            if requested_policy not in known_policies:
                return {
                    "documents": [],
                    "graph_results": [],
                    "mentioned_entities": [],
                    "clarification_required": False
                }

        mentioned_entities = [
            entity
            for entity in graph_entities
            if str(entity).casefold() in question_lower
        ]

        if mentioned_entities:

            selected_documents = [
                document
                for document in documents
                if any(
                    str(entity).casefold()
                    in document["content"].casefold()
                    for entity in mentioned_entities
                )
            ]

        else:
            selected_documents = documents

        source_names = {
            item["metadata"].get("source")
            for item in selected_documents
        }

        multi_policy_question = self.is_multi_policy_question(question)

        clarification_required = (
            not mentioned_entities
            and len(source_names) > 1
            and not multi_policy_question
        )

        graph_results = {}

        for document in selected_documents:

            content = document["content"].casefold()

            for entity in graph_entities:

                if str(entity).casefold() not in content:
                    continue

                relationships = (
                    self.graph_query.get_connected_entities(entity)
                )

                for relationship in relationships:

                    source = str(relationship["source"])
                    target = str(relationship["target"])

                    if (
                        source.casefold() not in content
                        or target.casefold() not in content
                    ):
                        continue

                    key = (
                        source,
                        relationship["relation"],
                        target
                    )

                    graph_results[key] = relationship

        return {
            "documents": selected_documents,
            "graph_results": list(graph_results.values()),
            "mentioned_entities": mentioned_entities,
            "clarification_required": clarification_required
        }

    # --------------------------------------------------
    # 3. FIND RELEVANT RELATIONSHIP
    # --------------------------------------------------

    def get_relevant_relationship(self, question, relationships):

        question_lower = question.casefold()

        relation_mapping = [
            (
                [
                    "coverage amount",
                    "coverage limit",
                    "sum insured",
                    "insured amount",
                    "coverage"
                ],
                "HAS_COVERAGE"
            ),
            (
                [
                    "annual premium",
                    "premium amount",
                    "premium"
                ],
                "HAS_PREMIUM"
            ),
            (
                [
                    "insurance company",
                    "insurance provider",
                    "insurer",
                    "provider"
                ],
                "ISSUED_BY"
            ),
            (
                [
                    "policyholder",
                    "insured person",
                    "customer name",
                    "customer"
                ],
                "OWNS"
            )
        ]

        for keywords, relation_type in relation_mapping:

            if any(
                keyword in question_lower
                for keyword in keywords
            ):

                for relationship in relationships:

                    if relationship["relation"] == relation_type:
                        return relationship

        return None

    # --------------------------------------------------
    # 4. GROUNDED FALLBACK
    # --------------------------------------------------

    def grounded_fallback(self, question, context):

        relationship = self.get_relevant_relationship(
            question,
            context["graph_results"]
        )

        if relationship:

            if relationship["relation"] == "OWNS":
                return f"The customer is {relationship['source']}."

            if relationship["relation"] == "ISSUED_BY":
                return f"The insurer is {relationship['target']}."

            if relationship["relation"] == "HAS_PREMIUM":
                return (
                    f"The annual premium is {relationship['target']}."
                )

            if relationship["relation"] == "HAS_COVERAGE":
                return (
                    f"The coverage amount is {relationship['target']}."
                )

        return (
            "I could not find a reliable answer in the selected "
            "insurance document."
        )

    # --------------------------------------------------
    # 5. VERIFIED SINGLE-POLICY ANSWER
    # --------------------------------------------------

    def get_verified_graph_answer(self, question, context):

        documents = context["documents"]
        relationships = context["graph_results"]

        if not documents or not relationships:
            return None

        source_names = {
            document["metadata"].get("source")
            for document in documents
        }

        if len(source_names) != 1:
            return None

        document_context = "\n".join(
            document["content"]
            for document in documents
        )

        policy_ids = {
            policy_id.upper()
            for policy_id in re.findall(
                r"\bPOL-\d+\b",
                document_context,
                re.IGNORECASE
            )
        }

        if len(policy_ids) != 1:
            return None

        relationship = self.get_relevant_relationship(
            question,
            relationships
        )

        if not relationship:
            return None

        relation_type = relationship["relation"]
        source = str(relationship["source"])
        target = str(relationship["target"])

        if relation_type == "OWNS":
            relationship_policy = target
            expected_value = source

        elif relation_type in (
            "ISSUED_BY",
            "HAS_PREMIUM",
            "HAS_COVERAGE"
        ):
            relationship_policy = source
            expected_value = target

        else:
            return None

        if relationship_policy.upper() not in policy_ids:
            return None

        if (
            source.casefold() not in document_context.casefold()
            or target.casefold() not in document_context.casefold()
        ):
            return None

        if expected_value.casefold() not in document_context.casefold():
            return None

        if relation_type == "OWNS":
            return f"The customer is {source}."

        if relation_type == "ISSUED_BY":
            return f"The insurer is {target}."

        if relation_type == "HAS_PREMIUM":
            return f"The annual premium is {target}."

        if relation_type == "HAS_COVERAGE":
            return f"The coverage amount is {target}."

        return None

    # --------------------------------------------------
    # 6. VERIFIED POLICY SUMMARY
    # --------------------------------------------------

    def get_verified_policy_summary(self, context):

        documents = context["documents"]
        relationships = context["graph_results"]

        if not documents or not relationships:
            return None

        source_names = {
            item["metadata"].get("source")
            for item in documents
        }

        if len(source_names) != 1:
            return None

        document_text = "\n".join(
            item["content"]
            for item in documents
        )

        policy_ids = {
            policy.upper()
            for policy in re.findall(
                r"\bPOL-\d+\b",
                document_text,
                re.IGNORECASE
            )
        }

        if len(policy_ids) != 1:
            return None

        policy_id = next(iter(policy_ids))

        policy_relationships = [
            item
            for item in relationships
            if (
                str(item["source"]).upper() == policy_id
                or str(item["target"]).upper() == policy_id
            )
        ]

        required_relations = {
            "OWNS",
            "ISSUED_BY",
            "HAS_PREMIUM",
            "HAS_COVERAGE"
        }

        relation_types = {
            item["relation"]
            for item in policy_relationships
        }

        if not required_relations.issubset(relation_types):
            return None

        facts = {}

        for item in policy_relationships:

            if item["relation"] == "OWNS":
                facts["customer"] = str(item["source"])

            elif item["relation"] == "ISSUED_BY":
                facts["insurer"] = str(item["target"])

            elif item["relation"] == "HAS_PREMIUM":
                facts["premium"] = str(item["target"])

            elif item["relation"] == "HAS_COVERAGE":
                facts["coverage"] = str(item["target"])

        if len(facts) != 4:
            return None

        for value in facts.values():

            if value.casefold() not in document_text.casefold():
                return None

        return (
            f"Policy {policy_id} belongs to {facts['customer']}. "
            f"It is issued by {facts['insurer']}. "
            f"The annual premium is {facts['premium']}, "
            f"and the coverage amount is {facts['coverage']}."
        )

    # --------------------------------------------------
    # 7. EXTRACT VERIFIED POLICY DATA FROM ALL DOCUMENTS
    # --------------------------------------------------

    def get_verified_policy_records(self, context):

        records = []
        seen_policies = set()

        grouped_documents = {}

        for document in context["documents"]:

            source = document["metadata"].get("source")

            grouped_documents.setdefault(source, []).append(
                document["content"]
            )

        for source, contents in grouped_documents.items():

            document_text = "\n".join(contents)

            policy_ids = {
                policy.upper()
                for policy in re.findall(
                    r"\bPOL-\d+\b",
                    document_text,
                    re.IGNORECASE
                )
            }

            # One policy per source document is required.
            if len(policy_ids) != 1:
                continue

            policy_id = next(iter(policy_ids))

            if policy_id in seen_policies:
                continue

            policy_relationships = [
                item
                for item in context["graph_results"]
                if (
                    str(item["source"]).upper() == policy_id
                    or str(item["target"]).upper() == policy_id
                )
            ]

            facts = {}

            for relationship in policy_relationships:

                relation = relationship["relation"]
                graph_source = str(relationship["source"])
                graph_target = str(relationship["target"])

                if (
                    graph_source.casefold() not in document_text.casefold()
                    or graph_target.casefold() not in document_text.casefold()
                ):
                    continue

                if relation == "OWNS" and graph_target.upper() == policy_id:
                    facts["customer"] = graph_source

                elif (
                    relation == "ISSUED_BY"
                    and graph_source.upper() == policy_id
                ):
                    facts["insurer"] = graph_target

                elif (
                    relation == "HAS_PREMIUM"
                    and graph_source.upper() == policy_id
                ):
                    facts["premium"] = graph_target

                elif (
                    relation == "HAS_COVERAGE"
                    and graph_source.upper() == policy_id
                ):
                    facts["coverage"] = graph_target

            if len(facts) != 4:
                continue

            if any(
                value.casefold() not in document_text.casefold()
                for value in facts.values()
            ):
                continue

            records.append({
                "policy": policy_id,
                "customer": facts["customer"],
                "insurer": facts["insurer"],
                "premium": facts["premium"],
                "coverage": facts["coverage"],
                "source": source
            })

            seen_policies.add(policy_id)

        return records

    # --------------------------------------------------
    # 8. MULTI-POLICY INTELLIGENCE
    # --------------------------------------------------

    def get_verified_multi_policy_answer(self, question, context):

        records = self.get_verified_policy_records(context)

        if not records:
            return None

        question_lower = question.casefold()

        premium_requested = any(
            word in question_lower
            for word in ["premium", "cost"]
        )

        coverage_requested = any(
            word in question_lower
            for word in ["coverage", "sum insured", "insured amount"]
        )

        ranking_requested = any(
            word in question_lower
            for word in ["rank", "ranking", "highest", "lowest",
                         "maximum", "minimum", "most", "least"]
        )

        compare_requested = any(
            word in question_lower
            for word in ["compare", "comparison", "between"]
        )

        policy_list_requested = any(
            phrase in question_lower
            for phrase in [
                "all policies",
                "every policy",
                "list policies",
            ]
        )

        customer_list_requested = any(
            phrase in question_lower
            for phrase in [
                "list customers",
                "list all customers",
                "all customers",
                "customer list",
            ]
        )

        insurer_list_requested = any(
            phrase in question_lower
            for phrase in [
                "list insurers",
                "list all insurers",
                "all insurers",
                "list all insurance providers",
                "all insurance providers",
                "insurance providers",
            ]
        )

        if premium_requested or coverage_requested:

            if premium_requested:
                field = "premium"
                label = "Premium"
            else:
                field = "coverage"
                label = "Coverage"

            def numeric_value(record):
                numbers = re.findall(
                    r"\d+(?:,\d+)*(?:\.\d+)?",
                    record[field]
                )

                if not numbers:
                    return None

                return float(numbers[0].replace(",", ""))

            valid_records = [
                record
                for record in records
                if numeric_value(record) is not None
            ]

            if not valid_records:
                return None

            if ranking_requested or compare_requested or policy_list_requested:

                lowest_requested = any(
                    word in question_lower
                    for word in ["lowest", "minimum", "least"]
                )

                ranked = sorted(
                    valid_records,
                    key=numeric_value,
                    reverse=not lowest_requested
                )

                lines = [
                    f"Policies ranked by {label.lower()}:"
                ]

                for index, record in enumerate(ranked, start=1):

                    lines.append(
                        f"{index}. {record['policy']} - "
                        f"{record['customer']} - "
                        f"{label}: {record[field]}"
                    )

                return "\n".join(lines)

            highest_requested = any(
                word in question_lower
                for word in ["highest", "maximum", "most"]
            )

            lowest_requested = any(
                word in question_lower
                for word in ["lowest", "minimum", "least"]
            )

            if highest_requested or lowest_requested:

                ranked = sorted(
                    valid_records,
                    key=numeric_value,
                    reverse=highest_requested
                )

                winner = ranked[0]

                return (
                    f"{winner['customer']} has the "
                    f"{'highest' if highest_requested else 'lowest'} "
                    f"{label.lower()} among the selected policies. "
                    f"Policy {winner['policy']} has {label.lower()} "
                    f"{winner[field]}."
                )

        if policy_list_requested or compare_requested:

            lines = ["Verified policy details:"]

            for record in records:

                lines.append(
                    f"{record['policy']} | "
                    f"Customer: {record['customer']} | "
                    f"Insurer: {record['insurer']} | "
                    f"Premium: {record['premium']} | "
                    f"Coverage: {record['coverage']}"
                )

            return "\n".join(lines)

        if customer_list_requested or (
            "customer" in question_lower and "list" in question_lower
        ):
            customers = sorted({
                record["customer"]
                for record in records
            })

            return "Customers: " + ", ".join(customers)

        if insurer_list_requested or any(
            word in question_lower
            for word in ["insurer", "provider", "insurance company"]
        ):
            insurers = sorted({
                record["insurer"]
                for record in records
            })

            return "Insurance providers: " + ", ".join(insurers)

        return None

    # --------------------------------------------------
    # 9. ANSWER FROM RETRIEVED CONTEXT
    # --------------------------------------------------

    def answer_from_context(
        self,
        question: str,
        context: dict,
        model,
        tokenizer
    ):

        if context["clarification_required"]:

            return {
                "question": question,
                "answer": (
                    "Multiple insurance policies match this question. "
                    "Please specify the customer name or policy number."
                ),
                "answer_source": "clarification",
                "sources": [
                    item["metadata"]
                    for item in context["documents"]
                ],
                "graph_evidence": context["graph_results"]
            }

        if not context["documents"]:

            return {
                "question": question,
                "answer": (
                    "I could not find a matching insurance document."
                ),
                "answer_source": "no_matching_document",
                "sources": [],
                "graph_evidence": []
            }

        sources = [
            item["metadata"]
            for item in context["documents"]
        ]

        direct_answer = self.get_verified_graph_answer(
            question,
            context
        )

        if direct_answer:

            return {
                "question": question,
                "answer": direct_answer,
                "answer_source": "knowledge_graph_direct",
                "sources": sources,
                "graph_evidence": context["graph_results"]
            }

        multi_policy_answer = self.get_verified_multi_policy_answer(
            question,
            context
        )

        if multi_policy_answer:

            return {
                "question": question,
                "answer": multi_policy_answer,
                "answer_source": "knowledge_graph_multi_policy",
                "sources": sources,
                "graph_evidence": context["graph_results"]
            }

        question_lower = question.casefold()

        summary_requested = (
            "summarize" in question_lower
            or "summary" in question_lower
        )

        if summary_requested:

            policy_summary = self.get_verified_policy_summary(
                context
            )

            if policy_summary:

                return {
                    "question": question,
                    "answer": policy_summary,
                    "answer_source": "knowledge_graph_summary",
                    "sources": sources,
                    "graph_evidence": context["graph_results"]
                }

        # Load the local LLM only when graph answering is insufficient.
        if hasattr(model, "get_model_and_tokenizer"):
            model, tokenizer = model.get_model_and_tokenizer()

        if model is None or tokenizer is None:
            raise ValueError(
                "A model and tokenizer are required for LLM answering."
            )

        document_context = "\n".join(
            item["content"]
            for item in context["documents"]
        )

        graph_context = "\n".join(
            f"{item['source']} --{item['relation']}--> "
            f"{item['target']}"
            for item in context["graph_results"]
        )

        prompt = f"""
You are an insurance document question-answering assistant.

Answer using only the supplied evidence.
Copy names and monetary values exactly.
Do not mix information from different policies.
If information is missing, say it is unavailable.

DOCUMENT EVIDENCE:
{document_context}

KNOWLEDGE GRAPH EVIDENCE:
{graph_context}

QUESTION:
{question}

Give a short, direct answer.
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        device = next(model.parameters()).device

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.inference_mode():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=40,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_ids.shape[-1]:]

        llm_answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        relevant_relationship = self.get_relevant_relationship(
            question,
            context["graph_results"]
        )

        if relevant_relationship:

            if relevant_relationship["relation"] == "OWNS":
                expected_value = relevant_relationship["source"]
            else:
                expected_value = relevant_relationship["target"]

            if expected_value.casefold() in llm_answer.casefold():

                final_answer = llm_answer
                answer_source = "local_llm"

            else:

                final_answer = self.grounded_fallback(
                    question,
                    context
                )

                answer_source = "knowledge_graph_fallback"

        elif llm_answer:

            final_answer = llm_answer
            answer_source = "local_llm"

        else:

            final_answer = self.grounded_fallback(
                question,
                context
            )

            answer_source = "knowledge_graph_fallback"

        return {
            "question": question,
            "answer": final_answer,
            "answer_source": answer_source,
            "sources": sources,
            "graph_evidence": context["graph_results"]
        }

    # --------------------------------------------------
    # 10. MAIN ANSWER METHOD
    # --------------------------------------------------

    def answer(self, question: str, model, tokenizer):

        context = self.retrieve(question)

        return self.answer_from_context(
            question,
            context,
            model,
            tokenizer
        )
