
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder


class GraphQuery:

    def __init__(self, graph_builder: KnowledgeGraphBuilder):
        self.graph = graph_builder.graph

    def search_entity(self, name: str):

        name = name.casefold()

        matches = [
            {
                "name": node,
                **attributes
            }
            for node, attributes in self.graph.nodes(data=True)
            if name in str(node).casefold()
        ]

        return matches

    def get_connected_entities(self, entity_name: str):

        if entity_name not in self.graph:
            return []

        results = []

        for source, target, attributes in self.graph.edges(
            entity_name, data=True
        ):
            results.append({
                "source": source,
                "target": target,
                **attributes
            })

        for source, target, attributes in self.graph.in_edges(
            entity_name, data=True
        ):
            results.append({
                "source": source,
                "target": target,
                **attributes
            })

        return results

    def get_customer_policies(self, customer_name: str):

        if customer_name not in self.graph:
            return []

        policies = []

        for _, target, attributes in self.graph.out_edges(
            customer_name, data=True
        ):
            if attributes.get("relation") == "OWNS":
                policies.append(target)

        return policies

    def get_policy_details(self, policy_name: str):

        if policy_name not in self.graph:
            return {}

        details = {
            "policy": policy_name,
            "customer": None,
            "insurer": None,
            "premium": None,
            "coverage": None,
            "relationships": []
        }

        for source, target, attributes in self.graph.out_edges(
            policy_name, data=True
        ):

            relation = attributes.get("relation")

            details["relationships"].append({
                "source": source,
                "target": target,
                **attributes
            })

            if relation == "ISSUED_BY":
                details["insurer"] = target

            elif relation == "HAS_PREMIUM":
                details["premium"] = target

            elif relation == "HAS_COVERAGE":
                details["coverage"] = target

        # Find the customer who owns this policy.
        for source, target, attributes in self.graph.in_edges(
            policy_name, data=True
        ):

            if attributes.get("relation") == "OWNS":
                details["customer"] = source

        return details

    def get_all_policies(self):
        """
        Return all policy entities stored in the graph.
        """

        return [
            node
            for node, attributes in self.graph.nodes(data=True)
            if (
                attributes.get("entity_type") == "Policy"
                or str(node).upper().startswith("POL-")
            )
        ]

    def get_all_policy_details(self):
        """
        Return structured details for every policy.
        """

        results = []

        for policy in self.get_all_policies():

            details = self.get_policy_details(policy)

            if details:
                results.append(details)

        return results

    def compare_policies(self, policy_names: list[str]):
        """
        Compare the requested policies.
        """

        results = []

        for policy_name in policy_names:

            details = self.get_policy_details(policy_name)

            if details:
                results.append(details)

        return results

    def rank_policies(self, field: str = "coverage", descending=True):
        """
        Rank policies by coverage or premium.

        Values are extracted from numeric portions of graph entities.
        """

        if field not in {"coverage", "premium"}:
            raise ValueError(
                "Field must be either 'coverage' or 'premium'."
            )

        import re

        ranked = []

        for details in self.get_all_policy_details():

            value = details.get(field)

            if value is None:
                continue

            numbers = re.findall(
                r"\d+(?:,\d+)*(?:\.\d+)?",
                str(value)
            )

            if not numbers:
                continue

            numeric_value = float(numbers[0].replace(",", ""))

            ranked.append({
                **details,
                "numeric_value": numeric_value
            })

        return sorted(
            ranked,
            key=lambda item: item["numeric_value"],
            reverse=descending
        )
