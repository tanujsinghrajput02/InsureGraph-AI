
import re
from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    entity_type: str
    evidence: str


class Relationship(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


class EntityExtractor:

    FIELD_TYPES = {
        "policy": {
            "type": "Policy",
            "labels": [
                "policy number",
                "policy no",
                "policy id"
            ]
        },
        "customer": {
            "type": "Customer",
            "labels": [
                "customer name",
                "insured name",
                "policyholder"
            ]
        },
        "insurer": {
            "type": "Insurer",
            "labels": [
                "insurer",
                "insurance company"
            ]
        },
        "premium": {
            "type": "Premium",
            "labels": [
                "annual premium",
                "premium amount",
                "premium"
            ]
        },
        "coverage": {
            "type": "Coverage",
            "labels": [
                "coverage amount",
                "sum insured",
                "sum assured"
            ]
        }
    }

    def normalize_label(self, value):
        return re.sub(
            r"\s+",
            " ",
            value.strip().lower()
        ).rstrip(":")

    def extract_fields(self, text):
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        extracted = {}

        label_mapping = {}

        for key, config in self.FIELD_TYPES.items():
            for label in config["labels"]:
                label_mapping[label] = key

        for index, line in enumerate(lines):

            # Format 1: Customer Name: Rahul Sharma
            if ":" in line:
                label, value = line.split(":", 1)
                normalized_label = self.normalize_label(label)
                value = value.strip()

                if normalized_label in label_mapping and value:
                    key = label_mapping[normalized_label]

                    extracted[key] = {
                        "value": value,
                        "evidence": line
                    }

                    continue

            # Format 2:
            # Customer Name
            # Rahul Sharma
            normalized_label = self.normalize_label(line)

            if normalized_label in label_mapping:
                key = label_mapping[normalized_label]

                if index + 1 < len(lines):
                    value = lines[index + 1].strip()

                    if ":" not in value:
                        extracted[key] = {
                            "value": value,
                            "evidence": f"{line}: {value}"
                        }

        return extracted

    def extract(self, text):

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        extracted = self.extract_fields(text)

        entities = {}

        for key, field in extracted.items():
            config = self.FIELD_TYPES[key]

            entities[key] = Entity(
                name=field["value"],
                entity_type=config["type"],
                evidence=field["evidence"]
            )

        relationships = []

        def add_relationship(source_key, relation, target_key):

            source = entities.get(source_key)
            target = entities.get(target_key)

            if source and target:
                relationships.append(
                    Relationship(
                        source=source.name,
                        relation=relation,
                        target=target.name,
                        evidence=(
                            source.evidence
                            + " | "
                            + target.evidence
                        )
                    )
                )

        add_relationship("customer", "OWNS", "policy")
        add_relationship("policy", "ISSUED_BY", "insurer")
        add_relationship("policy", "HAS_PREMIUM", "premium")
        add_relationship("policy", "HAS_COVERAGE", "coverage")

        return ExtractionResult(
            entities=list(entities.values()),
            relationships=relationships
        )
