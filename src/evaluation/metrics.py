
def calculate_metrics(expected, predicted):
    expected = set(expected)
    predicted = set(predicted)

    true_positive = len(expected.intersection(predicted))
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }


def evaluate_extraction(expected, predicted):
    expected_entities = {
        (item["name"], item["entity_type"])
        for item in expected["entities"]
    }

    predicted_entities = {
        (item.name, item.entity_type)
        for item in predicted.entities
    }

    expected_relationships = {
        (item["source"], item["relation"], item["target"])
        for item in expected["relationships"]
    }

    predicted_relationships = {
        (item.source, item.relation, item.target)
        for item in predicted.relationships
    }

    return {
        "entities": calculate_metrics(
            expected_entities, predicted_entities
        ),
        "relationships": calculate_metrics(
            expected_relationships, predicted_relationships
        )
    }
