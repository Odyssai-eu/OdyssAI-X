"""Presentation rules for decision probabilities; logits and selection stay raw."""


def display_probabilities(probabilities):
    values = list(probabilities)
    if not values:
        return values

    winner = max(range(len(values)), key=values.__getitem__)
    if values[winner] > 0.95 and all(
        value < 0.045 for i, value in enumerate(values) if i != winner
    ):
        return [1.0 if i == winner else 0.0 for i in range(len(values))]

    visible = [value if value >= 0.01 else 0.0 for value in values]
    total = sum(visible)
    return [value / total for value in visible]
