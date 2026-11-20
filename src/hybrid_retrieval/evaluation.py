from math import log2


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_ids = ranked_ids[:k]
    if not top_ids:
        return 0.0
    hits = sum(1 for item_id in top_ids if item_id in relevant_ids)
    return hits / len(top_ids)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_ids = ranked_ids[:k]
    hits = sum(1 for item_id in top_ids if item_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0

    precision_values: list[float] = []
    hits = 0
    for index, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            hits += 1
            precision_values.append(hits / index)

    if not precision_values:
        return 0.0
    return sum(precision_values) / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_ids = ranked_ids[:k]
    dcg = 0.0
    for index, item_id in enumerate(top_ids, start=1):
        gain = 1.0 if item_id in relevant_ids else 0.0
        dcg += gain / log2(index + 1)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg
