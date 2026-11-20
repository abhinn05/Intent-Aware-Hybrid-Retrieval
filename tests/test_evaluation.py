from hybrid_retrieval.evaluation import average_precision, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k_counts_relevant_items_in_top_k():
    ranked_ids = ["a", "b", "c", "d"]
    relevant_ids = {"a", "c", "x"}

    assert precision_at_k(ranked_ids, relevant_ids, k=3) == 2 / 3


def test_recall_at_k_uses_total_relevant_set_size():
    ranked_ids = ["a", "b", "c", "d"]
    relevant_ids = {"a", "c", "x"}

    assert recall_at_k(ranked_ids, relevant_ids, k=3) == 2 / 3


def test_average_precision_rewards_early_relevant_hits():
    ranked_ids = ["a", "b", "c", "d"]
    relevant_ids = {"a", "c"}

    assert average_precision(ranked_ids, relevant_ids) == (1.0 + (2 / 3)) / 2


def test_ndcg_at_k_is_one_for_perfect_ranking():
    ranked_ids = ["a", "b", "c"]
    relevant_ids = {"a", "b", "c"}

    assert ndcg_at_k(ranked_ids, relevant_ids, k=3) == 1.0
