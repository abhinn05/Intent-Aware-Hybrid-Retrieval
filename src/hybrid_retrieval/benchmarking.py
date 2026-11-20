from statistics import mean

from .benchmark_queries import BenchmarkQuery
from .evaluation import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from .profiles import Profile
from .search import SemanticSearcher, hybrid_search, search_lexical

_METRICS = ('precision_at_k', 'recall_at_k', 'average_precision', 'ndcg_at_k')


def build_benchmark_rows(
    profiles: list[Profile],
    benchmark_queries: list[BenchmarkQuery],
    semantic_searcher: SemanticSearcher,
    top_k: int = 10,
    candidate_pool_size: int | None = None,
    lexical_weight: float = 1.5,
    semantic_weight: float = 1.0,
) -> dict[str, list[dict[str, float | str]]]:
    rows = {'lexical': [], 'hybrid': []}

    for benchmark_query in benchmark_queries:
        relevant_ids = {profile.id for profile in profiles if benchmark_query.is_relevant(profile)}
        lexical_ranked_ids = [result.profile.id for result in search_lexical(profiles, benchmark_query.text, top_k=top_k)]
        hybrid_ranked_ids = [
            profile.id
            for profile in hybrid_search(
                profiles,
                benchmark_query.text,
                semantic_searcher,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                lexical_weight=lexical_weight,
                semantic_weight=semantic_weight,
            )
        ]

        rows['lexical'].append(_score_row(benchmark_query, lexical_ranked_ids, relevant_ids, top_k))
        rows['hybrid'].append(_score_row(benchmark_query, hybrid_ranked_ids, relevant_ids, top_k))

    return rows


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    return {metric: mean(float(row[metric]) for row in rows) for metric in _METRICS}


def _score_row(
    benchmark_query: BenchmarkQuery,
    ranked_ids: list[str],
    relevant_ids: set[str],
    top_k: int,
) -> dict[str, float | str]:
    return {
        'query_id': benchmark_query.query_id,
        'intent_family': benchmark_query.intent_family,
        'precision_at_k': precision_at_k(ranked_ids, relevant_ids, top_k),
        'recall_at_k': recall_at_k(ranked_ids, relevant_ids, top_k),
        'average_precision': average_precision(ranked_ids, relevant_ids),
        'ndcg_at_k': ndcg_at_k(ranked_ids, relevant_ids, top_k),
    }