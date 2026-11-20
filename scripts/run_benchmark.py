import argparse
import os

from hybrid_retrieval.benchmark_queries import BENCHMARK_QUERIES
from hybrid_retrieval.benchmarking import build_benchmark_rows, summarize_rows
from hybrid_retrieval.data_loader import load_profiles_from_csv
from hybrid_retrieval.search import SemanticSearcher


def run_benchmark(
    dataset_path: str,
    top_k: int = 10,
    candidate_pool_size: int | None = None,
    show_per_query: bool = True,
    lexical_weight: float = 1.5,
    semantic_weight: float = 1.0,
) -> None:
    profiles = load_profiles_from_csv(dataset_path)
    semantic_searcher = SemanticSearcher()
    rows_by_system = build_benchmark_rows(
        profiles,
        BENCHMARK_QUERIES,
        semantic_searcher,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    print_summary('lexical', rows_by_system['lexical'])
    print_summary('hybrid', rows_by_system['hybrid'])
    if show_per_query:
        print_per_query(rows_by_system)


def print_summary(label: str, rows: list[dict[str, float | str]]) -> None:
    print(f'\n{label.upper()} SUMMARY')
    print('-' * 60)
    for metric, value in summarize_rows(rows).items():
        print(f'{metric}: {value:.4f}')


def print_per_query(rows_by_system: dict[str, list[dict[str, float | str]]]) -> None:
    print('\nPER-QUERY BREAKDOWN')
    print('-' * 60)
    for lexical_row, hybrid_row in zip(rows_by_system['lexical'], rows_by_system['hybrid'], strict=False):
        print(
            f"{lexical_row['query_id']} [{lexical_row['intent_family']}] | "
            f"lex_ap={float(lexical_row['average_precision']):.4f} hyb_ap={float(hybrid_row['average_precision']):.4f} | "
            f"lex_ndcg={float(lexical_row['ndcg_at_k']):.4f} hyb_ndcg={float(hybrid_row['ndcg_at_k']):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default=os.getenv('DATASET_PATH') or os.getenv('HYBRID_RETRIEVAL_DATASET_PATH'))
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--candidate-pool-size', type=int)
    parser.add_argument('--no-per-query', action='store_true')
    parser.add_argument('--lexical-weight', type=float, default=1.5)
    parser.add_argument('--semantic-weight', type=float, default=1.0)
    args = parser.parse_args()
    if not args.dataset_path:
        raise SystemExit('dataset path required: pass --dataset-path or set DATASET_PATH / HYBRID_RETRIEVAL_DATASET_PATH')
    run_benchmark(
        dataset_path=args.dataset_path,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        show_per_query=not args.no_per_query,
        lexical_weight=args.lexical_weight,
        semantic_weight=args.semantic_weight,
    )


if __name__ == '__main__':
    main()
