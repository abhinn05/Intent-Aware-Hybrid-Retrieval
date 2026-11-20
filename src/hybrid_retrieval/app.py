import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hybrid_retrieval.data_loader import load_profiles_from_csv
from hybrid_retrieval.graph_search import GraphSearcher
from hybrid_retrieval.intent_classifier import IntentClassifier, analyze_query
from hybrid_retrieval.orchestration import retrieve_with_intent
from hybrid_retrieval.reformulation import QueryReformulator, analyze_reformulations
from hybrid_retrieval.search import LexicalSearcher, SemanticSearcher, hybrid_search

DEFAULT_DATASET_PATH = Path('dataset/data.csv')
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_POOL_MULTIPLIER = 5
DEFAULT_LEXICAL_WEIGHT = 1.5
DEFAULT_SEMANTIC_WEIGHT = 1.0
STATIC_DIR = Path(__file__).resolve().parent / 'static'

app = FastAPI(title='Intent-Aware Hybrid Retrieval')
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


def _resolve_dataset_path() -> Path:
    configured_path = os.getenv('DATASET_PATH') or os.getenv('HYBRID_RETRIEVAL_DATASET_PATH')
    dataset_path = Path(configured_path) if configured_path else DEFAULT_DATASET_PATH
    if dataset_path.exists():
        return dataset_path
    raise FileNotFoundError(
        f'Dataset CSV not found at {dataset_path}. Set DATASET_PATH or HYBRID_RETRIEVAL_DATASET_PATH to your dataset CSV.'
    )


@lru_cache(maxsize=1)
def _load_profiles():
    return load_profiles_from_csv(_resolve_dataset_path())


@lru_cache(maxsize=1)
def _semantic_searcher() -> SemanticSearcher:
    return SemanticSearcher()


@lru_cache(maxsize=1)
def _intent_classifier() -> IntentClassifier:
    return IntentClassifier()


@lru_cache(maxsize=1)
def _query_reformulator() -> QueryReformulator:
    return QueryReformulator()


@lru_cache(maxsize=1)
def _graph_searcher() -> GraphSearcher:
    return GraphSearcher()


def _resolved_candidate_pool_size(top_k: int, candidate_pool_size: int | None) -> int:
    return max(top_k, candidate_pool_size or top_k * DEFAULT_CANDIDATE_POOL_MULTIPLIER)


def _serialize_profile(profile) -> dict:
    return {
        'id': profile.id,
        'name': profile.name,
        'roles': getattr(profile, 'potential_roles', []),
    }


def _hybrid_strategy(top_k: int, candidate_pool_size: int | None, lexical_weight: float, semantic_weight: float) -> dict:
    resolved_candidate_pool_size = _resolved_candidate_pool_size(top_k, candidate_pool_size)
    return {
        'type': 'hybrid',
        'top_k': top_k,
        'candidate_pool_size': resolved_candidate_pool_size,
        'lexical_weight': lexical_weight,
        'semantic_weight': semantic_weight,
        'semantic_model': _semantic_searcher().model_name,
    }


def _hybrid_results(query: str, top_k: int, candidate_pool_size: int | None, lexical_weight: float, semantic_weight: float) -> dict:
    try:
        profiles = _load_profiles()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    strategy = _hybrid_strategy(top_k, candidate_pool_size, lexical_weight, semantic_weight)
    try:
        results = hybrid_search(
            profiles,
            query,
            semantic_searcher=_semantic_searcher(),
            top_k=top_k,
            candidate_pool_size=strategy['candidate_pool_size'],
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )
        serialized_results = [
            {
                'id': profile.id,
                'name': profile.name,
                'roles': profile.potential_roles,
                'core_skills': profile.core_skills,
                'secondary_skills': profile.secondary_skills,
                'soft_skills': profile.soft_skills,
                'years_of_experience': profile.years_of_experience,
                'skill_summary': profile.skill_summary,
            }
            for profile in results
        ]
    except ModuleNotFoundError:
        lexical_results = LexicalSearcher().search(profiles, query, top_k=top_k)
        strategy = {
            **strategy,
            'type': 'hybrid_fallback_lexical',
            'fallback_reason': 'semantic_dependencies_unavailable',
        }
        serialized_results = [
            {
                'id': result.profile.id,
                'name': result.profile.name,
                'roles': result.profile.potential_roles,
                'core_skills': result.profile.core_skills,
                'secondary_skills': result.profile.secondary_skills,
                'soft_skills': result.profile.soft_skills,
                'years_of_experience': result.profile.years_of_experience,
                'skill_summary': result.profile.skill_summary,
                'score': result.score,
            }
            for result in lexical_results
        ]
    return {
        'query': query,
        'strategy': strategy,
        'results': serialized_results,
    }


def _integrated_demo_payload(
    query: str,
    top_k: int,
    candidate_pool_size: int | None,
    lexical_weight: float,
    semantic_weight: float,
) -> dict:
    query_analysis = analyze_query(query, classifier=_intent_classifier())
    reformulations = analyze_reformulations(
        query,
        query_analysis.get('intent'),
        reformulator=_query_reformulator(),
    )
    try:
        profiles = _load_profiles()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    graph_payload = {
        'status': 'disabled',
        'summary': 'Graph retrieval did not run for this request.',
        'backend': 'neo4j',
        'results': [],
        'result_count': 0,
    }
    try:
        retrieval_payload = retrieve_with_intent(
            profiles=profiles,
            query=query,
            intent=query_analysis.get('intent'),
            reformulations=reformulations.get('queries'),
            semantic_searcher=_semantic_searcher(),
            graph_searcher=_graph_searcher(),
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )
        strategy = retrieval_payload['strategy']
        graph_payload = retrieval_payload['graph']
        serialized_results = [_serialize_profile(profile) for profile in retrieval_payload['results']]
    except ModuleNotFoundError:
        baseline_payload = _hybrid_results(query, top_k, candidate_pool_size, lexical_weight, semantic_weight)
        strategy = {
            **baseline_payload['strategy'],
            'type': 'integrated_fallback_hybrid',
            'fallback_reason': 'semantic_dependencies_unavailable',
            'executed_queries': [query],
            'intent': query_analysis.get('intent'),
            'graph_status': 'unavailable',
        }
        graph_payload = {
            'status': 'unavailable',
            'summary': 'Graph retrieval was skipped because the semantic retrieval stack could not complete the integrated path.',
            'backend': 'neo4j',
            'results': [],
            'result_count': 0,
        }
        serialized_results = baseline_payload['results']

    return {
        'query': {
            'text': query,
            'phase': 'integrated',
        },
        'query_analysis': query_analysis,
        'reformulations': reformulations,
        'retrieval': {
            'mode': strategy['type'],
            'strategy': strategy,
        },
        'graph': graph_payload,
        'results': serialized_results,
        'explanations': {
            'status': 'ready',
            'summary': 'Results were produced by intent-aware orchestration over the original query plus any successful reformulations.',
            'executed_queries': strategy.get('executed_queries', [query]),
            'intent': query_analysis.get('intent'),
            'graph_status': graph_payload['status'],
        },
    }


@app.get('/')
def demo_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/search/lexical')
def lexical_search(
    query: str = Query(..., min_length=2),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=100),
):
    try:
        profiles = _load_profiles()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    results = LexicalSearcher().search(profiles, query, top_k=top_k)
    return {
        'query': query,
        'strategy': {
            'type': 'lexical',
            'top_k': top_k,
        },
        'results': [
            {
                'id': result.profile.id,
                'name': result.profile.name,
                'roles': result.profile.potential_roles,
                'core_skills': result.profile.core_skills,
                'secondary_skills': result.profile.secondary_skills,
                'soft_skills': result.profile.soft_skills,
                'years_of_experience': result.profile.years_of_experience,
                'skill_summary': result.profile.skill_summary,
                'score': result.score,
            }
            for result in results
        ],
    }


@app.get('/search/hybrid')
def hybrid_profile_search(
    query: str = Query(..., min_length=2),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=100),
    candidate_pool_size: int | None = Query(None, ge=1, le=500),
    lexical_weight: float = Query(DEFAULT_LEXICAL_WEIGHT, gt=0),
    semantic_weight: float = Query(DEFAULT_SEMANTIC_WEIGHT, gt=0),
):
    return _hybrid_results(query, top_k, candidate_pool_size, lexical_weight, semantic_weight)


@app.get('/demo/search')
def demo_search(
    query: str = Query(..., min_length=2),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=100),
    candidate_pool_size: int | None = Query(None, ge=1, le=500),
    lexical_weight: float = Query(DEFAULT_LEXICAL_WEIGHT, gt=0),
    semantic_weight: float = Query(DEFAULT_SEMANTIC_WEIGHT, gt=0),
):
    return _integrated_demo_payload(query, top_k, candidate_pool_size, lexical_weight, semantic_weight)

