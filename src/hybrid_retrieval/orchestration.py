from __future__ import annotations

from typing import Any

from .search import LexicalSearcher, SemanticSearcher, hybrid_search, reciprocal_rank_fusion


def retrieve_with_intent(
    profiles: list,
    query: str,
    intent: str | None,
    reformulations: list[str] | None,
    semantic_searcher: SemanticSearcher | None = None,
    graph_searcher: Any | None = None,
    top_k: int = 10,
    candidate_pool_size: int | None = None,
    lexical_weight: float = 1.5,
    semantic_weight: float = 1.0,
) -> dict[str, Any]:
    semantic_searcher = semantic_searcher or SemanticSearcher()
    query_batch = [query] + [item for item in (reformulations or []) if item and item != query]
    policy = _policy_for_intent(intent, lexical_weight, semantic_weight, candidate_pool_size, top_k)

    rank_lists: list[list[str]] = []
    rank_weights: list[float] = []
    fusion_sources: list[str] = []
    for query_index, query_text in enumerate(query_batch):
        if policy['retrieval_type'] == 'lexical':
            ranked_ids = [result.profile.id for result in LexicalSearcher().search(profiles, query_text, top_k=policy['candidate_pool_size'])]
            fusion_source = 'lexical'
        else:
            ranked_ids = [
                profile.id
                for profile in hybrid_search(
                    profiles,
                    query_text,
                    semantic_searcher=semantic_searcher,
                    top_k=policy['candidate_pool_size'],
                    candidate_pool_size=policy['candidate_pool_size'],
                    lexical_weight=policy['lexical_weight'],
                    semantic_weight=policy['semantic_weight'],
                )
            ]
            fusion_source = 'hybrid'
        rank_lists.append(ranked_ids)
        rank_weights.append(_text_query_weight(query_index, intent))
        fusion_sources.append(fusion_source)

    graph_payload = _graph_payload_for_request(
        query=query,
        graph_searcher=graph_searcher,
        candidate_pool_size=policy['candidate_pool_size'],
        use_graph=policy['use_graph'],
    )
    if graph_payload['ranked_ids']:
        rank_lists.append(graph_payload['ranked_ids'])
        rank_weights.append(policy['graph_weight'])
        fusion_sources.append('graph')

    fused_ids = reciprocal_rank_fusion(rank_lists, weights=rank_weights)
    profiles_by_id = {profile.id: profile for profile in profiles}
    results = [profiles_by_id[profile_id] for profile_id in fused_ids[:top_k] if profile_id in profiles_by_id]

    return {
        'results': results,
        'strategy': {
            'type': policy['strategy_type'],
            'intent': intent,
            'retrieval_type': policy['retrieval_type'],
            'top_k': top_k,
            'candidate_pool_size': policy['candidate_pool_size'],
            'lexical_weight': policy['lexical_weight'],
            'semantic_weight': policy['semantic_weight'],
            'query_count': len(query_batch),
            'executed_queries': query_batch,
            'semantic_model': getattr(semantic_searcher, 'model_name', None),
            'graph_status': graph_payload['status'],
            'graph_backend': graph_payload.get('backend'),
            'graph_weight': policy['graph_weight'],
            'fusion_sources': fusion_sources,
        },
        'graph': _public_graph_payload(graph_payload),
    }


def _policy_for_intent(intent: str | None, lexical_weight: float, semantic_weight: float, candidate_pool_size: int | None, top_k: int) -> dict[str, Any]:
    normalized_intent = (intent or 'fallback').lower()
    resolved_candidate_pool_size = max(top_k, candidate_pool_size or top_k * 5)
    if normalized_intent == 'exact_lookup':
        return {
            'strategy_type': 'intent_exact_lookup',
            'retrieval_type': 'lexical',
            'candidate_pool_size': resolved_candidate_pool_size,
            'lexical_weight': lexical_weight + 0.5,
            'semantic_weight': semantic_weight,
            'use_graph': False,
            'graph_weight': 0.0,
        }
    if normalized_intent == 'exploratory':
        return {
            'strategy_type': 'intent_exploratory',
            'retrieval_type': 'hybrid',
            'candidate_pool_size': max(resolved_candidate_pool_size, top_k * 6),
            'lexical_weight': lexical_weight,
            'semantic_weight': semantic_weight + 0.2,
            'use_graph': True,
            'graph_weight': 1.1,
        }
    if normalized_intent == 'comparison':
        return {
            'strategy_type': 'intent_comparison',
            'retrieval_type': 'hybrid',
            'candidate_pool_size': max(resolved_candidate_pool_size, top_k * 6),
            'lexical_weight': lexical_weight,
            'semantic_weight': semantic_weight + 0.1,
            'use_graph': True,
            'graph_weight': 1.1,
        }
    if normalized_intent == 'ambiguous':
        return {
            'strategy_type': 'intent_ambiguous',
            'retrieval_type': 'lexical',
            'candidate_pool_size': resolved_candidate_pool_size,
            'lexical_weight': lexical_weight + 0.5,
            'semantic_weight': 0.0,
            'use_graph': False,
            'graph_weight': 0.0,
        }
    return {
        'strategy_type': 'hybrid_fallback',
        'retrieval_type': 'hybrid',
        'candidate_pool_size': resolved_candidate_pool_size,
        'lexical_weight': lexical_weight,
        'semantic_weight': semantic_weight,
        'use_graph': True,
        'graph_weight': 1.0,
    }


def _text_query_weight(query_index: int, intent: str | None) -> float:
    if query_index > 0:
        return 1.0
    if (intent or '').lower() in {'exact_lookup', 'ambiguous'}:
        return 1.6
    return 1.2


def _graph_payload_for_request(query: str, graph_searcher: Any | None, candidate_pool_size: int, use_graph: bool) -> dict[str, Any]:
    backend = getattr(graph_searcher, 'backend_name', 'neo4j') if graph_searcher is not None else 'neo4j'
    base_payload = {
        'backend': backend,
        'results': [],
        'result_count': 0,
        'ranked_ids': [],
    }
    if not use_graph:
        return {
            **base_payload,
            'status': 'skipped',
            'summary': 'Graph retrieval was intentionally skipped for this intent policy.',
        }
    if graph_searcher is None:
        return {
            **base_payload,
            'status': 'disabled',
            'summary': 'Graph retrieval is not configured for this run.',
        }

    try:
        raw_results = graph_searcher.search(query, top_k=candidate_pool_size)
    except Exception as error:
        return {
            **base_payload,
            'status': 'unavailable',
            'summary': 'Graph retrieval could not run, so ranking ignored it.',
            'error': str(error),
        }

    results = [_normalize_graph_result(item) for item in raw_results or []]
    results = [item for item in results if item and item.get('id')]
    ranked_ids = [item['id'] for item in results]
    if not ranked_ids:
        return {
            **base_payload,
            'status': 'empty',
            'summary': 'Graph retrieval ran but returned no candidate matches for this query.',
        }

    return {
        'backend': backend,
        'status': 'ready',
        'summary': f'Graph retrieval contributed {len(ranked_ids)} ranked candidate hits.',
        'results': results,
        'result_count': len(ranked_ids),
        'ranked_ids': ranked_ids,
    }


def _normalize_graph_result(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, dict):
        candidate_id = str(item.get('id', '')).strip()
        if not candidate_id:
            return None
        return {
            'id': candidate_id,
            'score': item.get('score'),
            'matched_skills': item.get('matched_skills', []),
            'matched_roles': item.get('matched_roles', []),
            'explanation': item.get('explanation', ''),
            'name': item.get('name'),
        }
    candidate_id = str(getattr(item, 'id', '')).strip()
    if not candidate_id:
        return None
    return {
        'id': candidate_id,
        'score': getattr(item, 'score', None),
        'matched_skills': getattr(item, 'matched_skills', []),
        'matched_roles': getattr(item, 'matched_roles', []),
        'explanation': getattr(item, 'explanation', ''),
        'name': getattr(item, 'name', None),
    }


def _public_graph_payload(graph_payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in graph_payload.items() if key != 'ranked_ids'}
