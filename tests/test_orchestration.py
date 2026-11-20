from hybrid_retrieval.orchestration import retrieve_with_intent
from hybrid_retrieval.profiles import Profile


class FakeSemanticSearcher:
    model_name = 'fake-semantic'

    def search(self, profiles, query, top_k=10):
        if 'comparison' in query:
            return [profiles[1], profiles[0], profiles[2]][:top_k]
        return profiles[:top_k]


class FakeGraphSearcher:
    backend_name = 'neo4j'

    def search(self, query: str, top_k: int = 10):
        return [
            {
                'id': 'p3',
                'score': 7.5,
                'matched_skills': ['postgres'],
                'matched_roles': ['database engineer'],
                'explanation': 'Matched through graph skill and role nodes.',
            }
        ][:top_k]


class ExplodingGraphSearcher:
    backend_name = 'neo4j'

    def search(self, query: str, top_k: int = 10):
        raise AssertionError('graph search should have been skipped')


def make_profile(profile_id: str, search_text: str, roles: list[str] | None = None) -> Profile:
    return Profile(
        id=profile_id,
        name=profile_id,
        years_of_experience=0.0,
        core_skills=[],
        secondary_skills=[],
        soft_skills=[],
        potential_roles=roles or [],
        skill_summary='',
        raw_core_skills='',
        raw_secondary_skills='',
        raw_soft_skills='',
        raw_potential_roles='',
        search_text=search_text,
    )


def test_exact_lookup_uses_lexical_policy_and_preserves_executed_queries():
    profiles = [
        make_profile('p1', 'backend developer python sql', ['backend developer']),
        make_profile('p2', 'software engineer java', ['software engineer']),
    ]

    payload = retrieve_with_intent(
        profiles=profiles,
        query='backend developer python sql',
        intent='exact_lookup',
        reformulations=['python sql backend developer'],
        semantic_searcher=FakeSemanticSearcher(),
        graph_searcher=ExplodingGraphSearcher(),
        top_k=2,
    )

    assert payload['strategy']['type'] == 'intent_exact_lookup'
    assert payload['strategy']['retrieval_type'] == 'lexical'
    assert payload['strategy']['executed_queries'] == ['backend developer python sql', 'python sql backend developer']
    assert payload['strategy']['graph_status'] == 'skipped'
    assert payload['graph']['status'] == 'skipped'
    assert payload['results'][0].id == 'p1'


def test_comparison_uses_hybrid_policy_and_fuses_graph_results():
    profiles = [
        make_profile('p1', 'postgres backend developer performance', ['backend developer']),
        make_profile('p2', 'comparison mongodb postgres tradeoffs scaling', ['database engineer']),
        make_profile('p3', 'distributed systems reliability', ['database engineer']),
    ]

    payload = retrieve_with_intent(
        profiles=profiles,
        query='postgres vs mongodb comparison',
        intent='comparison',
        reformulations=['comparison postgres mongodb tradeoffs'],
        semantic_searcher=FakeSemanticSearcher(),
        graph_searcher=FakeGraphSearcher(),
        top_k=3,
    )

    assert payload['strategy']['type'] == 'intent_comparison'
    assert payload['strategy']['retrieval_type'] == 'hybrid'
    assert payload['strategy']['semantic_model'] == 'fake-semantic'
    assert payload['strategy']['query_count'] == 2
    assert payload['strategy']['graph_status'] == 'ready'
    assert 'graph' in payload['strategy']['fusion_sources']
    assert payload['graph']['status'] == 'ready'
    assert payload['graph']['result_count'] == 1
    assert payload['graph']['results'][0]['id'] == 'p3'
    assert any(result.id == 'p3' for result in payload['results'])


def test_ambiguous_uses_lexical_only_without_semantic_or_graph_search():
    profiles = [
        make_profile('p1', 'chen backend developer python', ['backend developer']),
        make_profile('p2', 'mongodb comparison scaling', ['database engineer']),
    ]

    payload = retrieve_with_intent(
        profiles=profiles,
        query='Chen',
        intent='ambiguous',
        reformulations=None,
        semantic_searcher=FakeSemanticSearcher(),
        graph_searcher=ExplodingGraphSearcher(),
        top_k=2,
    )

    assert payload['strategy']['type'] == 'intent_ambiguous'
    assert payload['strategy']['retrieval_type'] == 'lexical'
    assert payload['strategy']['query_count'] == 1
    assert payload['strategy']['graph_status'] == 'skipped'
    assert payload['graph']['status'] == 'skipped'
