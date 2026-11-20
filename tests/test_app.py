from fastapi.testclient import TestClient

from hybrid_retrieval import app as app_module
from hybrid_retrieval.profiles import Profile


def make_profile(profile_id: str, name: str, search_text: str, roles: list[str] | None = None) -> Profile:
    return Profile(
        id=profile_id,
        name=name,
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


class DummySemanticSearcher:
    model_name = 'dummy-mini-model'

    def search(self, profiles: list[Profile], query: str, top_k: int = 10) -> list[Profile]:
        return profiles[:top_k]


class DummyIntentClassifier:
    model_dir = 'intent_classification/intent_model/best_model'

    def predict(self, query: str):
        class Prediction:
            intent = 'comparison'
            confidence = 0.88
            scores = {
                'exact_lookup': 0.02,
                'ambiguous': 0.03,
                'exploratory': 0.07,
                'comparison': 0.88,
            }
            model_path = 'intent_classification/intent_model/best_model'
            model_name = 'best_model'

        return Prediction()


class DummyReformulator:
    model = 'qwen2.5:1.5b'

    def reformulate(self, query: str, intent: str):
        query_text = query
        intent_name = intent

        class Result:
            intent = intent_name
            queries = [f'{query_text} comparison tradeoffs', f'{query_text} scalability']
            model = 'qwen2.5:1.5b'

        return Result()


class DummyGraphSearcher:
    backend_name = 'neo4j'

    def search(self, query: str, top_k: int = 10):
        return [
            {
                'id': 'p2',
                'score': 4.2,
                'matched_skills': ['sql'],
                'matched_roles': ['software engineer'],
                'explanation': 'Matched through graph edges.',
            }
        ][:top_k]


def test_health_endpoint_reports_ok():
    client = TestClient(app_module.app)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_hybrid_endpoint_returns_strategy_metadata(monkeypatch):
    profiles = [
        make_profile('p1', 'Alice', 'python backend developer sql', ['backend developer']),
        make_profile('p2', 'Bob', 'java software engineer', ['software engineer']),
    ]

    app_module._load_profiles.cache_clear()
    app_module._semantic_searcher.cache_clear()
    monkeypatch.setattr(app_module, '_load_profiles', lambda: profiles)
    monkeypatch.setattr(app_module, '_semantic_searcher', lambda: DummySemanticSearcher())

    client = TestClient(app_module.app)
    response = client.get(
        '/search/hybrid',
        params={
            'query': 'python backend',
            'top_k': 5,
            'candidate_pool_size': 20,
            'lexical_weight': 1.7,
            'semantic_weight': 0.9,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload['query'] == 'python backend'
    assert payload['strategy'] == {
        'type': 'hybrid',
        'top_k': 5,
        'candidate_pool_size': 20,
        'lexical_weight': 1.7,
        'semantic_weight': 0.9,
        'semantic_model': 'dummy-mini-model',
    }
    assert payload['results'][0]['id'] == 'p1'


def test_root_serves_demo_ui_shell():
    client = TestClient(app_module.app)

    response = client.get('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'Intent-Aware Hybrid Retrieval Demo' in response.text
    assert 'data-panel="query-plan"' in response.text
    assert 'data-panel="graph"' in response.text


def test_demo_search_returns_integrated_payload_with_graph_status(monkeypatch):
    profiles = [
        make_profile('p1', 'Alice', 'python backend developer sql', ['backend developer']),
        make_profile('p2', 'Bob', 'java software engineer sql', ['software engineer']),
    ]

    app_module._load_profiles.cache_clear()
    app_module._semantic_searcher.cache_clear()
    app_module._intent_classifier.cache_clear()
    app_module._query_reformulator.cache_clear()
    app_module._graph_searcher.cache_clear()
    monkeypatch.setattr(app_module, '_load_profiles', lambda: profiles)
    monkeypatch.setattr(app_module, '_semantic_searcher', lambda: DummySemanticSearcher())
    monkeypatch.setattr(app_module, '_intent_classifier', lambda: DummyIntentClassifier())
    monkeypatch.setattr(app_module, '_query_reformulator', lambda: DummyReformulator())
    monkeypatch.setattr(app_module, '_graph_searcher', lambda: DummyGraphSearcher())

    client = TestClient(app_module.app)
    response = client.get(
        '/demo/search',
        params={
            'query': 'python backend',
            'top_k': 5,
            'candidate_pool_size': 20,
            'lexical_weight': 1.7,
            'semantic_weight': 0.9,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload['query']['text'] == 'python backend'
    assert payload['query']['phase'] == 'integrated'
    assert payload['query_analysis']['status'] == 'ready'
    assert payload['query_analysis']['intent'] == 'comparison'
    assert payload['reformulations']['status'] == 'ready'
    assert payload['reformulations']['count'] == 2
    assert payload['retrieval']['strategy']['type'] == 'intent_comparison'
    assert payload['retrieval']['strategy']['graph_status'] == 'ready'
    assert payload['graph']['status'] == 'ready'
    assert payload['graph']['result_count'] == 1
    assert payload['graph']['results'][0]['id'] == 'p2'
    assert payload['results'][0] == {
        'id': 'p2',
        'name': 'Bob',
        'roles': ['software engineer'],
    }
    assert payload['explanations']['status'] == 'ready'
    assert payload['explanations']['graph_status'] == 'ready'
    assert len(payload['explanations']['executed_queries']) == 3


def test_hybrid_endpoint_degrades_cleanly_when_semantic_dependencies_are_missing(monkeypatch):
    profiles = [
        make_profile('p1', 'Alice', 'python backend developer sql', ['backend developer']),
        make_profile('p2', 'Bob', 'java software engineer', ['software engineer']),
    ]

    app_module._load_profiles.cache_clear()
    app_module._semantic_searcher.cache_clear()
    monkeypatch.setattr(app_module, '_load_profiles', lambda: profiles)

    class BrokenSemanticSearcher:
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        def search(self, profiles: list[Profile], query: str, top_k: int = 10) -> list[Profile]:
            raise ModuleNotFoundError('sentence_transformers')

    monkeypatch.setattr(app_module, '_semantic_searcher', lambda: BrokenSemanticSearcher())

    client = TestClient(app_module.app)
    response = client.get('/search/hybrid', params={'query': 'python backend', 'top_k': 5})

    payload = response.json()

    assert response.status_code == 200
    assert payload['strategy']['type'] == 'hybrid_fallback_lexical'
    assert payload['strategy']['fallback_reason'] == 'semantic_dependencies_unavailable'
    assert payload['results'][0]['id'] == 'p1'


class BrokenIntentClassifier:
    model_name = 'qwen2.5:1.5b'

    def predict(self, query: str):
        raise ConnectionError('ollama connection refused')


def test_demo_search_degrades_cleanly_when_intent_backend_is_down(monkeypatch):
    profiles = [
        make_profile('p1', 'Alice', 'python backend developer sql', ['backend developer']),
        make_profile('p2', 'Bob', 'java software engineer', ['software engineer']),
    ]

    app_module._load_profiles.cache_clear()
    app_module._semantic_searcher.cache_clear()
    app_module._intent_classifier.cache_clear()
    app_module._query_reformulator.cache_clear()
    app_module._graph_searcher.cache_clear()
    monkeypatch.setattr(app_module, '_load_profiles', lambda: profiles)
    monkeypatch.setattr(app_module, '_semantic_searcher', lambda: DummySemanticSearcher())
    monkeypatch.setattr(app_module, '_intent_classifier', lambda: BrokenIntentClassifier())
    monkeypatch.setattr(app_module, '_query_reformulator', lambda: DummyReformulator())
    monkeypatch.setattr(app_module, '_graph_searcher', lambda: DummyGraphSearcher())

    client = TestClient(app_module.app)
    response = client.get('/demo/search', params={'query': 'python backend', 'top_k': 5})

    payload = response.json()

    assert response.status_code == 200
    assert payload['query_analysis']['status'] == 'unavailable'
    assert payload['reformulations']['status'] == 'skipped'
    assert payload['results']
