from hybrid_retrieval.reformulation import QueryReformulator, _parse_reformulations, analyze_reformulations


class FakeClient:
    def invoke(self, payload):
        assert payload['intent'] == 'exploratory'
        return '1. transformer architecture internals\n2. transformer implementation libraries\n3. transformer limitations and failure modes\n4. transformer production use cases\n'


def test_parse_reformulations_strips_numbering_and_blanks():
    queries = _parse_reformulations('1. alpha\n\n2. beta\n- gamma\n', expected_count=4)

    assert queries == ['alpha', 'beta', 'gamma']


def test_reformulator_uses_injected_client_and_returns_queries():
    result = QueryReformulator(client=FakeClient()).reformulate('how transformers work', 'exploratory')

    assert result.intent == 'exploratory'
    assert len(result.queries) == 4
    assert result.queries[0] == 'transformer architecture internals'


def test_analyze_reformulations_skips_ambiguous_queries_to_avoid_hallucinated_rewrites():
    payload = analyze_reformulations('Chen', 'ambiguous')

    assert payload['status'] == 'skipped'
    assert payload['queries'] == ['Chen']
    assert payload['count'] == 1


class BrokenReformulator(QueryReformulator):
    def reformulate(self, query: str, intent: str):
        raise ModuleNotFoundError('langchain_ollama')


def test_analyze_reformulations_falls_back_to_original_query_on_failure():
    payload = analyze_reformulations('what is attention', 'exploratory', reformulator=BrokenReformulator())

    assert payload['status'] == 'unavailable'
    assert payload['queries'] == ['what is attention']
