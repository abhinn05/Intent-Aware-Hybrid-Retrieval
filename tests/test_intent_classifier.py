from hybrid_retrieval.intent_classifier import (
    IntentClassifier,
    _parse_intent_response,
    analyze_query,
)


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.response


def test_parse_intent_response_accepts_plain_label():
    assert _parse_intent_response('comparison') == 'comparison'


def test_parse_intent_response_accepts_labeled_output():
    assert _parse_intent_response('Intent: EXPLORATORY') == 'exploratory'


def test_classifier_uses_injected_client_and_returns_prediction():
    client = FakeClient('comparison')
    prediction = IntentClassifier(client=client).predict('compare postgres vs mongodb')

    assert prediction.intent == 'comparison'
    assert prediction.confidence == 1.0
    assert prediction.scores['comparison'] == 1.0
    assert 'exact_lookup' in prediction.scores
    assert client.prompts
    assert 'comparison' in client.prompts[0].lower()


class BrokenClassifier(IntentClassifier):
    def predict(self, query: str):
        raise ModuleNotFoundError('langchain_ollama')


def test_analyze_query_degrades_cleanly_when_dependencies_are_missing():
    payload = analyze_query('backend python', classifier=BrokenClassifier())

    assert payload['status'] == 'unavailable'
    assert payload['intent'] is None
    assert payload['scores'] == {}


class RuntimeBrokenClassifier(IntentClassifier):
    def predict(self, query: str):
        raise ConnectionError('ollama connection refused')


def test_analyze_query_degrades_cleanly_when_runtime_backend_is_down():
    payload = analyze_query('backend python', classifier=RuntimeBrokenClassifier())

    assert payload['status'] == 'unavailable'
    assert payload['intent'] is None
    assert 'connection refused' in payload['error']
