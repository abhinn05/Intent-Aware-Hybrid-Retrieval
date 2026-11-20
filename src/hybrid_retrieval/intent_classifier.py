from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any


_LABELS = ('exact_lookup', 'ambiguous', 'exploratory', 'comparison')
_DEFAULT_MODEL_NAME = 'qwen2.5:1.5b'
_INTENT_PROMPT = """You are an intent classifier for a retrieval system.

Classify the user's query into exactly one of these labels:
- EXACT_LOOKUP: the user wants one specific fact, entity, field value, identifier, or exact match.
  Use this for direct fetches, precise filters, IDs, names, single-record lookups, and yes/no attribute checks.
  Examples: 'show candidate 214446', 'does Naveen Gupta have teamwork', 'boiling point of ethanol'.

- AMBIGUOUS: the query is underspecified and has multiple plausible interpretations.
  Use this when the query is too short, too vague, missing the target, or could refer to multiple domains/entities/meanings.
  Examples: 'pipeline', 'attention', 'Beginner IT Support Analyst', 'Chen'.

- EXPLORATORY: the user wants discovery, browsing, overview, broad matching, or a set of possibly relevant results.
  Use this for open-ended search, role exploration, trend discovery, topic coverage, or 'show me candidates/documents related to X'.
  Examples: 'profiles suitable for IT support', 'browse candidates with creativity', 'what PyTorch competency levels are most common'.

- COMPARISON: the user explicitly compares two or more entities, options, people, systems, or asks who is better/stronger/more suitable.
  Use this whenever the core task is side-by-side evaluation or tradeoff judgment.
  Examples: 'PostgreSQL vs MongoDB', 'compare candidate 227493 and 214628', 'who is more ready for HR Manager'.

Decision rules:
- If the query explicitly compares things, choose COMPARISON.
- If it asks for one exact item or fact, choose EXACT_LOOKUP.
- If it is broad and discovery-oriented, choose EXPLORATORY.
- If it is too vague or plausibly means multiple different things, choose AMBIGUOUS.
- Return the best single label only.

Output format: return exactly one label from this set and nothing else:
EXACT_LOOKUP
AMBIGUOUS
EXPLORATORY
COMPARISON

User query: {query}
"""


@dataclass(slots=True)
class IntentPrediction:
    intent: str
    confidence: float
    scores: dict[str, float]
    model_path: str
    model_name: str


class IntentClassifier:
    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, client=None):
        self.model_name = model_name
        self.client = client

    def _get_client(self):
        if self.client is None:
            from langchain_ollama import ChatOllama

            base_url = os.getenv('OLLAMA_HOST')
            kwargs = {'model': self.model_name, 'temperature': 0}
            if base_url:
                kwargs['base_url'] = base_url
            self.client = ChatOllama(**kwargs)
        return self.client

    def predict(self, query: str) -> IntentPrediction:
        if not query.strip():
            raise ValueError('query must not be empty')

        raw_output = self._get_client().invoke(_INTENT_PROMPT.format(query=query))
        if hasattr(raw_output, 'content'):
            raw_output = raw_output.content
        intent = _parse_intent_response(str(raw_output))
        scores = {label: (1.0 if label == intent else 0.0) for label in _LABELS}
        return IntentPrediction(
            intent=intent,
            confidence=1.0,
            scores=scores,
            model_path=f'ollama:{self.model_name}',
            model_name=self.model_name,
        )


def _parse_intent_response(raw_output: str) -> str:
    normalized = raw_output.strip().lower().replace('-', '_').replace(' ', '_')
    for label in _LABELS:
        if normalized == label:
            return label
    for line in raw_output.splitlines():
        cleaned = line.strip().lower().replace('-', '_').replace(' ', '_')
        if cleaned.startswith('intent:'):
            cleaned = cleaned.split(':', 1)[1].strip().replace('-', '_').replace(' ', '_')
        if cleaned in _LABELS:
            return cleaned
    upper_output = raw_output.upper()
    for label in _LABELS:
        if label.upper() in upper_output:
            return label
    raise ValueError(f'Could not parse intent label from classifier output: {raw_output!r}')


def analyze_query(query: str, classifier: IntentClassifier | None = None) -> dict[str, Any]:
    classifier = classifier or IntentClassifier()
    try:
        prediction = classifier.predict(query)
    except (ModuleNotFoundError, ValueError, Exception) as error:
        return {
            'status': 'unavailable',
            'summary': 'Intent classifier could not run, so retrieval fell back to a safe default.',
            'error': str(error),
            'intent': None,
            'confidence': None,
            'scores': {},
            'model': {
                'path': f'ollama:{classifier.model_name}',
                'name': classifier.model_name,
            },
        }

    return {
        'status': 'ready',
        'summary': f'Intent classifier routed the query as {prediction.intent}.',
        'intent': prediction.intent,
        'confidence': prediction.confidence,
        'scores': prediction.scores,
        'model': {
            'path': prediction.model_path,
            'name': prediction.model_name,
        },
    }
