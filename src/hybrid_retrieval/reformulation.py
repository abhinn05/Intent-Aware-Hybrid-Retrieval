from __future__ import annotations

import os

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


_PROMPTS = {
    'exact_lookup': '''You are a backend query generation engine for an inverted-index retrieval system.
DO NOT answer the query. DO NOT explain anything.
Generate exactly 2 short keyword strings.
Rules:
- String 1: the most direct keyword form of the query.
- String 2: the closest accepted synonym, alternate notation, or alias for the same target.
- Each string must stand alone.
- The two strings must not repeat the same keywords.
Output: one string per line, nothing else.''',
    'exploratory': '''You are a backend query generation engine for a parallel semantic retrieval system.
DO NOT answer the query. DO NOT explain anything.
Generate exactly 4 search queries that together cover the topic.
Query 1 - internal mechanism.
Query 2 - implementation and tooling.
Query 3 - limitations and trade-offs.
Query 4 - applications and real-world usage.
Every query must be fully self-contained and non-overlapping.
Output: one query per line, nothing else.''',
    'comparison': '''You are a backend query generation engine for a parallel semantic retrieval system.
DO NOT answer the query. DO NOT explain anything.
Generate exactly 4 self-contained comparison queries.
Query 1 - performance and internals.
Query 2 - use cases and fit.
Query 3 - ecosystem or methodology.
Query 4 - trade-offs and decision criteria.
Both subjects must appear in every query.
Output: one query per line, nothing else.''',
}
_EXPECTED_QUERY_COUNTS = {
    'exact_lookup': 2,
    'exploratory': 4,
    'comparison': 4,
}


@dataclass(slots=True)
class ReformulationResult:
    intent: str
    queries: list[str]
    model: str


class QueryReformulator:
    def __init__(self, model: str = 'qwen2.5:1.5b', client=None):
        self.model = model
        self.client = client
        self._chains: dict[str, Any] | None = None

    def _build_chains(self) -> dict[str, Any]:
        if self._chains is not None:
            return self._chains

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        base_url = os.getenv('OLLAMA_HOST')
        kwargs = {'model': self.model, 'temperature': 0}
        if base_url:
            kwargs['base_url'] = base_url
        llm = ChatOllama(**kwargs)
        chains: dict[str, Any] = {}
        for intent, system_prompt in _PROMPTS.items():
            prompt = ChatPromptTemplate.from_messages([
                ('system', system_prompt),
                ('human', 'User Query: {query}\n\nSearch Queries:\n'),
            ])
            chains[intent] = prompt | llm | StrOutputParser()
        self._chains = chains
        return chains

    def reformulate(self, query: str, intent: str) -> ReformulationResult:
        if not query.strip():
            raise ValueError('query must not be empty')

        normalized_intent = intent.lower()
        if normalized_intent not in _PROMPTS:
            raise ValueError(f'unsupported intent: {intent}')

        if self.client is not None:
            raw_output = self.client.invoke({'query': query, 'intent': normalized_intent, 'prompt': _PROMPTS[normalized_intent]})
        else:
            raw_output = self._build_chains()[normalized_intent].invoke({'query': query})

        queries = _parse_reformulations(raw_output, expected_count=_EXPECTED_QUERY_COUNTS[normalized_intent])
        return ReformulationResult(intent=normalized_intent, queries=queries, model=self.model)


def _parse_reformulations(raw_output: str, expected_count: int) -> list[str]:
    cleaned: OrderedDict[str, None] = OrderedDict()
    for raw_line in raw_output.splitlines():
        line = raw_line.strip().lstrip('-').strip()
        if not line:
            continue
        if line[0].isdigit() and len(line) > 2 and line[1] in {'.', ')'}:
            line = line[2:].strip()
        cleaned.setdefault(line, None)
    parsed = list(cleaned.keys())
    return parsed[:expected_count]


def analyze_reformulations(query: str, intent: str | None, reformulator: QueryReformulator | None = None) -> dict[str, Any]:
    if not intent:
        return {
            'status': 'skipped',
            'summary': 'Intent was unavailable, so reformulation was skipped instead of guessing.',
            'intent': None,
            'queries': [],
            'count': 0,
            'model': None,
        }

    normalized_intent = intent.lower()
    if normalized_intent == 'ambiguous':
        return {
            'status': 'skipped',
            'summary': 'Ambiguous queries stay on exact lexical search only, so rewrite generation was intentionally skipped.',
            'intent': normalized_intent,
            'queries': [query],
            'count': 1,
            'model': None,
        }

    reformulator = reformulator or QueryReformulator()
    try:
        result = reformulator.reformulate(query, normalized_intent)
    except ModuleNotFoundError as error:
        return {
            'status': 'unavailable',
            'summary': 'Reformulation dependencies are missing, so retrieval used the original query only.',
            'intent': normalized_intent,
            'queries': [query],
            'count': 1,
            'model': reformulator.model,
            'error': str(error),
        }
    except Exception as error:
        return {
            'status': 'failed',
            'summary': 'Reformulation failed at runtime, so retrieval used the original query only.',
            'intent': normalized_intent,
            'queries': [query],
            'count': 1,
            'model': reformulator.model,
            'error': str(error),
        }

    queries = result.queries or [query]
    return {
        'status': 'ready',
        'summary': f'Generated {len(queries)} reformulated queries for {result.intent}.',
        'intent': result.intent,
        'queries': queries,
        'count': len(queries),
        'model': result.model,
    }
