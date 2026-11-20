from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re

from .profiles import Profile

_TOKEN_RE = re.compile(r"[a-z0-9+.#/-]+")


@dataclass(slots=True)
class SearchResult:
    profile: Profile
    score: float


class SemanticSearcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", encoder=None):
        self.model_name = model_name
        self.encoder = encoder
        self._cached_profile_ids: tuple[str, ...] = ()
        self._cached_profile_embeddings: list[list[float]] = []

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(self.model_name)
        return self.encoder

    def _encode_profiles(self, profiles: list[Profile]) -> list[list[float]]:
        profile_ids = tuple(profile.id for profile in profiles)
        if profile_ids != self._cached_profile_ids:
            texts = [profile.search_text for profile in profiles]
            self._cached_profile_embeddings = self._get_encoder().encode(texts, normalize_embeddings=True)
            self._cached_profile_ids = profile_ids
        return self._cached_profile_embeddings

    def search(self, profiles: list[Profile], query: str, top_k: int = 10) -> list[Profile]:
        profile_embeddings = self._encode_profiles(profiles)
        query_embedding = self._get_encoder().encode([query], normalize_embeddings=True)[0]

        scored_profiles: list[tuple[float, Profile]] = []
        for profile, profile_embedding in zip(profiles, profile_embeddings, strict=False):
            score = _dot_product(query_embedding, profile_embedding)
            if score > 0:
                scored_profiles.append((score, profile))

        scored_profiles.sort(key=lambda item: item[0], reverse=True)
        return [profile for _, profile in scored_profiles[:top_k]]


class LexicalSearcher:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._cached_profile_ids: tuple[str, ...] = ()
        self._doc_term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        self._inverse_document_frequency: dict[str, float] = {}
        self._average_document_length = 0.0

    def _build_index(self, profiles: list[Profile]) -> None:
        profile_ids = tuple(profile.id for profile in profiles)
        if profile_ids == self._cached_profile_ids:
            return

        tokenized_corpus = [_tokenize(profile.search_text) for profile in profiles]
        self._doc_term_frequencies = [Counter(tokens) for tokens in tokenized_corpus]
        self._document_lengths = [len(tokens) for tokens in tokenized_corpus]
        self._average_document_length = sum(self._document_lengths) / len(self._document_lengths) if self._document_lengths else 0.0
        self._inverse_document_frequency = _compute_idf(tokenized_corpus)
        self._cached_profile_ids = profile_ids

    def search(self, profiles: list[Profile], query: str, top_k: int = 10) -> list[SearchResult]:
        self._build_index(profiles)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored_results: list[SearchResult] = []
        for profile, term_frequencies, document_length in zip(
            profiles, self._doc_term_frequencies, self._document_lengths, strict=False
        ):
            score = _bm25_score(
                query_tokens=query_tokens,
                term_frequencies=term_frequencies,
                document_length=document_length,
                average_document_length=self._average_document_length,
                inverse_document_frequency=self._inverse_document_frequency,
                k1=self.k1,
                b=self.b,
            )
            if score > 0:
                scored_results.append(SearchResult(profile=profile, score=score))

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:top_k]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _compute_idf(tokenized_corpus: list[list[str]]) -> dict[str, float]:
    document_count = len(tokenized_corpus)
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_corpus:
        document_frequencies.update(set(tokens))

    inverse_document_frequency: dict[str, float] = {}
    for token, frequency in document_frequencies.items():
        numerator = document_count - frequency + 0.5
        denominator = frequency + 0.5
        inverse_document_frequency[token] = math.log(1 + (numerator / denominator))
    return inverse_document_frequency


def _bm25_score(
    query_tokens: list[str],
    term_frequencies: Counter[str],
    document_length: int,
    average_document_length: float,
    inverse_document_frequency: dict[str, float],
    k1: float,
    b: float,
) -> float:
    score = 0.0
    for token in query_tokens:
        frequency = term_frequencies.get(token, 0)
        if frequency == 0:
            continue

        idf = inverse_document_frequency.get(token, 0.0)
        normalization = 1 - b + b * (document_length / average_document_length) if average_document_length else 1.0
        score += idf * ((frequency * (k1 + 1)) / (frequency + k1 * normalization))
    return float(score)


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))


def search_lexical(profiles: list[Profile], query: str, top_k: int = 10) -> list[SearchResult]:
    return LexicalSearcher().search(profiles, query, top_k=top_k)


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    weights = weights or [1.0] * len(rank_lists)

    for rank_list, weight in zip(rank_lists, weights, strict=False):
        for rank, item_id in enumerate(rank_list, start=1):
            scores[item_id] += weight / (k + rank)

    return [item_id for item_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def hybrid_search(
    profiles: list[Profile],
    query: str,
    semantic_searcher: SemanticSearcher | None = None,
    top_k: int = 10,
    candidate_pool_size: int | None = None,
    lexical_weight: float = 1.5,
    semantic_weight: float = 1.0,
) -> list[Profile]:
    lexical_searcher = LexicalSearcher()
    semantic_searcher = semantic_searcher or SemanticSearcher()
    candidate_pool_size = max(top_k, candidate_pool_size or top_k * 5)

    lexical_results = lexical_searcher.search(profiles, query, top_k=candidate_pool_size)
    semantic_results = semantic_searcher.search(profiles, query, top_k=candidate_pool_size)

    lexical_rank = [result.profile.id for result in lexical_results]
    semantic_rank = [profile.id for profile in semantic_results]
    fused_ids = reciprocal_rank_fusion([lexical_rank, semantic_rank], weights=[lexical_weight, semantic_weight])

    profiles_by_id = {profile.id: profile for profile in profiles}
    return [profiles_by_id[profile_id] for profile_id in fused_ids[:top_k] if profile_id in profiles_by_id]
