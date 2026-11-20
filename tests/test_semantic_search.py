from hybrid_retrieval.profiles import Profile
from hybrid_retrieval.search import SemanticSearcher


class FakeEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, normalize_embeddings=True):
        return [self.mapping[text] for text in texts]


def make_profile(profile_id: str, search_text: str) -> Profile:
    return Profile(
        id=profile_id,
        name="",
        years_of_experience=0.0,
        core_skills=[],
        secondary_skills=[],
        soft_skills=[],
        potential_roles=[],
        skill_summary="",
        raw_core_skills="",
        raw_secondary_skills="",
        raw_soft_skills="",
        raw_potential_roles="",
        search_text=search_text,
    )


def test_semantic_search_can_rank_paraphrase_matches_with_encoder_vectors():
    profiles = [
        make_profile("p1", "backend developer python sql"),
        make_profile("p2", "regulatory affairs legal research compliance"),
        make_profile("p3", "java software engineer microservices"),
    ]
    encoder = FakeEncoder(
        {
            "Profiles suitable for compliance or regulatory work": [0.0, 1.0],
            "backend developer python sql": [1.0, 0.0],
            "regulatory affairs legal research compliance": [0.0, 1.0],
            "java software engineer microservices": [0.7, 0.0],
        }
    )

    searcher = SemanticSearcher(encoder=encoder)
    results = searcher.search(profiles, "Profiles suitable for compliance or regulatory work", top_k=3)

    assert [profile.id for profile in results[:1]] == ["p2"]
