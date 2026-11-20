from hybrid_retrieval.profiles import Profile
from hybrid_retrieval.search import hybrid_search


class FakeSemanticSearcher:
    def search(self, profiles, query, top_k=10):
        return [profiles[1], profiles[0]]


class DepthAwareFakeSemanticSearcher:
    def search(self, profiles, query, top_k=10):
        semantic_rank = [profiles[3], profiles[0], profiles[1], profiles[2]]
        return semantic_rank[:top_k]


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


def test_hybrid_search_combines_lexical_and_semantic_rank_lists():
    profiles = [
        make_profile("p1", "backend developer python sql"),
        make_profile("p2", "software engineer java"),
        make_profile("p3", "data analyst excel"),
    ]

    ranked = hybrid_search(
        profiles=profiles,
        query="backend developer python",
        semantic_searcher=FakeSemanticSearcher(),
        top_k=3,
    )

    assert ranked[0].id == "p1"
    assert ranked[1].id == "p2"


def test_hybrid_search_can_fuse_deeper_candidate_lists_than_top_k():
    profiles = [
        make_profile("p1", "backend developer python sql"),
        make_profile("p2", "backend developer python"),
        make_profile("p3", "backend developer"),
        make_profile("p4", "compliance regulatory legal research"),
    ]

    ranked = hybrid_search(
        profiles=profiles,
        query="backend developer python",
        semantic_searcher=DepthAwareFakeSemanticSearcher(),
        top_k=4,
        candidate_pool_size=4,
    )

    ranked_ids = [profile.id for profile in ranked]
    assert ranked_ids[:2] == ["p2", "p1"]
    assert "p4" in ranked_ids
