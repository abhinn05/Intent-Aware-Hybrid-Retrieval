from hybrid_retrieval.profiles import Profile
from hybrid_retrieval.search import SearchResult, search_lexical


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


def test_search_lexical_prefers_profiles_with_more_query_overlap():
    profiles = [
        make_profile("p1", "backend developer python sql apis"),
        make_profile("p2", "software engineer java spring"),
        make_profile("p3", "backend developer python"),
    ]

    results = search_lexical(profiles, "backend developer python sql", top_k=3)

    assert [result.profile.id for result in results[:2]] == ["p1", "p3"]


def test_search_lexical_can_match_exact_name_and_identifier_tokens():
    profiles = [
        make_profile("230862", "230862 priyaa guptaa backend developer python sql"),
        make_profile("p2", "software engineer java spring"),
    ]

    name_results = search_lexical(profiles, "Priyaa Guptaa", top_k=2)
    id_results = search_lexical(profiles, "230862", top_k=2)

    assert name_results[0].profile.id == "230862"
    assert id_results[0].profile.id == "230862"


def test_search_lexical_returns_result_scores_as_floats():
    profiles = [
        make_profile("p1", "python backend developer"),
        make_profile("p2", "java engineer"),
    ]

    results = search_lexical(profiles, "python backend", top_k=2)

    assert isinstance(results[0], SearchResult)
    assert isinstance(results[0].score, float)


def test_reciprocal_rank_fusion_rewards_profiles_appearing_in_multiple_rank_lists():
    from hybrid_retrieval.search import reciprocal_rank_fusion

    rank_lists = [["p2", "p1", "p3"], ["p2", "p1", "p4"]]

    fused = reciprocal_rank_fusion(rank_lists, k=60)

    assert fused[0] == "p2"
    assert fused[1] == "p1"
