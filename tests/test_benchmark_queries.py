from hybrid_retrieval.benchmark_queries import BENCHMARK_QUERIES
from hybrid_retrieval.profiles import profile_from_row


def make_profile(**overrides):
    row = {
        "id": "1",
        "name": "Alice",
        "core_skills": "Python (Beginner), SQL (Beginner)",
        "secondary_skills": "",
        "soft_skills": "Teamwork (Competent), Adaptability (Competent)",
        "years_of_experience": "2",
        "potential_roles": "Backend Developer, Software Engineer",
        "skill_summary": "Backend profile with Python and SQL experience.",
    }
    row.update(overrides)
    return profile_from_row(row)


def test_backend_python_sql_query_matches_expected_profile():
    profile = make_profile()

    assert BENCHMARK_QUERIES[0].is_relevant(profile)


def test_devops_intern_query_rejects_non_matching_role():
    profile = make_profile(potential_roles="Backend Developer", core_skills="Python (Beginner)")

    assert not BENCHMARK_QUERIES[3].is_relevant(profile)
