from hybrid_retrieval.benchmark_queries import BENCHMARK_QUERIES
from hybrid_retrieval.benchmarking import build_benchmark_rows, summarize_rows
from hybrid_retrieval.profiles import Profile


class FakeSemanticSearcher:
    def search(self, profiles, query, top_k=10):
        return profiles[:top_k]


def make_profile(profile_id: str, role: str, core_skills: list[str], secondary_skills: list[str] | None = None, soft_skills: list[str] | None = None) -> Profile:
    secondary_skills = secondary_skills or []
    soft_skills = soft_skills or []
    return Profile(
        id=profile_id,
        name=profile_id,
        years_of_experience=0.0,
        core_skills=core_skills,
        secondary_skills=secondary_skills,
        soft_skills=soft_skills,
        potential_roles=[role],
        skill_summary=" ".join(core_skills + secondary_skills + soft_skills),
        raw_core_skills=", ".join(core_skills),
        raw_secondary_skills=", ".join(secondary_skills),
        raw_soft_skills=", ".join(soft_skills),
        raw_potential_roles=role,
        search_text=" ".join([role] + core_skills + secondary_skills + soft_skills),
    )


def test_build_benchmark_rows_returns_per_query_metrics_with_intent_family():
    profiles = [
        make_profile("p1", "backend developer", ["python", "sql"]),
        make_profile("p2", "software engineer", ["java"]),
        make_profile("p3", "regulatory affairs associate", ["legal research"]),
        make_profile("p4", "devops engineer intern", ["python"]),
        make_profile("p5", "supply chain manager", ["inventory management"]),
        make_profile("p6", "backend developer", ["python", "sql"]),
        make_profile("p7", "compliance analyst", ["fda"]),
        make_profile("p8", "backend developer", ["python"], soft_skills=["teamwork", "adaptability"]),
    ]

    rows = build_benchmark_rows(profiles, BENCHMARK_QUERIES, FakeSemanticSearcher(), top_k=3, candidate_pool_size=6)

    assert set(rows) == {"lexical", "hybrid"}
    assert len(rows["lexical"]) == len(BENCHMARK_QUERIES)
    assert rows["lexical"][0]["query_id"] == "P1"
    assert rows["lexical"][0]["intent_family"] == "exact/filter"
    assert "precision_at_k" in rows["lexical"][0]


def test_summarize_rows_averages_metric_columns_only():
    rows = [
        {"query_id": "P1", "intent_family": "exact/filter", "precision_at_k": 1.0, "recall_at_k": 0.5, "average_precision": 0.5, "ndcg_at_k": 1.0},
        {"query_id": "P2", "intent_family": "semantic/paraphrase", "precision_at_k": 0.0, "recall_at_k": 0.5, "average_precision": 0.25, "ndcg_at_k": 0.5},
    ]

    summary = summarize_rows(rows)

    assert summary == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "average_precision": 0.375,
        "ndcg_at_k": 0.75,
    }
