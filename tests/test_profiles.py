from hybrid_retrieval.profiles import parse_skill_field, profile_from_row


def test_parse_skill_field_removes_proficiency_suffixes_and_blanks():
    raw = "Python (Advanced Beginner), SQL (Beginner), , Azure DevOps (Beginner)"

    skills = parse_skill_field(raw)

    assert skills == ["python", "sql", "azure devops"]


def test_profile_from_row_builds_search_text_from_name_id_roles_skills_and_summary():
    row = {
        "id": "181457",
        "name": "itd itd",
        "core_skills": "Python (Advanced Beginner), SQL (Advanced Beginner), Azure DevOps (Beginner)",
        "secondary_skills": "CI/CD (Beginner), Cloud (Beginner)",
        "soft_skills": "Teamwork and collaboration (Competent), Time Management (Competent), Adaptability (Competent)",
        "years_of_experience": "0.0",
        "potential_roles": "Software Engineer, Backend Developer, DevOps Engineer Intern",
        "skill_summary": "This candidate is a fresher with foundational knowledge in Python and SQL.",
    }

    profile = profile_from_row(row)

    assert profile.id == "181457"
    assert profile.core_skills == ["python", "sql", "azure devops"]
    assert profile.potential_roles == ["software engineer", "backend developer", "devops engineer intern"]
    assert "181457" in profile.search_text
    assert "itd itd" in profile.search_text.lower()
    assert "backend developer" in profile.search_text
    assert "python" in profile.search_text
    assert "foundational knowledge in python and sql" in profile.search_text.lower()
