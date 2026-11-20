import csv
from pathlib import Path

from hybrid_retrieval.data_loader import load_profiles_from_csv


def test_load_profiles_from_csv_reads_rows_and_returns_profiles(tmp_path: Path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "id,name,core_skills,secondary_skills,soft_skills,years_of_experience,potential_roles,skill_summary\n"
        "1,Alice,Python (Beginner),SQL (Beginner),Teamwork (Competent),2,Backend Developer,Builds APIs in Python\n"
        "2,Bob,Java (Beginner),,Adaptability (Competent),4,Software Engineer,Builds Java services\n",
        encoding="utf-8",
    )

    profiles = load_profiles_from_csv(csv_path)

    assert len(profiles) == 2
    assert profiles[0].id == "1"
    assert profiles[0].core_skills == ["python"]
    assert profiles[1].potential_roles == ["software engineer"]
