from dataclasses import dataclass
import re
from typing import Any


_PROFICIENCY_SUFFIX = re.compile(r"\s*\([^)]*\)\s*")


@dataclass(slots=True)
class Profile:
    id: str
    name: str
    years_of_experience: float
    core_skills: list[str]
    secondary_skills: list[str]
    soft_skills: list[str]
    potential_roles: list[str]
    skill_summary: str
    raw_core_skills: str
    raw_secondary_skills: str
    raw_soft_skills: str
    raw_potential_roles: str
    search_text: str


def parse_skill_field(raw_value: str) -> list[str]:
    cleaned_items: list[str] = []
    for item in raw_value.split(','):
        item = _PROFICIENCY_SUFFIX.sub('', item).strip().lower()
        if item:
            cleaned_items.append(item)
    return cleaned_items


def _parse_role_field(raw_value: str) -> list[str]:
    return [item.strip().lower() for item in raw_value.split(',') if item.strip()]


def _to_float(raw_value: Any) -> float:
    try:
        return float(raw_value or 0)
    except (TypeError, ValueError):
        return 0.0


def profile_from_row(row: dict[str, Any]) -> Profile:
    profile_id = str(row.get('id', '')).strip()
    name = (row.get('name') or '').strip()
    core_skills = parse_skill_field(row.get('core_skills', ''))
    secondary_skills = parse_skill_field(row.get('secondary_skills', ''))
    soft_skills = parse_skill_field(row.get('soft_skills', ''))
    potential_roles = _parse_role_field(row.get('potential_roles', ''))
    skill_summary = (row.get('skill_summary') or '').strip()

    search_parts = [
        profile_id,
        name.lower(),
        ' '.join(potential_roles),
        ' '.join(core_skills),
        ' '.join(secondary_skills),
        ' '.join(soft_skills),
        skill_summary,
    ]
    search_text = ' '.join(part for part in search_parts if part).strip()

    return Profile(
        id=profile_id,
        name=name,
        years_of_experience=_to_float(row.get('years_of_experience')),
        core_skills=core_skills,
        secondary_skills=secondary_skills,
        soft_skills=soft_skills,
        potential_roles=potential_roles,
        skill_summary=skill_summary,
        raw_core_skills=row.get('core_skills', ''),
        raw_secondary_skills=row.get('secondary_skills', ''),
        raw_soft_skills=row.get('soft_skills', ''),
        raw_potential_roles=row.get('potential_roles', ''),
        search_text=search_text,
    )
