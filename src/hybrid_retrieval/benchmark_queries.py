from dataclasses import dataclass

from .profiles import Profile


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    text: str
    intent_family: str
    description: str

    def is_relevant(self, profile: Profile) -> bool:
        roles = set(profile.potential_roles)
        skills = set(profile.core_skills + profile.secondary_skills + profile.soft_skills)

        if self.query_id == 'P1':
            return 'backend developer' in roles and {'python', 'sql'} <= skills
        if self.query_id == 'P2':
            return 'software engineer' in roles and 'java' in skills
        if self.query_id == 'P3':
            regulatory_role = any('regulatory affairs' in role for role in roles)
            return regulatory_role and 'legal research' in skills
        if self.query_id == 'P4':
            return 'devops engineer intern' in roles and 'python' in skills
        if self.query_id == 'P5':
            return 'supply chain manager' in roles and 'inventory management' in skills
        if self.query_id == 'P6':
            return 'backend developer' in roles and {'python', 'sql'} <= skills
        if self.query_id == 'P7':
            regulatory_role = any('regulatory affairs' in role for role in roles)
            return regulatory_role or bool({'fda', 'legal research'} & skills)
        if self.query_id == 'P8':
            return 'python' in skills and 'teamwork' in ' '.join(profile.soft_skills) and 'adaptability' in ' '.join(profile.soft_skills)
        return False


BENCHMARK_QUERIES = [
    BenchmarkQuery('P1', 'Backend developer profiles with Python and SQL', 'exact/filter', 'Role plus two hard skills.'),
    BenchmarkQuery('P2', 'Software engineer candidates with Java', 'exact/filter', 'Role plus one core skill.'),
    BenchmarkQuery('P3', 'Regulatory affairs profiles with legal research', 'relation-aware', 'Role-to-skill bridge.'),
    BenchmarkQuery('P4', 'DevOps engineer intern candidates with Python', 'constraint-heavy', 'Specific role plus skill.'),
    BenchmarkQuery('P5', 'Supply-chain profiles with inventory management', 'semantic/paraphrase', 'Supply-chain role and domain skill.'),
    BenchmarkQuery('P6', 'Candidates for backend work with databases and Python', 'semantic/paraphrase', 'Natural language version of backend + Python + SQL.'),
    BenchmarkQuery('P7', 'Profiles suitable for compliance or regulatory work', 'semantic/paraphrase', 'Regulatory role family or regulatory skills.'),
    BenchmarkQuery('P8', 'Candidates with teamwork, adaptability, and Python', 'constraint-heavy', 'Soft skills plus one hard skill.'),
]
