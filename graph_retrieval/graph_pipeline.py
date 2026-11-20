"""
Knowledge Graph Pipeline for Candidate Search
================================================
Builds a Neo4j property-graph from the candidate CSV, then exposes a
Cypher-powered search that returns a *ranked* list of matching rows.

Graph model  (following https://neo4j.com/blog/knowledge-graph/how-to-build-knowledge-graph/):

  Nodes
  ─────
  (:Candidate {id, name, years_of_experience, skill_summary})
  (:Skill      {name, category})           -- category ∈ {core, secondary, soft}
  (:Role       {name})

  Relationships
  ─────────────
  (Candidate)-[:HAS_SKILL {level, category}]->(Skill)
  (Candidate)-[:SUITED_FOR]->(Role)

Search strategy
───────────────
Given a free-text query we:
  1. Tokenise the query into keywords.
  2. Match keywords against Skill names, Role names, and Candidate summaries
     using case-insensitive CONTAINS.
  3. Score each candidate by counting:
       • +3  per core-skill keyword match
       • +2  per secondary-skill keyword match
       • +1  per soft-skill keyword match
       • +2  per role keyword match
       • +1  per summary keyword match
  4. Return candidates ordered by descending score.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from neo4j import GraphDatabase


# ── helpers ──────────────────────────────────────────────────────────────────

_SKILL_LEVEL_RE = re.compile(
    r"^(.+?)\s*\(([^)]+)\)\s*$"
)  # "Python (Competent)" → ("Python", "Competent")


def _is_missing(raw: object) -> bool:
    return raw is None or (isinstance(raw, float) and raw != raw)


def _parse_skill_list(raw: str) -> List[Tuple[str, str]]:
    """Return [(skill_name, proficiency_level), …] from a comma-separated cell."""
    if _is_missing(raw) or not str(raw).strip():
        return []
    results: List[Tuple[str, str]] = []
    # Skills are comma-separated but some skill names contain commas inside
    # parentheses.  We split *outside* parentheses only.
    parts = re.split(r",\s*(?=[A-Za-z])", raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _SKILL_LEVEL_RE.match(part)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
        else:
            results.append((part, "Unknown"))
    return results


def _parse_roles(raw: str) -> List[str]:
    """Return list of role names from a comma-separated cell."""
    if _is_missing(raw) or not str(raw).strip():
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


# ── data-loading ─────────────────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    id: int
    name: str
    core_skills: List[Tuple[str, str]]
    secondary_skills: List[Tuple[str, str]]
    soft_skills: List[Tuple[str, str]]
    years_of_experience: float
    potential_roles: List[str]
    skill_summary: str


def load_candidates(csv_path: str) -> List[CandidateRecord]:
    """Read the CSV into a list of structured CandidateRecords."""
    import pandas as pd

    df = pd.read_csv(csv_path, dtype={"id": int})
    records: List[CandidateRecord] = []
    for _, row in df.iterrows():
        records.append(
            CandidateRecord(
                id=int(row["id"]),
                name=str(row.get("name", "")).strip(),
                core_skills=_parse_skill_list(row.get("core_skills", "")),
                secondary_skills=_parse_skill_list(row.get("secondary_skills", "")),
                soft_skills=_parse_skill_list(row.get("soft_skills", "")),
                years_of_experience=float(row.get("years_of_experience", 0)),
                potential_roles=_parse_roles(row.get("potential_roles", "")),
                skill_summary=str(row.get("skill_summary", "")),
            )
        )
    return records


# ── Neo4j graph builder ─────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """Creates the knowledge graph inside Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # -- schema --
    def create_constraints_and_indexes(self):
        """Create uniqueness constraints and full-text indexes."""
        queries = [
            "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT role_name IF NOT EXISTS FOR (r:Role) REQUIRE r.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception:
                    pass  # constraint may already exist

    # -- ingestion --
    def clear_graph(self):
        """Remove all nodes and relationships (use with care)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def ingest_candidates(self, candidates: List[CandidateRecord], batch_size: int = 200):
        """Batch-ingest candidates, skills, roles and their relationships."""
        with self.driver.session() as session:
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                self._ingest_batch(session, batch)

    def _ingest_batch(self, session, batch: List[CandidateRecord]):
        """Ingest a single batch inside one transaction."""
        for c in batch:
            # 1. Create or merge Candidate node
            session.run(
                """
                MERGE (cand:Candidate {id: $id})
                SET cand.name = $name,
                    cand.years_of_experience = $yoe,
                    cand.skill_summary = $summary
                """,
                id=c.id, name=c.name, yoe=c.years_of_experience,
                summary=c.skill_summary,
            )

            # 2. Core skills
            for skill_name, level in c.core_skills:
                session.run(
                    """
                    MERGE (s:Skill {name: $skill_name})
                    ON CREATE SET s.category = 'core'
                    WITH s
                    MATCH (cand:Candidate {id: $cid})
                    MERGE (cand)-[r:HAS_SKILL]->(s)
                    SET r.level = $level, r.category = 'core'
                    """,
                    skill_name=skill_name, level=level, cid=c.id,
                )

            # 3. Secondary skills
            for skill_name, level in c.secondary_skills:
                session.run(
                    """
                    MERGE (s:Skill {name: $skill_name})
                    ON CREATE SET s.category = 'secondary'
                    WITH s
                    MATCH (cand:Candidate {id: $cid})
                    MERGE (cand)-[r:HAS_SKILL]->(s)
                    SET r.level = $level, r.category = 'secondary'
                    """,
                    skill_name=skill_name, level=level, cid=c.id,
                )

            # 4. Soft skills
            for skill_name, level in c.soft_skills:
                session.run(
                    """
                    MERGE (s:Skill {name: $skill_name})
                    ON CREATE SET s.category = 'soft'
                    WITH s
                    MATCH (cand:Candidate {id: $cid})
                    MERGE (cand)-[r:HAS_SKILL]->(s)
                    SET r.level = $level, r.category = 'soft'
                    """,
                    skill_name=skill_name, level=level, cid=c.id,
                )

            # 5. Potential roles
            for role_name in c.potential_roles:
                session.run(
                    """
                    MERGE (r:Role {name: $role_name})
                    WITH r
                    MATCH (cand:Candidate {id: $cid})
                    MERGE (cand)-[:SUITED_FOR]->(r)
                    """,
                    role_name=role_name, cid=c.id,
                )

    def get_stats(self) -> Dict[str, int]:
        """Return node/relationship counts."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Candidate) WITH count(c) AS candidates
                MATCH (s:Skill)     WITH candidates, count(s) AS skills
                MATCH (r:Role)      WITH candidates, skills, count(r) AS roles
                MATCH ()-[rel]->()  WITH candidates, skills, roles, count(rel) AS rels
                RETURN candidates, skills, roles, rels
                """
            )
            row = result.single()
            return {
                "candidates": row["candidates"],
                "skills": row["skills"],
                "roles": row["roles"],
                "relationships": row["rels"],
            }


# ── Graph Search Engine ──────────────────────────────────────────────────────

class GraphSearchEngine:
    """
    Searches the Neo4j knowledge graph and returns a ranked list of
    candidate rows for a given free-text query.

    Scoring weights:
        core skill match       → 3
        secondary skill match  → 2
        soft skill match       → 1
        role match             → 2
        summary keyword match  → 1
    """

    CORE_WEIGHT = 3
    SECONDARY_WEIGHT = 2
    SOFT_WEIGHT = 1
    ROLE_WEIGHT = 2
    SUMMARY_WEIGHT = 1

    # Boost for proficiency level
    LEVEL_BOOST = {
        "Proficient": 1.5,
        "Competent": 1.2,
        "Advanced Beginner": 1.0,
        "Beginner": 0.8,
    }

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ── public API ──

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Dict]:
        """
        Run a graph-based search for *query* and return up to *top_k* ranked
        candidate rows.

        Each result dict contains:
            id, name, years_of_experience, skill_summary,
            matched_skills, matched_roles, score, explanation
        """
        keywords = self._tokenize(query)
        if not keywords:
            return []

        with self.driver.session() as session:
            results = self._graph_search(session, keywords, top_k)

        return results

    # ── internal ──

    @staticmethod
    def _tokenize(query: str) -> List[str]:
        """Split query into lowercase keyword tokens (≥2 chars)."""
        tokens = re.split(r"[,;\s]+", query.strip().lower())
        # Remove very short / stop-word-like tokens
        stop = {"a", "an", "the", "is", "in", "of", "for", "and", "or", "to", "with"}
        return [t for t in tokens if len(t) >= 2 and t not in stop]

    def _graph_search(
        self, session, keywords: List[str], top_k: int
    ) -> List[Dict]:
        """
        Execute a Cypher query that:
          1. Finds all Skills / Roles whose names contain any keyword.
          2. Traverses relationships back to Candidates.
          3. Also checks candidate skill_summary for keyword presence.
          4. Aggregates a weighted score per candidate.
          5. Returns top-k results in descending score order.
        """

        # Build a Cypher query that uses UNWIND to check each keyword
        cypher = """
        // Parameters: $keywords (list of lowercase strings), $top_k (int)

        // ── Step 1: Match skills whose name contains any keyword ──
        UNWIND $keywords AS kw
        OPTIONAL MATCH (s:Skill)
            WHERE toLower(s.name) CONTAINS kw
        WITH kw, collect(DISTINCT s) AS matched_skills

        // ── Step 2: For each matched skill, find connected candidates ──
        UNWIND matched_skills AS ms
        OPTIONAL MATCH (c:Candidate)-[r:HAS_SKILL]->(ms)
        WITH kw, c, r, ms
        WHERE c IS NOT NULL

        // Compute per-link weight
        WITH kw, c,
             ms.name AS skill_name,
             r.category AS cat,
             r.level AS lvl,
             CASE r.category
                 WHEN 'core'      THEN $core_w
                 WHEN 'secondary' THEN $sec_w
                 WHEN 'soft'      THEN $soft_w
                 ELSE 1
             END AS base_weight

        WITH kw, c, skill_name, cat, lvl, base_weight,
             CASE lvl
                 WHEN 'Proficient'       THEN 1.5
                 WHEN 'Competent'        THEN 1.2
                 WHEN 'Advanced Beginner' THEN 1.0
                 ELSE 0.8
             END AS level_boost

        WITH c,
             collect(DISTINCT skill_name) AS skill_matches,
             sum(base_weight * level_boost) AS skill_score

        // ── Step 3: role matches ──
        WITH c, skill_matches, skill_score
        UNWIND $keywords AS kw2
        OPTIONAL MATCH (c)-[:SUITED_FOR]->(rl:Role)
            WHERE toLower(rl.name) CONTAINS kw2
        WITH c, skill_matches, skill_score,
             collect(DISTINCT rl.name) AS role_matches,
             count(DISTINCT rl) * $role_w AS role_score

        // ── Step 4: summary keyword match ──
        WITH c, skill_matches, skill_score, role_matches, role_score
        UNWIND $keywords AS kw3
        WITH c, skill_matches, skill_score, role_matches, role_score,
             CASE WHEN toLower(c.skill_summary) CONTAINS kw3 THEN 1 ELSE 0 END AS sw
        WITH c, skill_matches, skill_score, role_matches, role_score,
             sum(sw) * $sum_w AS summary_score

        // ── Step 5: total score & return ──
        WITH c,
             skill_matches,
             role_matches,
             skill_score + role_score + summary_score AS total_score

        WHERE total_score > 0
        RETURN c.id            AS id,
               c.name          AS name,
               c.years_of_experience AS years_of_experience,
               c.skill_summary AS skill_summary,
               skill_matches,
               role_matches,
               round(total_score * 100) / 100 AS score
        ORDER BY total_score DESC
        LIMIT $top_k
        """

        result = session.run(
            cypher,
            keywords=keywords,
            top_k=top_k,
            core_w=self.CORE_WEIGHT,
            sec_w=self.SECONDARY_WEIGHT,
            soft_w=self.SOFT_WEIGHT,
            role_w=self.ROLE_WEIGHT,
            sum_w=self.SUMMARY_WEIGHT,
        )

        rows = []
        for rec in result:
            explanation_parts = []
            if rec["skill_matches"]:
                explanation_parts.append(
                    f"Matched skills: {', '.join(rec['skill_matches'])}"
                )
            if rec["role_matches"]:
                explanation_parts.append(
                    f"Matched roles: {', '.join(rec['role_matches'])}"
                )
            explanation = "; ".join(explanation_parts) if explanation_parts else "Matched via skill summary keywords"

            rows.append({
                "id": rec["id"],
                "name": rec["name"],
                "years_of_experience": rec["years_of_experience"],
                "skill_summary": rec["skill_summary"],
                "matched_skills": rec["skill_matches"],
                "matched_roles": rec["role_matches"],
                "score": rec["score"],
                "explanation": explanation,
            })

        return rows


# ── Convenience wrappers ─────────────────────────────────────────────────────

def build_graph(
    csv_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_existing: bool = True,
) -> Dict[str, int]:
    """
    End-to-end: load CSV → build graph → return stats.
    """
    print(f"[1/4] Loading candidates from {csv_path} …")
    candidates = load_candidates(csv_path)
    print(f"       Loaded {len(candidates)} candidates.")

    builder = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    try:
        print("[2/4] Creating constraints & indexes …")
        builder.create_constraints_and_indexes()

        if clear_existing:
            print("[3/4] Clearing existing graph data …")
            builder.clear_graph()
        else:
            print("[3/4] Keeping existing graph data (append mode).")

        print("[4/4] Ingesting candidates into Neo4j …")
        builder.ingest_candidates(candidates)

        stats = builder.get_stats()
        print(f"       Done! Graph stats: {stats}")
        return stats
    finally:
        builder.close()


def search_graph(
    query: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    top_k: int = 10,
) -> List[Dict]:
    """
    Search the knowledge graph and return ranked results.
    """
    engine = GraphSearchEngine(neo4j_uri, neo4j_user, neo4j_password)
    try:
        return engine.search(query, top_k)
    finally:
        engine.close()
