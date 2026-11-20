from hybrid_retrieval.graph_search import GraphSearcher


def test_graph_searcher_prefers_environment_over_env_file(tmp_path, monkeypatch):
    graph_dir = tmp_path / "graph_retrieval"
    graph_dir.mkdir()
    (graph_dir / ".env").write_text("NEO4J_URI=bolt://from-file:7687\nNEO4J_USER=file-user\nNEO4J_PASSWORD=file-pass\n")
    (graph_dir / "graph_pipeline.py").write_text(
        "def search_graph(query, neo4j_uri, neo4j_user, neo4j_password, top_k=10):\n"
        "    return [{\"id\": neo4j_uri, \"name\": neo4j_user, \"score\": neo4j_password}]\n"
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://from-env:7687")
    monkeypatch.setenv("NEO4J_USER", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-pass")

    searcher = GraphSearcher(graph_dir=graph_dir)

    results = searcher.search("sql", top_k=3)

    assert results == [{"id": "bolt://from-env:7687", "name": "env-user", "score": "env-pass"}]


def test_graph_searcher_loads_module_with_future_annotations_and_dataclass(tmp_path):
    graph_dir = tmp_path / "graph_retrieval"
    graph_dir.mkdir()
    (graph_dir / ".env").write_text("NEO4J_PASSWORD=test-secret\n")
    (graph_dir / "graph_pipeline.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Row:\n"
        "    id: str\n\n"
        "def search_graph(query, neo4j_uri, neo4j_user, neo4j_password, top_k=10):\n"
        "    return [{\"id\": \"row-1\", \"score\": 1.0}]\n"
    )

    searcher = GraphSearcher(graph_dir=graph_dir)

    results = searcher.search("sql", top_k=3)

    assert results[0]["id"] == "row-1"
