from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


class GraphSearchUnavailable(RuntimeError):
    pass


class GraphSearcher:
    backend_name = 'neo4j'

    def __init__(self, graph_dir: Path | None = None, env_path: Path | None = None):
        self.graph_dir = Path(graph_dir) if graph_dir is not None else Path(__file__).resolve().parents[2] / 'graph_retrieval'
        self.env_path = Path(env_path) if env_path is not None else self.graph_dir / '.env'
        self._module = None

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        config = self._config()
        module = self._load_module()
        search_graph = getattr(module, 'search_graph', None)
        if search_graph is None:
            raise GraphSearchUnavailable(f'graph search entrypoint missing in {self.graph_dir / "graph_pipeline.py"}')
        return search_graph(
            query=query,
            neo4j_uri=config['NEO4J_URI'],
            neo4j_user=config['NEO4J_USER'],
            neo4j_password=config['NEO4J_PASSWORD'],
            top_k=top_k,
        )

    def _config(self) -> dict[str, str]:
        env = _read_env_file(self.env_path)
        env.update({key: value for key, value in os.environ.items() if value})
        password = env.get('NEO4J_PASSWORD', '').strip()
        if not password:
            raise GraphSearchUnavailable(
                f'NEO4J_PASSWORD is missing. Configure it in {self.env_path} or the environment before using graph retrieval.'
            )
        return {
            'NEO4J_URI': env.get('NEO4J_URI', 'bolt://localhost:7687').strip() or 'bolt://localhost:7687',
            'NEO4J_USER': env.get('NEO4J_USER', 'neo4j').strip() or 'neo4j',
            'NEO4J_PASSWORD': password,
        }

    def _load_module(self):
        if self._module is not None:
            return self._module
        module_path = self.graph_dir / 'graph_pipeline.py'
        if not module_path.exists():
            raise GraphSearchUnavailable(f'graph pipeline not found at {module_path}')
        spec = importlib.util.spec_from_file_location('hybrid_retrieval_graph_pipeline', module_path)
        if spec is None or spec.loader is None:
            raise GraphSearchUnavailable(f'could not load graph pipeline from {module_path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._module = module
        return module


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
