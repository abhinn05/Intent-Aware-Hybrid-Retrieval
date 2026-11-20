# Intent-Aware Hybrid Retrieval

A hackathon project for **intent-aware candidate retrieval** that combines:
- **lexical retrieval** for exact and keyword-heavy matches
- **semantic retrieval** for broader meaning-based matching
- **graph retrieval** with Neo4j for relational evidence
- **intent classification** to decide how queries should be handled
- **query reformulation** for exploratory and comparison-style queries
- a simple **FastAPI demo UI** for live search

The goal is straightforward: make retrieval behave differently depending on what the user is actually asking.

---

## What this project does

Given a user query, the system can:

1. **classify the query intent**
   - `exact_lookup`
   - `ambiguous`
   - `exploratory`
   - `comparison`

2. **choose a retrieval strategy** based on that intent
   - exact / ambiguous queries lean conservative
   - exploratory / comparison queries can use broader retrieval

3. **retrieve candidates** using a mix of:
   - BM25-style lexical search
   - sentence-transformer semantic search
   - graph-based retrieval from Neo4j

4. **fuse ranked results** into a final result list

5. **show results and routing details** in the demo UI

---

## Current retrieval behavior

### 1) Exact lookup
Used for direct entity / identifier style requests.

Examples:
- `show candidate 214446`
- `Ankita Rastogi`
- `230862`

Behavior:
- lexical-first behavior
- no unnecessary broadening
- preserves exact-match signal strongly

### 2) Ambiguous
Used for vague or underspecified queries.

Examples:
- `Chen`
- `pipeline`
- `attention`

Behavior:
- **lexical only**
- **no semantic expansion**
- **no rewrite generation**

This was intentional: ambiguous rewrite generation caused garbage hallucinated subqueries, so that path was killed.

### 3) Exploratory
Used for discovery-oriented requests.

Examples:
- `profiles suitable for IT support`
- `software engineers with SQL and teamwork`

Behavior:
- hybrid retrieval
- can use reformulations
- graph retrieval may contribute

### 4) Comparison
Used for side-by-side or tradeoff-style queries.

Examples:
- `postgres vs mongodb`
- `compare candidate 227493 and 214628`

Behavior:
- hybrid retrieval
- can use reformulations
- graph retrieval may contribute

---

## Main components

### Lexical retrieval
Implemented with BM25-style scoring over normalized searchable profile text.

Used for:
- exact lookups
- ambiguous queries
- hybrid fusion

### Semantic retrieval
Implemented with sentence-transformers.

Current default model:
- `sentence-transformers/all-MiniLM-L6-v2`

Used for:
- hybrid retrieval
- exploratory/comparison behavior

### Graph retrieval
Implemented using Neo4j.

Graph model includes:
- Candidate nodes
- Skill nodes
- Role nodes
- candidate-skill relationships
- candidate-role relationships

Used for:
- additional relational evidence
- exploratory/comparison-style retrieval fusion

### Intent classifier
Current runtime classifier uses Ollama with:
- `qwen2.5:1.5b`

Used to route queries into different retrieval policies.

### Query reformulation
Also uses Ollama with:
- `qwen2.5:1.5b`

Used only when broadening is actually useful.

---

## Repository structure

```text
intent-aware-hybrid-retrieval/
├── dataset/                         # main dataset CSV
├── docs/                            # plans / notes
├── graph_retrieval/                 # Neo4j graph build + graph search code
├── scripts/                         # benchmark / startup helper scripts
├── src/hybrid_retrieval/
│   ├── app.py                       # FastAPI app
│   ├── data_loader.py               # CSV loading
│   ├── profiles.py                  # profile normalization
│   ├── search.py                    # lexical / semantic / hybrid retrieval
│   ├── graph_search.py              # graph search wrapper
│   ├── intent_classifier.py         # intent classification
│   ├── reformulation.py             # query reformulation
│   ├── orchestration.py             # intent-aware routing
│   └── static/                      # demo UI assets
├── tests/                           # tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

# Running the project

You have two realistic options:

---

## Option A — Run locally on your machine (recommended for demo reliability)

This avoids Docker weirdness and is the safest live-demo path.

### Prerequisites
You need:
- Python 3.10+
- Ollama
- Neo4j

### 1. Start Ollama
Install if needed:

```bash
brew install --cask ollama
```

Start it:

```bash
open -a Ollama
```

Pull the required model:

```bash
ollama pull qwen2.5:1.5b
```

Check it:

```bash
curl http://localhost:11434/api/tags
```

---

### 2. Start Neo4j
Use either:
- Neo4j Desktop
- or a local Neo4j install

Expected connection:
- `bolt://localhost:7687`

Default username usually:
- `neo4j`

Set / remember your Neo4j password.

Check port:

```bash
nc -vz localhost 7687
```

---

### 3. Create a Python virtual environment
From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt pandas python-dotenv
```

---

### 4. Export environment variables

```bash
export DATASET_PATH=dataset/data.csv
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='YOUR_NEO4J_PASSWORD'
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=qwen2.5:1.5b
```

---

### 5. Build the graph

```bash
cd graph_retrieval
cat > .env <<EOF
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=$NEO4J_PASSWORD
DATA_CSV_PATH=data.csv
EOF

../.venv/bin/python run_graph_pipeline.py build
cd ..
```

If successful, it should print graph stats.

---

### 6. Start the app

```bash
source .venv/bin/activate
PYTHONPATH=src uvicorn hybrid_retrieval.app:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- App: `http://localhost:8000`

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

---

## Option B — Run with Docker Compose

The repo also includes a Compose-based stack.

### What it starts
- FastAPI app
- Neo4j
- graph init job
- Ollama
- Ollama model init job

### Run

```bash
docker compose up --build
```

### Ports
- App: `http://localhost:8000`
- Neo4j Browser: `http://localhost:7474`
- Ollama API: `http://localhost:11434`

### Default credentials
- Neo4j user: `neo4j`
- Neo4j password: `hackathon123`

### Notes
- First boot can take a while because Ollama pulls `qwen2.5:1.5b`
- If Docker behaves weirdly on Mac, local native run is usually more reliable for a live demo

---

## Demo usage

Once the app is running, open:

```text
http://localhost:8000
```

Suggested demo queries:
- `python backend developer with SQL`
- `Ankita Rastogi`
- `230862`
- `Chen`
- `software engineer with sql and teamwork`
- `postgres mongodb comparison`

---

## API endpoints

### Health
```http
GET /health
```

### Lexical search
```http
GET /search/lexical?query=python+backend&top_k=10
```

### Hybrid search
```http
GET /search/hybrid?query=python+backend&top_k=10&candidate_pool_size=50&lexical_weight=1.5&semantic_weight=1.0
```

### Integrated demo search
```http
GET /demo/search?query=python+backend+developer+with+SQL&top_k=10&candidate_pool_size=50&lexical_weight=1.5&semantic_weight=1.0
```

This is the main endpoint used by the UI.

---

## Graph retrieval details

The graph pipeline:
- loads candidate records from CSV
- creates Neo4j nodes for candidates, skills, and roles
- creates skill / role relationships
- searches by matching query terms across skills, roles, and summaries
- scores candidates based on weighted matches

Graph retrieval currently contributes mainly to:
- exploratory queries
- comparison queries
- fallback hybrid behavior

It is intentionally **not** used for exact / ambiguous routing.

---

## Important implementation notes

### Exact-name bug that was fixed
Initially lexical retrieval was weak for exact names and IDs because `search_text` did not include those fields properly.
That has been fixed so exact name / ID queries now have a real lexical signal.

### Ambiguous-query rewrite bug that was fixed
Ambiguous queries were previously generating nonsense reformulations.
That path now deliberately:
- skips reformulation
- skips semantic expansion
- uses lexical-only retrieval

### LLM dependency behavior
If Ollama is unavailable:
- intent classification degrades cleanly
- reformulation degrades cleanly
- the app should still return retrieval results instead of crashing

---

## Testing

Run tests from repo root:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

Useful targeted runs:

```bash
PYTHONPATH=src python -m pytest tests/test_app.py -q
PYTHONPATH=src python -m pytest tests/test_graph_search.py -q
PYTHONPATH=src python -m pytest tests/test_orchestration.py -q
PYTHONPATH=src python -m pytest tests/test_intent_classifier.py -q
PYTHONPATH=src python -m pytest tests/test_reformulation.py -q
```

---

## Benchmarking

Run the benchmark script:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_benchmark.py --dataset-path dataset/data.csv --top-k 10
```

---

## Troubleshooting

### App returns 500 on `/demo/search`
Most likely causes:
- Ollama not running
- Neo4j not reachable
- wrong Neo4j password

### Ollama connection refused
Start Ollama:

```bash
open -a Ollama
ollama list
```

### Neo4j connection refused
Check:

```bash
nc -vz localhost 7687
```

### Graph retrieval not contributing
Rebuild the graph:

```bash
cd graph_retrieval
../.venv/bin/python run_graph_pipeline.py build
```

### Docker weirdness on Mac
If Docker ports / healthchecks get weird, run the project locally instead. It is simpler and safer under demo pressure.

---

## Limitations

This is still a hackathon project, not a polished production system.

Current limitations include:
- retrieval quality still depends on dataset structure
- graph retrieval is useful but not yet deeply explanation-rich
- Ollama-based intent / reformulation adds latency
- schema generalization across arbitrary datasets is still incomplete

---

## Future improvements

Possible next steps:
- better reranking
- richer graph explanations / path evidence
- stronger intent prompts or fine-tuned classifier
- better dataset-agnostic field mapping
- simpler deployment story with a more reliable Compose stack

---

## One-line summary

**Intent-Aware Hybrid Retrieval** is a hackathon search system that adapts candidate retrieval behavior using intent classification, lexical search, semantic search, and graph-based evidence to produce more useful results than a one-size-fits-all retriever.
