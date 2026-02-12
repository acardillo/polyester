# Polyester

**Polystore AI Retrieval System** — A comparative study of AI memory architectures: vector, graph, BM25, and hybrid retrieval over structured documentation (e.g. Python stdlib).

---

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- ~2GB free disk for embedding models on first run (sentence-transformers)

---

## Quick start

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd polyester
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

Use the included Python stdlib docs, or regenerate them:

```bash
# Optional: regenerate Python stdlib docs (takes a few minutes)
python scripts/extract_python_docs.py
```

Output: `data/python_docs.json` (created by the script or already in the repo).

### 3. Run the CLI

**Query** (default: vector store, 5 results):

```bash
python polyester.py query "JSON parsing"
python polyester.py query "file I/O" --store hybrid --top-k 3
```

**Index** (build in-memory index):

```bash
python polyester.py index --data data/python_docs.json --store vector
```

**Info** (dataset and store summary):

```bash
python polyester.py info --data data/python_docs.json --store hybrid
```

---

## Retrieval systems

Each store implements the same interface (`index`, `query`, `clear`, `size`) but uses a different retrieval strategy.

### Vector store

**Semantic search.** Documents and the query are turned into embeddings (sentence-transformers); retrieval is by cosine similarity in vector space. Best when the query is conceptual and wording differs from the docs (e.g. “parse JSON” matching “deserialize a JSON string”). Uses ChromaDB and the `all-MiniLM-L6-v2` model.

### Graph store

**Structure + keyword.** Documents are nodes; relationships (e.g. “calls”, “inherits”) are edges. Retrieval uses exact ID match, a simple keyword index over content, and neighbor expansion. Best for “what does X call?” or “what inherits from X?”. Uses NetworkX. Does not use embeddings.

### BM25 store

**Keyword ranking.** Classic IR: documents are tokenized; retrieval uses the BM25 formula (term frequency, inverse document frequency, length normalization). Best for exact or strong term overlap (e.g. “JSON load”). Pure keyword, no embeddings or graph. Uses the `rank_bm25` library.

### Hybrid store

**Fusion of all three.** Runs vector, graph, and BM25 in parallel, then merges their ranked lists with **weighted Reciprocal Rank Fusion (RRF)**. Default weights: vector 0.4, graph 0.3, BM25 0.3. Good when you want both semantic match and structural/keyword signal. Single `query()` call; no separate “strategy” options.

---

## Store types (CLI)

| Store   | CLI value | Description |
|--------|-----------|-------------|
| Vector | `vector`  | Semantic search (ChromaDB + embeddings). |
| Graph  | `graph`   | Relationship traversal + keyword (NetworkX). |
| BM25   | `bm25`    | Keyword ranking (BM25). |
| Hybrid | `hybrid`  | Vector + graph + BM25 with weighted RRF. |

Use `--store` / `-s`: e.g. `--store hybrid`.

---

## CLI options

**query** `TEXT` — Run a search.

- `--store`, `-s` — `vector` \| `graph` \| `bm25` \| `hybrid` (default: `vector`).
- `--top-k`, `-k` — Number of results (default: 5).
- `--data`, `-d` — Path to JSON data (default: `data/python_docs.json`).

**index** — Load data and index into the chosen store (in-memory).

- `--store`, `--data` — Same as above.

**info** — Show document count, modules, and store type.

- `--store`, `--data` — Same as above.

---

## Project layout

```
polyester/
├── polyester.py          # CLI entrypoint
├── requirements.txt
├── data/
│   └── python_docs.json  # Python stdlib docs (or run scripts/extract_python_docs.py)
├── scripts/
│   └── extract_python_docs.py
├── src/
│   ├── adapters/         # Data loaders (e.g. PythonDocsAdapter)
│   ├── core/             # Document, Relationship models
│   └── stores/           # VectorStore, GraphStore, BM25Store, HybridStore
└── tests/
```

---

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

With coverage (if `pytest-cov` is installed):

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Status

In development. Implemented: Python stdlib adapter, vector/graph/BM25/hybrid stores, CLI (query, index, info).
