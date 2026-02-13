# Polyester

[![CI](https://github.com/acardillo/polyester/actions/workflows/ci.yml/badge.svg)](https://github.com/acardillo/polyester/actions/workflows/ci.yml)

**Polystore AI Retrieval System** — A comparative study of AI memory architectures: vector, graph, BM25, and hybrid retrieval over structured documentation (e.g. Python stdlib).

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- ~2GB free disk for embedding models on first run

## Quick start

### 1. Setup Virtual Environment & Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a Data Source

Use the included Python stdlib docs, or optionally regenerate them using:

```bash
python scripts/extract_python_docs.py
```

### 3. Run the Polyester CLI

A full list of CLI options is available in the [CLI options section](#cli-options) below. Here are some sample commands to get you started:

```bash
# Default Query - Top 5 results using Vector Store
python polyester.py query "JSON parsing"

# Custom Query Options - Top 3 results using Graph Store
python polyester.py query "file I/O" --store hybrid --top-k 3

# Index - Creates a vector store from the default data source
python polyester.py index --data data/python_docs.json --store vector

# Info - Provides metrics about the hybrid store
python polyester.py info --data data/python_docs.json --store hybrid
```

### 4. Run Benchmarks (optional)

Compare all stores on speed and accuracy over the Python docs benchmark set:

```bash
python -m scripts.run_python_benchmarks
```

Results are written to **`RESULTS.md`** in the project root (key findings plus tables for average, semantic, keyword, and structural query categories).

## Retrieval systems

Each store implements the same interface (`index`, `query`, `clear`, `size`) but uses a different retrieval strategy.

### Vector store

**Semantic search.** Documents and queries are turned into embeddings using ChromaDB's sentence Transformers. Cosine similarity between vectors is used for ranking.

This store is best used when the query differs semantically from the docs. For example, “parse JSON” could match “deserialize a JSON string”

### Graph store

**Structure + keyword.** Data is expressed as a structured graph (NetworkX) where documents are nodes and relationships are edges. Retrieval uses an inverted index to find key word matches, and neighbor expansion to find related documents.

This store is best for queries that require relational context. For example, “what does X call?” or “what inherits from X?”.

### BM25 store

**Keyword ranking.** Matches documents directly from keywords using a well-defined ranking function (BM25). Documents are tokenized and term frequency, inverse document frequency, and length normalization measure similarity.

This store is best for exact or strong term overlap. For example "JSON load".

### Hybrid store

**Fusion of all three.** Runs vector, graph, and BM25 in parallel, then merges their ranked lists with **weighted Reciprocal Rank Fusion (RRF)**.

## Polyester CLI

To run the Polyester CLI: `python polyester.py <command> [options]`

### Commands

| Command | Syntax                                       | Description                                                           |
| ------- | -------------------------------------------- | --------------------------------------------------------------------- |
| `query` | `python polyester.py query "Text" [options]` | Search the indexed data for relevant results matching the input text. |
| `index` | `python polyester.py index [options]`        | Load data and index it into the chosen store (in-memory only).        |
| `info`  | `python polyester.py info [options]`         | Display metrics and information about the indexed dataset and store.  |

### Options

| Option  | Short | Values                                                | Default                 | Description                    |
| ------- | ----- | ----------------------------------------------------- | ----------------------- | ------------------------------ |
| `store` | `s`   | `vector` &#124; `graph` &#124; `bm25` &#124; `hybrid` | `vector`                | Store type                     |
| `data`  | `d`   | File Path                                             | `data/python_docs.json` | Path to JSON data              |
| `top-k` | `k`   | Integer                                               | `5`                     | Number of results (query only) |

## Development

From the project root with your venv activated:

```bash
# Run tests
pytest

# Lint
ruff check .

# Format
black .
```

CI runs tests, Ruff, and Black on push and pull requests to `main`.

## Project Structure

```
polyester/
├── polyester.py          # CLI entrypoint
├── requirements.txt
├── RESULTS.md            # Benchmark results
├── data/
│   ├── python_docs.json          # Python stdlib docs (or run scripts/extract_python_docs.py)
│   └── python_benchmarks.json    # Benchmark Queries for Python docs
├── scripts/
│   ├── extract_python_docs.py
│   └── run_python_benchmarks.py
├── src/
│   ├── adapters/         # PythonDocsAdapter
│   ├── classifiers/      # Structural intent classifier (for Graph Store)
│   ├── core/             # Document, Relationship
│   └── stores/           # VectorStore, GraphStore, BM25Store, HybridStore
└── tests/
```
