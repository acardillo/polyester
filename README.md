# Polyester

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

## Retrieval systems

Each store implements the same interface (`index`, `query`, `clear`, `size`) but uses a different retrieval strategy.

### Vector store

**Semantic search.** Documents and queries are turned into embeddings using ChromaDB's sentence Transformers. Cosine similiarity between vectors is used for ranking.

This store is best used when the query is conceptual differs semantically from the docs. For example, “parse JSON” could match “deserialize a JSON string”

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

| Option    | Shorthand | Values                                                | Default                 | Description              |
| --------- | --------- | ----------------------------------------------------- | ----------------------- | ------------------------ |
| `--store` | `-s`      | `vector` &#124; `graph` &#124; `bm25` &#124; `hybrid` | `vector`                | Store type               |
| `--data`  | `-d`      | File Path                                             | `data/python_docs.json` | Path to JSON data        |
| `--top-k` | `-k`      | Integer                                               | `5`                     | # of result (query only) |

## Project Structure

```
polyester/
├── polyester.py          # CLI entrypoint
├── requirements.txt
├── data/
│   └── python_docs.json  # Python stdlib docs (or run scripts/extract_python_docs.py)
├── scripts/
│   └── extract_python_docs.py
├── src/
│   ├── adapters/         # PythonDocsAdapter
│   ├── core/             # Document, Relationship
│   └── stores/           # VectorStore, GraphStore, BM25Store, HybridStore
└── tests/
```
