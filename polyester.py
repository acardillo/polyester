#!/usr/bin/env python3
"""
Polyester CLI - Command-line interface for polystore retrieval system.
"""

import typer
import os
import warnings
import logging
from rich.console import Console
from rich.table import Table
from pathlib import Path

from src.adapters import PythonDocsAdapter
from src.stores import VectorStore, GraphStore, BM25Store

# Suppress ALL non-critical warnings
warnings.filterwarnings("ignore")

# Suppress specific library loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable transformers progress bars (this is what's showing the load report)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

app = typer.Typer(help="Polyester - Polystore AI Retrieval System")
console = Console()

@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text to search for"),
    store_type: str = typer.Option("vector", "--store", "-s", help="Store type: vector, graph, hybrid"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    data_path: str = typer.Option("data/python_docs.json", "--data", "-d", help="Path to data file")
):
    """Query the retrieval system."""
    console.print(f"[bold blue]Loading data from {data_path}...[/bold blue]")
    
    # Load adapter
    adapter = PythonDocsAdapter(data_path=data_path)
    docs = adapter.load_documents()
    console.print(f"[green]✓[/green] Loaded {len(docs)} documents")
    
    # Create and index store
    store = create_store(store_type)
    store.index(docs)
    console.print(f"[green]✓[/green] Indexed {store.size()} documents")
    
    # Run query
    console.print(f"[bold blue]Searching for:[/bold blue] {query_text}\n")
    results = store.query(query_text, n_results=top_k)
    
    # Display results
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    # Create Rich table
    table = Table(title=f"Top {len(results)} Results", show_header=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("ID", style="magenta")
    table.add_column("Module", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Content Preview", style="white")
    
    for i, doc in enumerate(results, 1):
        content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
        table.add_row(
            str(i),
            doc.id,
            doc.metadata.get("module", "N/A"),
            doc.metadata.get("type", "N/A"),
            content_preview
        )
    
    console.print(table)


@app.command()
def index(
    data_path: str = typer.Option("data/python_docs.json", "--data", "-d", help="Path to data file"),
    store_type: str = typer.Option("vector", "--store", "-s", help="Store type")
):
    """Index documents into a store."""
    console.print(f"[bold blue]Loading data from {data_path}...[/bold blue]")
    
    # Load adapter
    adapter = PythonDocsAdapter(data_path=data_path)
    docs = adapter.load_documents()
    console.print(f"[green]✓[/green] Loaded {len(docs)} documents")
    
    # Create store
    store = create_store(store_type)
    
    # Index documents
    with console.status("[bold green]Generating embeddings..."):
        store.index(docs)
    
    console.print(f"[green]✓[/green] Successfully indexed {store.size()} documents")
    console.print(f"[yellow]Note: In-memory store. Data will be lost when program exits.[/yellow]")


@app.command()
def info(
    store_type: str = typer.Option("vector", "--store", "-s", help="Store type"),
    data_path: str = typer.Option("data/python_docs.json", "--data", "-d", help="Path to data file")
):
    """Show information about indexed documents."""
    console.print(f"[bold blue]Loading data from {data_path}...[/bold blue]")
    
    # Load adapter
    adapter = PythonDocsAdapter(data_path=data_path)
    docs = adapter.load_documents()
    
    # Analyze documents
    total_docs = len(docs)
    modules = set(doc.metadata.get("module", "unknown") for doc in docs)
    types = {}
    for doc in docs:
        doc_type = doc.metadata.get("type", "unknown")
        types[doc_type] = types.get(doc_type, 0) + 1
    
    # Create info table
    table = Table(title="Dataset Information", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Data Source", data_path)
    table.add_row("Total Documents", str(total_docs))
    table.add_row("Unique Modules", str(len(modules)))
    table.add_row("Functions", str(types.get("function", 0)))
    table.add_row("Classes", str(types.get("class", 0)))
    table.add_row("Store Type", store_type)
    
    console.print(table)

def create_store(store_type: str):
    """Factory function to create store by type."""
    if store_type == "vector":
        return VectorStore(collection_name="python_docs_cli")
    elif store_type == "graph":
        return GraphStore()
    elif store_type == "bm25":
        return BM25Store()
    else:
        console.print(f"[red]Error: Store type '{store_type}' not implemented[/red]")
        console.print(f"[yellow]Available: vector, graph[/yellow]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()