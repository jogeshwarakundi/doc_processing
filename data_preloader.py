"""
data_preloader.py

Migrated to use Chroma DB for storing text chunks and embeddings.

Usage:
    python3 data_preloader.py --file inputs/preload.yaml --model <openai-model> [--dry-run]

Notes:
- The script creates one Chroma collection per embedding model.
- Use `--dry-run` to store documents and metadata without calling OpenAI.
"""

import argparse
import yaml
import sys
from db_handler import DBHandler


def ingest_yaml(file_path: str, model: str, db_handle: DBHandler):
    """Read YAML file and process each chunk entry."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read YAML file: {e}")
        raise ValueError(f"Invalid YAML file: {e}")

    if not isinstance(chunks, list):
        print("[ERROR] YAML root must be a list of chunk entries.")
        raise ValueError(f"Invalid YAML entry at index {idx}. Must have 'chunk_data'.")

    for idx, item in enumerate(chunks, start=1):
        if not isinstance(item, dict) or "chunk_data" not in item:
            print(f"[ERROR] Invalid YAML entry at index {idx}. Must have 'chunk_data'.")
            raise ValueError(f"Invalid YAML entry at index {idx}. Must have 'chunk_data'.")

        # allow either 'prompt' or 'parsing_strategy'
        prompt = item.get("prompt")
        parsing_strategy = item.get("parsing_strategy")

        if not prompt and not parsing_strategy:
            print(f"[ERROR] Invalid YAML entry at index {idx} for chunk {item['chunk_data']}. Must have either 'prompt' or 'parsing_strategy'.")
            raise ValueError(f"Invalid YAML entry at index {idx} for chunk {item['chunk_data']}. Must have either 'prompt' or 'parsing_strategy'.")

        print(f"Processing {idx}:")
        print("Chunk:", item["chunk_data"]) 
        if prompt:
            print("Prompt:", prompt)
        if parsing_strategy:
            print("Parsing Strategy:", parsing_strategy)

        store_value = prompt if prompt is not None else parsing_strategy
        db_handle.store_chunk(item["chunk_data"], store_value, model)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preload chunks and prompts into Chroma DB with embeddings.")
    parser.add_argument("--file", default="inputs/preload.yaml", help="Path to YAML file with chunk/prompt pairs.")
    parser.add_argument("--model", default="nomic-embed-text", help="Which embedding model to use (default: nomic-embed-text for local Ollama; use text-embedding-3-small for OpenAI).")
    parser.add_argument("--dry-run", action="store_true", help="Do not call embedding service; store documents without embeddings (for testing).")
    parser.add_argument("--chroma-dir", default="db", help="Local directory for Chroma persistent storage (duckdb+parquet).")
    parser.add_argument("--list-entries", action="store_true", help="List all entries stored in the Chroma DB and exit.")
    parser.add_argument("--query", help="Run a vector query against the Chroma collection for the given model. Provide the query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to return for --query.")

    args = parser.parse_args()

    print(f"Starting preload for {args.file} using embedding model: {args.model}")
    if args.model != "nomic-embed-text":
        print("[INFO] Using non-default embedding model. Ensure appropriate API keys are set.")

    # Initialize DB handler (manages Chroma client)
    handler = DBHandler(args.chroma_dir)

    # If dry-run is requested, temporarily replace handler.embedding_handler.generate_embedding
    # so store_chunk will add documents without embeddings.
    if args.dry_run:
        original_generate = handler.embedding_handler.generate_embedding
        handler.embedding_handler.generate_embedding = lambda text, model=None: None

    # If user only wants to list entries, do that and exit
    if args.list_entries:
        handler.list_entries()
        # restore generate embedding if needed and exit
        if args.dry_run:
            handler.embedding_handler.generate_embedding = original_generate
        sys.exit(0)

    # If user provided a --query string, run the query and print results
    if args.query:
        try:
            hits = handler.query(args.query, args.model, top_k=args.top_k, distance_threshold=0.1)
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            if args.dry_run:
                handler.generate_embedding = original_generate
            sys.exit(1)

        if not hits:
            print("No results.")
        else:
            print(f"Got {len(hits)} hits")
            for i, h in enumerate(hits, start=1):
                print(f"Result #{i} (collection={h.get('collection')}, id={h.get('id')})")
                print(f"  Distance: {h.get('distance')}")
                print(f"  Metadata: {h.get('metadata')}")
                doc = h.get('document')
                if doc:
                    snippet = (doc[:400] + '...') if len(doc) > 400 else doc
                    print(f"  Document snippet: {snippet}")
                emb = h.get('embedding')
                if emb is not None:
                    print(f"  Embedding length: {len(emb)}")
                print("")

        # restore generate embedding and exit
        if args.dry_run:
            handler.embedding_handler.generate_embedding = original_generate
        sys.exit(0)

    ingest_yaml(args.file, args.model, handler)

    if args.dry_run:
        # restore original
        handler.embedding_handler.generate_embedding = original_generate
    # Try to persist the Chroma client if available (some clients support persist())
    try:
        if handler is not None and handler.persist():
            print(f"[INFO] Chroma embeddings persisted to {args.chroma_dir}")
    except Exception as e:
        print(f"[WARN] Failed to persist Chroma client: {e}")
    print("[INFO] Using local Ollama embeddings (nomic-embed-text) - no API costs!")
