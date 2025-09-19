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
    parser.add_argument("--model", default="text-embedding-3-small", help="Which OpenAI embedding model to use.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenAI; store documents without embeddings (for testing).")
    parser.add_argument("--chroma-dir", default="db", help="Local directory for Chroma persistent storage (duckdb+parquet).")
    parser.add_argument("--list-entries", action="store_true", help="List all entries stored in the Chroma DB and exit.")

    args = parser.parse_args()

    print(f"Starting preload for {args.file} and {args.model}")

    # Initialize DB handler (manages Chroma client)
    handler = DBHandler(args.chroma_dir)

    # If dry-run is requested, temporarily replace handler.generate_embedding
    # so store_chunk will add documents without embeddings.
    if args.dry_run:
        original_generate = handler.generate_embedding
        handler.generate_embedding = lambda model, text: None

    # If user only wants to list entries, do that and exit
    if args.list_entries:
        handler.list_entries()
        # restore generate embedding if needed and exit
        if args.dry_run:
            handler.generate_embedding = original_generate
        sys.exit(0)

    ingest_yaml(args.file, args.model, handler)

    if args.dry_run:
        # restore original
        handler.generate_embedding = original_generate
    # Try to persist the Chroma client if available (some clients support persist())
    try:
        if handler is not None and handler.persist():
            print(f"[INFO] Chroma persisted to {args.chroma_dir}")
    except Exception as e:
        print(f"[WARN] Failed to persist Chroma client: {e}")
