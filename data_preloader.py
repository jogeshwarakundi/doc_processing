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
import hashlib
import yaml
import openai
import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings


# -----------------------------
# Chroma Setup (initialized in main)
# -----------------------------
# We will initialize a persistent Chroma client (duckdb+parquet) in `main`
# using a directory provided by `--chroma-dir`. This avoids regenerating
# embeddings between runs.
chroma_client = None

def init_chroma(persist_directory: str = "db"):
    """Initialize the global Chroma client with a persistent directory."""
    global chroma_client
    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        return
    except Exception:
        # Some chromadb versions expose a different Client signature or
        # deprecated config behavior. Fall back to default client and warn.
        print("[WARN] Chroma settings-based initialization failed; falling back to default Client. Persistence may not be available with this chromadb version.")
        chroma_client = chromadb.Client()
        return


# -----------------------------
# Utility Functions
# -----------------------------
def compute_chunk_id(chunk_data: str) -> str:
    """Generate a SHA256 hash for the chunk text."""
    return hashlib.sha256(chunk_data.encode("utf-8")).hexdigest()


def generate_embedding(model: str, text: str):
    """Generate an embedding for the given text using the specified model.

    This keeps using the OpenAI embedding endpoint to produce the vector
    and returns a list/sequence of floats.
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Failed to generate embedding for chunk: {e}")
        sys.exit(1)


def _sanitize_collection_name(model: str) -> str:
    """Return a safe collection name for the given model string."""
    # Replace characters that may be problematic in collection names
    return "chunks_" + "".join([c if c.isalnum() else "_" for c in model])


def store_chunk(chunk_data: str, prompt: str, model: str):
    """Store a chunk in a Chroma collection keyed by model.

    Behavior:
    - Each embedding model gets its own Chroma collection (so storing a new
      model does not overwrite previous ones).
    - Document `id` is the SHA256 of the chunk text (matching previous logic).
    - Metadata stores the `prompt` and original `text` (chunk).
    - If an entry for the same id already exists in that collection, we skip
      insertion (preserves previous skip-on-existing-model behavior).
    """
    chunk_id = compute_chunk_id(chunk_data)
    # Generate embedding normally. If generate_embedding raises or if the
    # caller wants to skip embeddings (dry-run), this function may be
    # adjusted by the caller to pass None.
    embedding = generate_embedding(model, chunk_data)

    collection_name = _sanitize_collection_name(model)
    col = chroma_client.get_or_create_collection(name=collection_name)

    # Check if the id already exists in this model collection
    try:
        existing = col.get(ids=[chunk_id])
    except Exception:
        # Some Chroma backends may raise for unknown ids; treat as not found
        existing = {"ids": []}

    if existing and existing.get("ids"):
        print(f"[SKIP] Model '{model}' already has an embedding for this chunk.")
        return

    # Add the document with embedding and metadata
    # If embedding is None, add without specifying embeddings (Chroma will
    # accept the document+metadata but not a vector). Otherwise, add the
    # embedding vector as well.
    if embedding is None:
        col.add(
            ids=[chunk_id],
            documents=[chunk_data],
            metadatas=[{"prompt": prompt}],
        )
    else:
        col.add(
            ids=[chunk_id],
            documents=[chunk_data],
            metadatas=[{"prompt": prompt}],
            embeddings=[embedding]
        )
    print(f"[INSERT] Stored chunk in Chroma collection '{collection_name}' with id: {chunk_id}")


def ingest_yaml(file_path: str, model: str):
    """Read YAML file and process each chunk entry."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read YAML file: {e}")
        sys.exit(1)

    if not isinstance(chunks, list):
        print("[ERROR] YAML root must be a list of chunk entries.")
        sys.exit(1)

    for idx, item in enumerate(chunks, start=1):
        if not isinstance(item, dict) or "chunk_data" not in item or "prompt" not in item:
            print(f"[ERROR] Invalid YAML entry at index {idx}. Must have 'chunk_data' and 'prompt'.")
            sys.exit(1)

        print(f"Processing {idx}:")
        print("Chunk:", item["chunk_data"]) 
        print("Prompt:", item["prompt"]) 

        store_chunk(item["chunk_data"], item["prompt"], model)


def list_entries():
    """List all entries in the initialized Chroma client and print them.

    Prints nothing if no entries are found (per user requirement).
    """
    if chroma_client is None:
        print("[ERROR] Chroma client is not initialized.")
        return

    # Try several strategies to get collection names/objects
    collections = []
    # 1) public API
    try:
        cols = chroma_client.list_collections()
        for c in cols:
            # cols may be a list of dicts with 'name' or collection objects
            if hasattr(c, 'name'):
                collections.append(c.name)
            elif isinstance(c, dict) and 'name' in c:
                collections.append(c['name'])
    except Exception:
        pass

    # 2) try underlying client attribute (some versions expose _client)
    if not collections:
        try:
            underlying = getattr(chroma_client, '_client', None)
            if underlying is not None and hasattr(underlying, 'list_collections'):
                cols = underlying.list_collections()
                for c in cols:
                    if isinstance(c, dict) and 'name' in c:
                        collections.append(c['name'])
        except Exception:
            pass

    # 3) try to list known prefix collections by checking internal state or trying to get a list of names
    if not collections:
        # attempt to inspect protected attributes
        try:
            if hasattr(chroma_client, '_collections'):
                for k, v in getattr(chroma_client, '_collections').items():
                    collections.append(k)
        except Exception:
            pass

    # 4) As a last resort, try a set of likely collection names (no-op if they don't exist)
    if not collections:
        # collect a small set of guesses, user-created collections typically begin with 'chunks_'
        # but we don't know model names; try scanning a few common embeddings names
        guesses = [
            'chunks_text_embedding_3_small',
            'chunks_text-embedding-3-small',
        ]
        for g in guesses:
            try:
                chroma_client.get_collection(name=g)
                collections.append(g)
            except Exception:
                pass

    for name in collections:
        try:
            col = chroma_client.get_collection(name=name)
        except Exception:
            # skip if cannot get collection
            continue

        # Try to retrieve all entries
        try:
            contents = col.get()
        except Exception:
            # some implementations require explicit ids or pagination; skip
            continue

        ids = contents.get('ids', [])
        docs = contents.get('documents', [])
        metadatas = contents.get('metadatas', [])

        for i, doc_id in enumerate(ids):
            snippet = (docs[i][:200] + '...') if i < len(docs) and docs[i] and len(docs[i]) > 200 else (docs[i] if i < len(docs) else '')
            metadata = metadatas[i] if i < len(metadatas) else {}
            print(f"Collection: {name} | id: {doc_id} | metadata: {metadata} | snippet: {snippet}")



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

    # Initialize chroma with persistence directory
    init_chroma(args.chroma_dir)

    # If dry-run is requested, temporarily replace generate_embedding with
    # a stub that returns None so store_chunk will add documents without
    # embeddings.
    if args.dry_run:
        original_generate = globals().get("generate_embedding")
        globals()["generate_embedding"] = lambda model, text: None

    # If user only wants to list entries, do that and exit
    if args.list_entries:
        list_entries()
        # restore generate embedding if needed and exit
        if args.dry_run:
            globals()["generate_embedding"] = original_generate
        sys.exit(0)

    ingest_yaml(args.file, args.model)

    if args.dry_run:
        # restore original
        globals()["generate_embedding"] = original_generate
    # Try to persist the Chroma client if available (some clients support persist())
    try:
        if chroma_client is not None and hasattr(chroma_client, "persist"):
            chroma_client.persist()
            print(f"[INFO] Chroma persisted to {args.chroma_dir}")
    except Exception as e:
        print(f"[WARN] Failed to persist Chroma client: {e}")
