import hashlib
import sys
import chromadb
import logging
from embedding_handler import EmbeddingHandler

try:
    from chromadb.config import Settings
except Exception:
    Settings = None


class DBHandler:
    """Encapsulate Chroma client and storage helpers."""

    def __init__(self, persist_directory: str = "db", embedding_model: str = "nomic-embed-text"):
        # instance logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

        self.embedding_handler = EmbeddingHandler(embedding_model)
        self.client = None
        # Try a few initialization strategies to support multiple chromadb versions
        try:
            # Newer chroma may expose PersistentClient
            if hasattr(chromadb, 'PersistentClient'):
                self.client = chromadb.PersistentClient(path=persist_directory)
                return
        except Exception:
            pass

        try:
            if Settings is not None:
                settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
                self.client = chromadb.Client(settings=settings)
                return
        except Exception:
            pass

        # Fallback
        try:
            self.client = chromadb.Client()
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma client: {e}")
            sys.exit(1)

    @staticmethod
    def compute_chunk_id(chunk_data: str) -> str:
        return hashlib.sha256(chunk_data.encode("utf-8")).hexdigest()

    def generate_embedding(self, model: str, text: str):
        """Delegate embedding generation to EmbeddingHandler."""
        return self.embedding_handler.generate_embedding(text, model=model)

    @staticmethod
    def _sanitize_collection_name(model: str) -> str:
        return "chunks_" + "".join([c if c.isalnum() else "_" for c in model])

    def store_chunk(self, chunk_data: str, prompt: str, model: str):
        chunk_id = self.compute_chunk_id(chunk_data)
        embedding = self.generate_embedding(model, chunk_data)

        collection_name = self._sanitize_collection_name(model)
        col = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Chunks with embeddings from model {model}"}
        )

        try:
            existing = col.get(ids=[chunk_id])
        except Exception:
            existing = {"ids": []}

        if existing and existing.get("ids"):
            self.logger.info(f"[SKIP] Model '{model}' already has an embedding for this chunk.")
            return

        if embedding is None:
            col.add(ids=[chunk_id], documents=[chunk_data], metadatas=[{"prompt": prompt}])
        else:
            col.add(ids=[chunk_id], documents=[chunk_data], metadatas=[{"prompt": prompt}], embeddings=[embedding])
        # Log insertion
        try:
            self.logger.info(f"[INSERT] Stored chunk in Chroma collection '{collection_name}' with id: {chunk_id}")
        except Exception:
            # Ensure logging failure doesn't break storage flow
            pass

    def list_entries(self):
        if self.client is None:
            self.logger.error("Chroma client is not initialized.")
            return

        collections = []
        try:
            cols = self.client.list_collections()
            for c in cols:
                if hasattr(c, 'name'):
                    collections.append(c.name)
                elif isinstance(c, dict) and 'name' in c:
                    collections.append(c['name'])
        except Exception:
            pass

        if not collections:
            try:
                underlying = getattr(self.client, '_client', None)
                if underlying is not None and hasattr(underlying, 'list_collections'):
                    cols = underlying.list_collections()
                    for c in cols:
                        if isinstance(c, dict) and 'name' in c:
                            collections.append(c['name'])
            except Exception:
                pass

        if not collections:
            try:
                if hasattr(self.client, '_collections'):
                    for k in getattr(self.client, '_collections'):
                        collections.append(k)
            except Exception:
                pass

        for name in collections:
            try:
                col = self.client.get_collection(name=name)
            except Exception:
                continue

            try:
                contents = col.get()
            except Exception:
                continue

            ids = contents.get('ids', [])
            docs = contents.get('documents', [])
            metadatas = contents.get('metadatas', [])

            for i, doc_id in enumerate(ids):
                snippet = (docs[i][:200] + '...') if i < len(docs) and docs[i] and len(docs[i]) > 200 else (docs[i] if i < len(docs) else '')
                metadata = metadatas[i] if i < len(metadatas) else {}
                self.logger.info(f"Collection: {name} | id: {doc_id} | metadata: {metadata} | snippet: {snippet}")

    def persist(self):
        try:
            if self.client is not None and hasattr(self.client, 'persist'):
                self.client.persist()
                return True
        except Exception:
            pass
        return False

    def query(self, query_text: str, model: str, top_k: int = 5, distance_threshold: float = 0.1):
        """Run a vector search for `query_text` against the Chroma collection for `model`.

        This implementation assumes the documented API: get the collection by name
        and call `collection.query(query_embeddings=[embedding], n_results=top_k)`.
        
        Returns a list of hits filtered by both top_k and distance_threshold.
        Results are returned in order of increasing distance (closest first).
        A result is included only if:
        1. Its distance is <= distance_threshold (lower distance = higher similarity)
        2. It is within the top_k results
        
        Args:
            query_text: The text to search for.
            model: The embedding model to use.
            top_k: Maximum number of results to retrieve from Chroma (default: 5).
            distance_threshold: Maximum allowed distance (default: 0.1). Results with
                              distance > threshold are discarded.
        
        Returns:
            A list of hits with keys: collection, id, document, metadata, distance, embedding.
            Only includes hits where distance <= distance_threshold.
        """
        if self.client is None:
            raise RuntimeError("Chroma client is not initialized")

        q_embedding = self.generate_embedding(model, query_text)

        collection_name = self._sanitize_collection_name(model)
        col = self.client.get_collection(name=collection_name)

        # Request all useful return fields in one call per docs: documents, metadatas,
        # embeddings and distances.
        res = col.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            include=["embeddings", "distances", "documents", "metadatas"],
        )

        hits = []
        all_ids = res.get('ids', []) if isinstance(res, dict) else []
        all_docs = res.get('documents', []) if isinstance(res, dict) else []
        all_metadatas = res.get('metadatas', []) if isinstance(res, dict) else []
        all_distances = res.get('distances', []) if isinstance(res, dict) else []
        all_embeddings = res.get('embeddings', []) if isinstance(res, dict) else []
        ids = all_ids[0] if all_ids else []
        docs = all_docs[0] if all_docs else []
        metadatas = all_metadatas[0] if all_metadatas else []
        distances = all_distances[0] if all_distances else []
        embeddings = all_embeddings[0] if all_embeddings else []

        for i, doc_id in enumerate(ids):
            # Get the distance for this result
            distance = distances[i] if i < len(distances) else float('inf')
            
            # Filter by distance threshold: only include if distance <= threshold
            if distance <= distance_threshold:
                hit = {
                    'collection': collection_name,
                    'id': doc_id,
                    'document': docs[i] if i < len(docs) else None,
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'distance': distance,
                    'embedding': embeddings[i] if i < len(embeddings) else None,
                }
                hits.append(hit)
            else:
                # Log when first entry is skipped due to distance threshold
                if i == 0:
                    doc_snippet = (docs[i][:100] + '...') if i < len(docs) and docs[i] and len(docs[i]) > 100 else (docs[i] if i < len(docs) else 'N/A')
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    try:
                        self.logger.info(f"Skipping entry: distance={distance}, threshold={distance_threshold}, metadata={metadata}, document={doc_snippet}")
                    except Exception:
                        pass
        
        try:
            self.logger.info(f"Returning {len(hits)} hits from query (distance_threshold={distance_threshold}, top_k={top_k}).")
        except Exception:
            pass

        return hits
