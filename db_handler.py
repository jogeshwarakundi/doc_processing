import hashlib
import sys
import openai
import chromadb

try:
    from chromadb.config import Settings
except Exception:
    Settings = None


class DBHandler:
    """Encapsulate Chroma client and embedding/storage helpers."""

    def __init__(self, persist_directory: str = "db"):
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
            print(f"[ERROR] Failed to initialize Chroma client: {e}")
            sys.exit(1)

    @staticmethod
    def compute_chunk_id(chunk_data: str) -> str:
        return hashlib.sha256(chunk_data.encode("utf-8")).hexdigest()

    def generate_embedding(self, model: str, text: str):
        try:
            response = openai.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding for chunk: {e}")
            raise

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
            print(f"[SKIP] Model '{model}' already has an embedding for this chunk.")
            return

        if embedding is None:
            col.add(ids=[chunk_id], documents=[chunk_data], metadatas=[{"prompt": prompt}])
        else:
            col.add(ids=[chunk_id], documents=[chunk_data], metadatas=[{"prompt": prompt}], embeddings=[embedding])

        print(f"[INSERT] Stored chunk in Chroma collection '{collection_name}' with id: {chunk_id}")

    def list_entries(self):
        if self.client is None:
            print("[ERROR] Chroma client is not initialized.")
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
                print(f"Collection: {name} | id: {doc_id} | metadata: {metadata} | snippet: {snippet}")

    def persist(self):
        try:
            if self.client is not None and hasattr(self.client, 'persist'):
                self.client.persist()
                return True
        except Exception:
            pass
        return False
