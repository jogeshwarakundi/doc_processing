import logging
import openai
import requests


class EmbeddingHandler:
    """Encapsulate embedding generation from multiple backends (OpenAI and local Ollama)."""

    def __init__(self, model: str = "nomic-embed-text"):
        """Initialize the EmbeddingHandler with a default model.

        Args:
            model: The embedding model to use (default: nomic-embed-text).
                   Supported models:
                   - "nomic-embed-text": Uses local Ollama (default, free)
                   - "text-embedding-3-small": Uses OpenAI API (costs money)
        """
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ollama_base_url = "http://localhost:11434"

    def generate_embedding(self, text: str, model: str = None) -> list:
        """Generate an embedding vector for the given text.

        Routes to the appropriate backend (OpenAI or local Ollama) based on model name.

        Args:
            text: The text to embed.
            model: The embedding model to use. If None, uses self.model.

        Returns:
            A list representing the embedding vector.

        Raises:
            Exception: If embedding generation fails.
        """
        embedding_model = model or self.model

        if embedding_model == "nomic-embed-text":
            return self.generate_local_embeddings(text, embedding_model)
        else:
            # Default to OpenAI for text-embedding-3-small and other OpenAI models
            return self.generate_openai_embeddings(text, embedding_model)

    def generate_openai_embeddings(self, text: str, model: str) -> list:
        """Generate embedding using OpenAI API.

        Args:
            text: The text to embed.
            model: The OpenAI embedding model to use.

        Returns:
            A list representing the embedding vector.

        Raises:
            Exception: If OpenAI API call fails.
        """
        try:
            response = openai.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate OpenAI embedding for chunk: {e}")
            raise

    def generate_local_embeddings(self, text: str, model: str) -> list:
        """Generate embedding using local Ollama model.

        Args:
            text: The text to embed.
            model: The Ollama embedding model to use (e.g., "nomic-embed-text").

        Returns:
            A list representing the embedding vector.

        Raises:
            Exception: If Ollama API call fails.
        """
        try:
            # Call the Ollama /api/embed endpoint
            url = f"{self.ollama_base_url}/api/embed"
            payload = {
                "model": model,
                "input": text
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embeddings = data.get("embeddings")
            
            if not embeddings or len(embeddings) == 0:
                raise ValueError("No embeddings returned from Ollama")
            
            # Return the first embedding (for single input)
            return embeddings[0]
        except Exception as e:
            self.logger.error(f"Failed to generate local embedding (Ollama) for chunk: {e}")
            raise
