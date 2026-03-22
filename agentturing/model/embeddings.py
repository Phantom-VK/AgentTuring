"""Embedding model loader for local vector search."""

from langchain_huggingface import HuggingFaceEmbeddings

from agentturing.constants import EMBEDDING_MODEL_NAME


def get_embedder():
    """Create the embedding model used for Qdrant indexing and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )
