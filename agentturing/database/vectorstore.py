from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from agentturing.constants import QDRANT_PATH, COLLECTION_NAME
from agentturing.model.embeddings import get_embedder


def get_qdrant_client():
    print("Connecting to QDrant")
    return QdrantClient(path=QDRANT_PATH)

def get_vectorstore(embedder=None, client=None):
    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_qdrant_client()
    print("Creating vectorstore")

    return QdrantVectorStore(
        client=client,
        embedding=embedder,
        collection_name=COLLECTION_NAME
    )


