from langchain_huggingface import HuggingFaceEmbeddings

from agentturing.constants import EMBEDDING_MODEL_NAME


def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
