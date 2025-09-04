import os
from uuid import uuid4
from datasets import load_dataset
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from agentturing.constants import DATA_PATH, COLLECTION_NAME
from agentturing.database.vectorstore import get_qdrant_client, get_vectorstore
from agentturing.model.embeddings import get_embedder


def load_local_documents():
    """Load all .txt documents from DATA_PATH subdirectories."""
    all_docs = []
    for subdir in os.listdir(DATA_PATH):
        subdir_path = os.path.join(DATA_PATH, subdir)
        if os.path.isdir(subdir_path):
            loader = DirectoryLoader(subdir_path, glob="*.txt")
            docs = loader.load()
            all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} local documents")
    return all_docs


def load_dpo_dataset():
    """Load xinlai/Math-Step-DPO-10K dataset and convert to LangChain docs."""
    dpo = load_dataset("xinlai/Math-Step-DPO-10K", split="train")
    dpo_docs = []
    for row in dpo:
        content = (
                "\nQuestion:\n" + row.get("prompt", "") +
                "\nSteps:\n" + row.get("full_chosen", "") +
                "\nAnswer:\n" + row.get("answer", "")
        )
        dpo_docs.append(Document(page_content=content, metadata={"source": "dpo"}))
    print(f"Loaded {len(dpo_docs)} DPO dataset documents")
    return dpo_docs


def load_metamath_dataset():
    """Load MetaMathQA dataset and convert to LangChain docs."""
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    ds_docs = []
    for row in ds:
        content = (
                "\nQuestion:\n" + row.get("query", "") +
                "\nSteps:\n" + row.get("response", "")
        )
        ds_docs.append(Document(page_content=content, metadata={"source": "metamath"}))
    print(f"Loaded {len(ds_docs)} MetaMath dataset documents")
    return ds_docs


def create_chunks(documents, chunk_size=1000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} docs into {len(chunks)} chunks")
    return chunks


def ingest_into_qdrant(documents):

    vectorstore = get_vectorstore()
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vectorstore.add_documents(documents=documents, ids=uuids)
    print("Ingestion complete.")
    return vectorstore


def build_knowledge_base():
    print("Inside build_knowledge_base")
    local_docs = load_local_documents()
    dpo_docs = load_dpo_dataset()
    ds_docs = load_metamath_dataset()

    all_docs = local_docs + dpo_docs + ds_docs
    chunks = create_chunks(all_docs)

    vectorstore = ingest_into_qdrant(chunks)
    return vectorstore
