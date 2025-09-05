from uuid import uuid4

from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agentturing.database.vectorstore import get_vectorstore


def load_dpo_dataset():
    """Load xinlai/Math-Step-DPO-10K dataset and convert to LangChain docs."""
    print("Loading Math-Step-DPO-10K dataset...")
    dpo_dataset = load_dataset("xinlai/Math-Step-DPO-10K", split="train")

    dpo_documents = []
    for example in dpo_dataset:
        # Create a comprehensive document with problem and solution approach
        content = f"""
        Mathematical Problem: {example['prompt']}

        Solution Approach: {example['initial_reason_steps']}

        Step-by-Step Solution: {example['chosen']}

        Final Answer: {example['answer']}
        """

        doc = Document(
            page_content=content,
            metadata={
                "dataset": "Math-Step-DPO-10K",
                "problem_type": "step_by_step_solution",
                "has_reasoning_steps": True,
                "has_final_answer": True
            }
        )
        dpo_documents.append(doc)
    return dpo_documents


def load_metamath_dataset():
    """Load MetaMathQA dataset and convert to LangChain docs."""
    print("Loading MetaMathQA dataset...")
    mathqa_dataset = load_dataset("meta-math/MetaMathQA", split="train").select(range(9000))

    mathqa_documents = []
    for example in mathqa_dataset:
        content = f"""
        Mathematical Query: {example['query']}

        Detailed Solution: {example['response']}

        Problem Type: {example['type']}
        """

        doc = Document(
            page_content=content,
            metadata={
                "dataset": "MetaMathQA",
                "problem_type": example['type'],
                "has_original_question": 'original_question' in example
            }
        )
        mathqa_documents.append(doc)

        return mathqa_documents


def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(chunks)}")
    return chunks


def ingest_into_qdrant(documents):
    vectorstore = get_vectorstore()
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vectorstore.add_documents(documents=documents, ids=uuids)
    print("Ingestion complete.")
    return vectorstore


def build_knowledge_base():
    print("Inside build_knowledge_base")
    dpo_docs = load_dpo_dataset()
    ds_docs = load_metamath_dataset()

    all_docs = dpo_docs + ds_docs
    chunks = create_chunks(all_docs)

    vectorstore = ingest_into_qdrant(chunks)
    return vectorstore
