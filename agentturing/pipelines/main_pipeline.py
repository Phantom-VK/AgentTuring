from langgraph.graph import START, StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from agentturing.database.schemas import State
from agentturing.database.vectorstore import get_vectorstore
from agentturing.model.llm import get_llm
from agentturing.prompts import SYSTEM_PROMPT

vectorstore = get_vectorstore()
llm = get_llm()


prompt_template = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("user", "Context:\n{context}\n\nUser's Question:{question}\n\n\nAnswer101:")
])

def retrieve(state: State):
    docs = vectorstore.similarity_search(state["question"])
    return {"context": docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke({"context": docs_content, "question": state["question"]})
    response = llm.predict(prompt.to_string())
    return {"answer": response}

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()
