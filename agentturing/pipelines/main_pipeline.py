from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from agentturing.database.vectorstore import get_vectorstore
from agentturing.model.llm import get_llm
from agentturing.prompts import SYSTEM_PROMPT

class State(TypedDict):
    question: str
    context: List[str]
    answer: str

vectorstore = get_vectorstore()
llm = get_llm()
tavily = TavilySearch(
    max_results=3,
    topic="general",
    tavily_api_key="YOUR_TAVILY_API_KEY"
)

prompt_template = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT),
    ("user", "Context:\n{context}\n\nUser's Question: {question}\n\nAnswer101:")
])

def retrieve(state: State):
    docs = vectorstore.similarity_search(state["question"], k=3, score_threshold=0.75)
    return {"context": [doc.page_content for doc in docs]}

def tavily_search(state: State):
    results = tavily.invoke(state["question"])
    web_docs = results['results']
    print(web_docs)
    existing_context = state.get("context", [])
    new_context = existing_context + web_docs
    return {"context": new_context}

def generate(state: State):
    docs_content = "\n\n".join(state.get("context", []))
    prompt = prompt_template.invoke({"context": docs_content, "question": state["question"]})
    response = llm.predict(prompt.to_string())
    return {"answer": response}

def route(state: State):
    # If retrieval produced fewer than 2 docs, call web search
    return "tavily" if len(state.get("context", [])) < 3 else "generate"

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("tavily", tavily_search)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_conditional_edges("retrieve", route, {"tavily": "tavily", "generate": "generate"})
    graph_builder.add_edge("tavily", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()
